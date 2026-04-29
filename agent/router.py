"""
router.py — API routes
  GET  /api/wines        → full wine list for the grid
  POST /api/ask-wine     → barkeep agent with selected wine + memory
  WS   /api/ws/chat      → true streaming: LLM tokens → ElevenLabs WS → browser audio
"""
import json
import asyncio
import os

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel
from dotenv import load_dotenv

from agent.agent import barkeep, sanitize_for_json
from agent.data import get_schema, query_wines, get_dataframe, get_similar_wines

load_dotenv()

router = APIRouter()

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = "pNInz6obpgDQGcFmaJgB"


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.get("/wines")
def get_wines():
    df = get_dataframe()
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded yet")
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    for i, r in enumerate(records):
        if "id" not in r or r["id"] is None:
            r["id"] = i
        if isinstance(r.get("professional_ratings"), str):
            try:
                r["professional_ratings"] = json.loads(r["professional_ratings"])
            except Exception:
                r["professional_ratings"] = []
    return sanitize_for_json(records)


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(req: AskRequest):
    """REST fallback — collects the full streaming response into one string."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    tokens = [token async for token in barkeep(req.question.strip())]
    return {"answer": "".join(tokens), "question": req.question}


# ── WebSocket — true streaming via ElevenLabs WS input API ──────────────────

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            req  = json.loads(data)

            wine     = req.get('wine')            # None → browsing mode
            question = req.get('question', '').strip()
            history  = req.get('history', [])

            # ── One agent handles everything ──────────────────────────────────
            # barkeep() yields either:
            #   str  → LLM token, forward as text_chunk
            #   dict → side-channel event (e.g. rec_wines), forward as-is
            full_text = []
            async for event in barkeep(question, wine=wine, history=history):
                if isinstance(event, dict):
                    # Side-channel event from a tool call — forward directly to browser
                    await websocket.send_json(sanitize_for_json(event))
                else:
                    await websocket.send_json({"type": "text_chunk", "content": event})
                    full_text.append(event)

            complete_text = "".join(full_text).strip()
            if not complete_text:
                await websocket.send_json({"type": "audio_done"})
                continue

            # 2. Single REST call to ElevenLabs TTS — far cheaper than streaming WS
            #    (~1 credit/char vs ~100 credit minimum for stream-input)
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
            headers = {
                "xi-api-key": ELEVEN_LABS_API_KEY,
                "Content-Type": "application/json",
            }
            body = {
                "text": complete_text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }

            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(tts_url, headers=headers, json=body)
                if resp.status_code == 200:
                    # Send in 32KB chunks — single large WS frame gets dropped/truncated
                    audio = resp.content
                    chunk_size = 32 * 1024
                    for i in range(0, len(audio), chunk_size):
                        await websocket.send_bytes(audio[i:i+chunk_size])
                else:
                    print(f"ElevenLabs TTS error: {resp.status_code} {resp.text}")

            await websocket.send_json({"type": "audio_done"})

    except Exception as e:
        print(f"WebSocket Error: {e}")


# ── Content-based recommendation endpoint ────────────────────────────────────

@router.get("/similar/{wine_id}")
def similar_wines(wine_id: int, top_k: int = 6):
    """
    Return top_k wines most similar to wine_id by cosine similarity.
    Similarity is computed at startup over TF-IDF text + normalised numerics.
    """
    results = get_similar_wines(wine_id, top_k)
    if results is None:
        raise HTTPException(status_code=503, detail="Similarity index not ready")
    # Parse professional_ratings JSON strings (same as /wines endpoint)
    for r in results:
        if isinstance(r.get("professional_ratings"), str):
            try:
                r["professional_ratings"] = json.loads(r["professional_ratings"])
            except Exception:
                r["professional_ratings"] = []
    return sanitize_for_json(results)


# ── Debug endpoints ──────────────────────────────────────────────────────────

@router.get("/schema")
def schema():
    return sanitize_for_json(get_schema())

@router.get("/sample")
def sample():
    return query_wines(limit=5)

@router.get("/health")
def health():
    return {"status": "ok"}