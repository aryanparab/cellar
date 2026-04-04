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

from agent.agent import ask_agent, ask_barkeep, sanitize_for_json, llm, BARKEEP_SYSTEM
from agent.data import get_schema, query_wines, get_dataframe

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


class Message(BaseModel):
    role: str
    content: str


class AskWineRequest(BaseModel):
    question: str
    wine: dict
    history: list[Message]


@router.post("/ask-wine")
async def ask_wine(req: AskWineRequest):
    clean_wine = sanitize_for_json(req.wine)
    history = [{"role": m.role, "content": m.content} for m in req.history]
    answer = ask_barkeep(question=req.question.strip(), wine=clean_wine, history=history)
    return {"answer": answer}


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    answer = ask_agent(req.question.strip())
    return {"answer": answer, "question": req.question}


# ── WebSocket — true streaming via ElevenLabs WS input API ──────────────────

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)

            prompt = (
                f"System: {BARKEEP_SYSTEM}\n"
                f"Context: {json.dumps(req.get('wine'))}\n"
                f"Q: {req.get('question')}"
            )

            # 1. Stream LLM tokens to browser for the chat bubble,
            #    accumulate full text for TTS
            full_text = []
            async for chunk in llm.astream(prompt):
                if chunk.content:
                    await websocket.send_json(
                        {"type": "text_chunk", "content": chunk.content}
                    )
                    full_text.append(chunk.content)

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