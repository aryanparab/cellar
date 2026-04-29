import os
import json
import math
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from agent.data import query_wines, get_schema, get_similar_wines, get_wine_id_by_name
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1,   # low temp → consistent tool extraction + factual answers
)

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def filter_wines_tool(
    max_price: float = None,
    min_price: float = None,
    min_rating: float = None,
    region_contains: str = None,
    country_contains: str = None,
    variety_contains: str = None,
    name_contains: str = None,
    sort_by: str = "rating",
    sort_desc: bool = True,
    limit: int = 5,
) -> str:
    """Query the wine dataset with filters for price, rating, region, country, or variety."""
    results = query_wines(
        max_price=max_price, min_price=min_price, min_rating=min_rating,
        region_contains=region_contains, country_contains=country_contains,
        variety_contains=variety_contains,
        name_contains=name_contains, sort_by=sort_by, sort_desc=sort_desc, limit=limit,
    )
    if not results:
        return "No wines matched your criteria."
    return json.dumps(results)


@tool
def get_wine_schema_tool() -> str:
    """Returns the column names and sample data for the wine dataset."""
    return json.dumps(get_schema())


@tool
def recommend_similar_wines_tool(wine_name: str, top_k: int = 5) -> str:
    """
    Get wines most similar to a given wine using content-based similarity.
    Call this when the user asks for recommendations based on a specific wine.
    Pass the wine's full name (e.g. "La Crema Chardonnay Monterey") — the tool
    will resolve the name to an ID automatically.
    Can be called multiple times with different wine names — e.g. once per wine
    on the tasting bench if the user wants recommendations for several wines.
    Returns a ranked list of similar wines with full details.
    """
    wine_id = get_wine_id_by_name(wine_name)
    if wine_id is None:
        return f"Could not find wine '{wine_name}' in the catalog."
    results = get_similar_wines(wine_id=wine_id, top_k=top_k)
    if not results:
        return f"No similar wines found for '{wine_name}'."
    return json.dumps({"wine_id": wine_id, "wines": results})


TOOLS          = [filter_wines_tool, get_wine_schema_tool, recommend_similar_wines_tool]
llm_with_tools = llm.bind_tools(TOOLS)

# ── System prompt ─────────────────────────────────────────────────────────────
# One persona for everything. Wine context is injected below when a wine is
# present — the barkeep's character doesn't change based on what's in front of them.
BARKEEP_SYSTEM = (
    "You are a warm, knowledgeable barkeep with a deep knowledge of wine. "
    "You have access to a wine catalog and can look up real wines using your tools. "
    "Keep replies to 2–3 sentences. Be conversational, never stiff. "
    "Only state facts you can verify from the wine data or your tools — never guess. "
    "If the guest asks something you can't answer, say so honestly and offer to look it up."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_wine_context(wine: Dict[str, Any]) -> str:
    """
    Build a rich, readable wine context string from the full wine dict.
    Only includes fields that actually have a value so the prompt stays clean.
    """
    fields = [
        ("Name",            wine.get("name")),
        ("Producer",        wine.get("producer")),
        ("Varietal",        wine.get("varietal")),
        ("Color",           wine.get("color")),
        ("Vintage",         wine.get("vintage")),
        ("Region",          wine.get("region")),
        ("Appellation",     wine.get("appellation")),
        ("Country",         wine.get("country")),
        ("Price",           f"${wine.get('retail')}" if wine.get("retail") else None),
        ("ABV",             f"{wine.get('abv')}%" if wine.get("abv") else None),
        ("Volume",          f"{wine.get('volume_ml')} ml" if wine.get("volume_ml") else None),
    ]
  
    ratings = wine.get("professional_ratings") or []
    if isinstance(ratings, list) and ratings:
        rating_str = ", ".join(
            f"{r.get('score')} ({r.get('source')})" for r in ratings if r.get("score")
        )
        if rating_str:
            fields.append(("Ratings", rating_str))

    return "\n".join(f"{label}: {val}" for label, val in fields if val)


def sanitize_for_json(obj):
    """Recursively convert NaN/Inf to None so JSON encoder doesn't crash."""
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


_TOOL_MAP = {
    "filter_wines_tool":              filter_wines_tool,
    "get_wine_schema_tool":           get_wine_schema_tool,
    "recommend_similar_wines_tool":   recommend_similar_wines_tool,
}

# ── Unified barkeep — one agent for everything ────────────────────────────────
async def barkeep(
    question: str,
    wine:     Dict[str, Any]      | None = None,
    history:  List[Dict[str, str]]       = [],
):
    """
    Single async-generator agent for every conversation turn.

    Flow:
      1. Build system prompt — wine context injected if present, plain
         barkeep persona otherwise (browsing mode).
      2. llm_with_tools.invoke() — one round-trip to check if tools needed.
         • No tool calls → yield ai_msg.content directly and return.
           The answer is already in ai_msg; a second LLM call would see a
           complete conversation and return empty (that was the original bug).
         • Tool calls → resolve them, then stream the synthesised final answer.
      3. (Tool path only) resolve each tool call in executor, append results.
      4. (Tool path only) llm.astream(messages) → yield tokens one by one.
    """
    import asyncio
    loop = asyncio.get_event_loop()

    # ── 1. Build system prompt ────────────────────────────────────────────────
    # Same barkeep persona always. Wine context appended when a wine is present.
    system = BARKEEP_SYSTEM
    if wine:
        wine_ctx = _build_wine_context(wine)
        system   = f"{BARKEEP_SYSTEM}\n\nWine on the counter:\n{wine_ctx}"

    lc_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]
    messages = [SystemMessage(content=system), *lc_history, HumanMessage(content=question)]

    # ── 2. Tool-check ────────────────────────────────────────────────────────
    # invoke() with tools bound: LLM either returns content OR tool_calls.
    # Run in executor so we don't block the async event loop.
    ai_msg = await loop.run_in_executor(None, llm_with_tools.invoke, messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        # ── No tools needed — first invoke already has the full answer ────────
        # Do NOT call llm.astream(messages) here: messages now ends with the
        # AI reply, so a second LLM call sees a complete conversation and
        # returns empty. Just yield the content we already have.
        if ai_msg.content:
            yield ai_msg.content
        return

    # ── 3. Resolve tool calls ─────────────────────────────────────────────────
    for call in ai_msg.tool_calls:
        fn = _TOOL_MAP.get(call["name"])
        if not fn:
            messages.append(ToolMessage(content="Unknown tool.", tool_call_id=call["id"]))
            continue
        # Strip None/null — Groq rejects explicit nulls for optional params
        args   = {k: v for k, v in call["args"].items() if v is not None}
        output = await loop.run_in_executor(None, fn.invoke, args)
        messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        # When the LLM calls recommend_similar_wines_tool, emit a rec_wines event
        # so the router can forward the exact wines to the browser immediately.
        # Yielding a dict (not a str) signals the router to send it as its own
        # WS message rather than as a text_chunk.
        if call["name"] == "recommend_similar_wines_tool":
            try:
                payload = json.loads(output)
                # output is {"wine_id": int, "wines": [...]}
                yield {
                    "type":           "rec_wines",
                    "source_wine_id": payload.get("wine_id"),
                    "wines":          payload.get("wines", []),
                }
            except (json.JSONDecodeError, TypeError):
                pass

    # ── 4. Stream final answer (only reached when tools were used) ────────────
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content