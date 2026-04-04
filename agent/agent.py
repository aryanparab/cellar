import os
import json
from typing import List, Dict, Any
import math
# LangChain & Groq Import
# s
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from elevenlabs.client import ElevenLabs,AsyncElevenLabs


from agent.data import query_wines, get_schema
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = "pNInz6obpgDQGcFmaJgB" 
GROQ_MODEL = "llama-3.3-70b-versatile"
el_client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

# Initialize the Groq Chat Model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.1 # Low temperature for consistent tool extraction
)

#

@tool
def filter_wines_tool(
    max_price: float = None,
    min_price: float = None,
    min_rating: float = None,
    region_contains: str = None,
    variety_contains: str = None,
    name_contains: str = None,
    sort_by: str = "rating",
    sort_desc: bool = True,
    limit: int = 5
) -> str:
    """Query the wine dataset with filters for price, rating, region, or variety."""
    results = query_wines(
        max_price=max_price, min_price=min_price, min_rating=min_rating,
        region_contains=region_contains, variety_contains=variety_contains,
        name_contains=name_contains, sort_by=sort_by, sort_desc=sort_desc, limit=limit
    )
    if not results:
        return "No wines matched your criteria."
    return json.dumps(results)

@tool
def get_wine_schema_tool() -> str:
    """Returns the column names and sample data for the wine dataset."""
    return json.dumps(get_schema())

TOOLS = [filter_wines_tool, get_wine_schema_tool]
llm_with_tools = llm.bind_tools(TOOLS)



GLOBAL_SYSTEM = "You are a professional sommelier. Answer strictly using tool data in 2-4 sentences."
BARKEEP_SYSTEM = "You are a warm, knowledgeable barkeep. Limit to 3 sentences. Be friendly."


def sanitize_for_json(obj):
    """Recursively convert NaN/Inf to None so JSON encoder doesn't crash."""
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj



# async def stream_barkeep_voice(question: str, wine_context: dict, websocket):
#     """
#     Streams text from Groq and pipes it to ElevenLabs for real-time audio.
#     """
    
#     # 1. Start Groq Stream
#     instruction = f"You are a barkeep. Use this wine data: {json.dumps(wine_context)}"
#     groq_stream = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": instruction},
#             {"role": "user", "content": question}
#         ],
#         stream=True,
#     )

#     # 2. Setup ElevenLabs WebSocket for Input-to-Speech
#     # Note: Using the ElevenLabs WebSocket directly allows for 'barge-in' 
#     # and lower latency than their standard REST API.
    
#     async def text_iterator():
#         for chunk in groq_stream:
#             delta = chunk.choices[0].delta.content
#             if delta:
#                 # Send text to frontend for the chat bubble
#                 await websocket.send_json({"type": "text", "content": delta})
#                 yield delta

#     # 3. Stream Audio to Frontend
#     # ElevenLabs 'generate' with stream=True returns a generator of audio bytes
#     audio_stream = el_client.generate(
#         text=text_iterator(),
#         voice=VOICE_ID,
#         model="eleven_turbo_v2", # Fastest model for low-latency
#         stream=True
#     )

#     for chunk in audio_stream:
#         if chunk:
#             # Send binary audio data to frontend
#             await websocket.send_bytes(chunk)
async def stream_barkeep_voice(question: str, wine_context: dict, websocket):
    """
    Asynchronously streams text from Groq and pipes it to ElevenLabs for real-time audio.
    """
    
    # 1. Create the async text generator
    async def text_iterator():
        # Define the personality and context
        prompt = f"System: {BARKEEP_SYSTEM}\nContext: {json.dumps(wine_context)}\nUser: {question}"
        
        # Use LangChain's astream to get tokens as they are generated
        async for chunk in llm.astream(prompt):
            if chunk.content:
                # Send text to the UI chat bubble immediately
                await websocket.send_json({"type": "text_chunk", "content": chunk.content})
                yield chunk.content

    # 2. Pipe the async iterator into the ElevenLabs Async stream
    # 'model_id' and 'voice_id' are the required v1.x parameters
    audio_stream = await el_client.text_to_speech.stream(
        text=text_iterator(),
        voice_id=VOICE_ID,
        model_id="eleven_turbo_v2" # Optimized for low-latency speed
    )

    # 3. Stream the binary audio chunks to the frontend WebSocket
    async for audio_chunk in audio_stream:
        if audio_chunk:
            await websocket.send_bytes(audio_chunk)
def ask_agent(question: str) -> str:
    messages = [SystemMessage(content=GLOBAL_SYSTEM), HumanMessage(content=question)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"filter_wines_tool": filter_wines_tool, "get_wine_schema_tool": get_wine_schema_tool}[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        return llm.invoke(messages).content
    return ai_msg.content

def ask_barkeep(question: str, wine: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    wine_context = f"Wine: {wine.get('name')} from {wine.get('region')}. Price: ${wine.get('retail')}."
    lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in history[:-1]]
    messages = [SystemMessage(content=f"{BARKEEP_SYSTEM}\n\n{wine_context}"), *lc_history, HumanMessage(content=question)]
    return llm.invoke(messages).content