"""
Wine Voice Explorer — FastAPI backend
Run: cd wine-explorer && uvicorn main:app --reload --port 8000
"""
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from agent.router import router as agent_router
from agent.data import load_dataset, build_similarity_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dataset()
    print("✅ Wine dataset loaded")
    build_similarity_index()
    yield

app = FastAPI(title="Wine Voice Explorer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(agent_router, prefix="/api")

# Serve static assets — no caching so changes are always picked up
app.mount("/static", StaticFiles(directory="static"), name="static")

def nocache(path: str) -> FileResponse:
    return FileResponse(path, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    })

# Landing page — entry point
@app.get("/")
def landing():
    return nocache("static/landing.html")

# Main cellar app
@app.get("/cellar")
def cellar():
    return nocache("static/cellar.html")