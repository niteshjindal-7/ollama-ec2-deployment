import os
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(override=True)

app = FastAPI(title="Ollama API")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

class PromptRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL


@app.get("/")
def home():
    return {
        "message": "Ollama API is running",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "generate": "/generate"
        }
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as Client:
            response = await Client.get(f"{OLLAMA_BASE_URL}/api/tags")

        return {
            "status": "healthy",
            "models": response.json()["models"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
    

@app.get("/models")
async def models():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
    return response.json()


@app.post("/generate-gemini")
async def generate_gemini(request: PromptRequest):
    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": request.prompt}]}]}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, json=payload, headers=headers)
    data = response.json()
    if "candidates" not in data:
        return {"model": GEMINI_MODEL, "raw_response": data}
    return {
        "model": GEMINI_MODEL,
        "response": data["candidates"][0]["content"]["parts"][0]["text"]
    }


@app.post("/generate")
async def generate(request: PromptRequest):
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json = payload
        )
        
    return response.json()