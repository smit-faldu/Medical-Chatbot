"""
FastAPI Application
Web API for the Medical Chatbot.
"""
import os
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Fix OpenMP conflict before importing other modules
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .flow import MedicalChatFlow

# Pydantic models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    explanation: str
    key_points: List[str]
    subject: str
    confidence: float
    source: str
    is_fallback: bool

# Initialize FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    description="AI chatbot for medical questions using MedMCQA dataset",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable for chat flow
chat_flow = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chat flow on startup."""
    global chat_flow
    try:
        print("🚀 Initializing Medical Chat Flow...")
        print("💡 Using existing Pinecone index with pre-loaded data")
        chat_flow = MedicalChatFlow(
            data_dir="data",
            confidence_threshold=0.3
        )
        print("✅ Medical Chat Flow initialized successfully")
        try:
            stats = chat_flow.get_flow_stats()
            print(f"📊 Data loaded: {stats.get('total_documents', 'Unknown')} documents")
        except:
            print("📊 Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chat flow: {str(e)}")
        print("Please check the error above and try restarting the server.")

@app.get("/")
async def root():
    """Redirect to the chat UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")

@app.get("/ui")
async def chat_ui():
    """Serve the chat UI."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/health")
async def health_check():
    global chat_flow
    if chat_flow is None:
        return {"status": "unhealthy", "message": "Chat flow not initialized"}
    return {"status": "healthy", "message": "Medical chatbot is ready"}

@app.get("/chat")
async def chat_info():
    return {
        "message": "Medical Chat API - Use POST method to ask questions",
        "method": "POST",
        "endpoint": "/chat",
        "content_type": "application/json",
        "body_format": {"question": "Your medical question here"},
        "example_request": {"question": "What is hypertension and what causes it?"},
        "web_ui": "Visit /ui for the interactive chat interface",
        "api_docs": "Visit /docs for complete API documentation"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global chat_flow
    if chat_flow is None:
        raise HTTPException(status_code=503, detail="Chat flow not initialized. Please check server logs.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        response = chat_flow.chat(request.question.strip())
        return ChatResponse(
            question=response["question"],
            answer=response["answer"],
            explanation=response["explanation"],
            key_points=response["key_points"],
            subject=response["subject"],
            confidence=response["confidence"],
            source=response["source"],
            is_fallback=response["is_fallback"]
        )
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return ChatResponse(
            question=request.question,
            answer="I'm not confident enough to answer that based on my current knowledge base.",
            explanation="An error occurred while processing your question.",
            key_points=["System error", "Please try again"],
            subject="Error",
            confidence=0.0,
            source="Error",
            is_fallback=True
        )

@app.get("/stats")
async def get_stats():
    global chat_flow
    if chat_flow is None:
        raise HTTPException(status_code=503, detail="Chat flow not initialized")
    try:
        stats = chat_flow.get_flow_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}") 