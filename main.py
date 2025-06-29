"""
Medical Chatbot - NexgAI AI Engineering Challenge
Main entry point for the application.
"""
import os
import uvicorn

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    """Main function to run the FastAPI server."""
    print("🏥 Starting Medical Chatbot...")
    print("💡 Using existing Pinecone index with pre-loaded data")
    print("🚀 Server will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation: http://127.0.0.1:8000/docs")
    print("❤️  Health Check: http://127.0.0.1:8000/health")
    
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main() 