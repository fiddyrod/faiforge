import uvicorn

if __name__ == "__main__":
    print("🔥 Starting FAIForge API...")
    print("📝 API docs: http://localhost:8000/docs")
    print("❤️  Health check: http://localhost:8000/health")
    print("🔄 Auto-reload enabled (dev mode)")
    
    uvicorn.run(
        "main:app",  # ← Import string format
        host="0.0.0.0",
        port=8000,
        reload=True  # ← Enable auto-reload on code changes
    )