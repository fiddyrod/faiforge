import uvicorn
from core.config import load_config

if __name__ == "__main__":
     # Load config
    config = load_config()

    print("🔥 Starting FAIForge API...")
    print(f"📝 API docs: http://{config.api.host}:{config.api.port}/docs")
    print(f"❤️  Health check: http://{config.api.host}:{config.api.port}/health")
    print(f"🔄 Auto-reload: {config.api.reload}")
    
    uvicorn.run(
        "main:app",
        host=config.api.host,      
        port=config.api.port,      
        reload=config.api.reload   
    )