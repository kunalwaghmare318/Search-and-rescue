"""
Entry point for Render deployment (aliases serve.py)
"""
import os
import uvicorn
from serve import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
