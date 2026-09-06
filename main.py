"""Uvicorn entrypoint used by Render / production."""
import os

from app.web.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)