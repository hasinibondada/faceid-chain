"""FastAPI web service + simple UI for the face->search->blockchain pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.blockchain.chain import get_chain
from app.pipeline.pipeline import run_pipeline

app = FastAPI(title="FaceID Chain", version="1.0.0",
              description="Face scan -> web/social search -> blockchain verification")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    chain = get_chain()
    return {"status": "ok", "chain": chain.status()}


@app.get("/api/chain")
def chain_status():
    chain = get_chain()
    return {
        "status": chain.status(),
        "tamper": chain.verify_tamper(),
        "blocks": chain.blocks,
    }


@app.post("/api/pipeline")
async def pipeline(
    image: UploadFile = File(...),
    query: Optional[str] = Form(None),
    platform: Optional[str] = Form(None),
    max_checks: int = Form(4),
):
    data = await image.read()
    result = run_pipeline(data, query=query, platform=platform,
                          max_checks=max_checks)
    status = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status)


@app.get("/api/verify")
def verify(data_hash: str):
    chain = get_chain()
    return chain.verify_record(data_hash)