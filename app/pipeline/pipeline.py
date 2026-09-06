"""End-to-end pipeline: face scan -> web/social search -> blockchain verify."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from app.face.matcher import faces_from_bytes
from app.search.webfinder import search_face
from app.blockchain.chain import get_chain


def _clean_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint(input_img_hash: str, post: Dict[str, Any]) -> Dict[str, Any]:
    """Build the post fingerprint that gets stored on-chain."""
    return {
        "type": "verified_post",
        "input_image_hash": input_img_hash,
        "post_url": post.get("page_url", ""),
        "post_image_url": post.get("image_url", ""),
        "post_title": post.get("title", ""),
        "post_snippet": post.get("snippet", ""),
        "post_source": post.get("source", ""),
        "match_score": post.get("match_score", 0.0),
        "match_route": post.get("match_route", ""),
        "matched_face_id": post.get("matched_face_id", ""),
        "candidate_image_sha256": post.get("candidate_image_sha256", ""),
        "url_hash": post.get("url_hash", ""),
        "discovered_at": time.time(),
    }


def run_pipeline(image_bytes: bytes,
                 query: Optional[str] = None,
                 platform: Optional[str] = None,
                 max_checks: int = 4) -> Dict[str, Any]:
    """Run the full face->search->blockchain pipeline.

    Returns a structured result containing the face encodings, the search
    matches, and the on-chain verification of the top match.
    """
    started = time.time()

    # 1. Face identification
    faces = faces_from_bytes(image_bytes)
    input_img_hash = _clean_bytes(image_bytes)
    if not faces:
        return {"success": False, "error": "No face detected in the input image."}

    # 2. Web / social media search (genuine reverse-image + face-verified)
    search = search_face(faces, image_bytes=image_bytes, query=query,
                         platform=platform, max_results=14,
                         max_face_checks=max_checks)
    matches = search.get("matched_posts", [])

    # 3. Blockchain upload + verification
    chain = get_chain()
    records = []
    verified = None
    if matches:
        top = matches[0]
        fingerprint = _fingerprint(input_img_hash, top)
        tx = chain.add_record(fingerprint)
        verified = chain.verify_record(tx["data_hash"])
        records.append({"tx_id": tx["tx_id"],
                        "data_hash": tx["data_hash"],
                        "block_index": tx.get("block_index"),
                        "block_hash": tx.get("block_hash"),
                        "post": top})
    chain_status = chain.status()
    tamper = chain.verify_tamper()

    return {
        "success": True,
        "elapsed_seconds": round(time.time() - started, 2),
        "input_image_hash": input_img_hash,
        "faces_detected": len(faces),
        "face_ids": [f["face_id"] for f in faces],
        "search": {
            "backend": search.get("backend"),
            "query": search.get("query"),
            "total_hits": search.get("total_images", 0),
            "face_checked": search.get("face_checked", 0),
            "matched_posts": matches,
        },
        "blockchain": {
            "chain_name": chain_status.get("name"),
            "height": chain_status.get("height"),
            "ledger_path": chain_status.get("ledger_path"),
            "tamper_check": tamper,
            "records": records,
            "verification": verified,
        },
    }