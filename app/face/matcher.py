"""Face detection and matching module.

Detects faces in an input image, extracts face crops, and produces a compact
numeric fingerprint ("face encoding") used to match the same person across
different images.

Pipeline:
  * **Detection** - YuNet ONNX face detector (``cv2.FaceDetectorYN``), the modern
    OpenCV face detector. Falls back to the Haar cascade or "whole frame".
  * **Encoding**  - SFace (``cv2.FaceRecognizerSF``) produces a 128-dim face
    embedding - the same approach used by real face-recognition systems.
    If the SFace model is missing (offline/slim install), a perceptual
    gradient-hash + colour-histogram descriptor is used instead.
  * **Matching**  - cosine similarity between embeddings (SFace) or a weighted
    bit/colour score (fallback). Same person -> high similarity; different
    people -> low similarity.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_YUNET = os.getenv("FACE_DET_YUNET", os.path.join(_MODEL_DIR,
                                                  "face_detection_yunet_2023mar.onnx"))
_SFACE = os.getenv("FACE_REC_SFACE", os.path.join(_MODEL_DIR,
                                                  "face_recognition_sface_2021dec.onnx"))

_FR_COSINE_THRESHOLD = 0.363  # SFace recommended cosine threshold


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_yunet():
    try:
        if os.path.exists(_YUNET) and hasattr(cv2, "FaceDetectorYN"):
            return cv2.FaceDetectorYN.create(_YUNET, "", (320, 320))
    except Exception:
        pass
    return None


def _load_sface():
    try:
        if os.path.exists(_SFACE) and hasattr(cv2, "FaceRecognizerSF"):
            return cv2.FaceRecognizerSF.create(_SFACE, "")
    except Exception:
        pass
    return None


_YUNET_NET = _load_yunet()
_SFACE_NET = _load_sface()

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
try:
    _CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
except Exception:
    _CASCADE = None


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def decode_image(data: bytes):
    """Decode raw bytes -> (bgr ndarray, PIL Image, format string)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, img, (img.format or "jpeg").lower()


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _detect_yunet(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if _YUNET_NET is None:
        return []
    h, w = bgr.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []
    for scale in (1.0, 0.5, 0.25, 0.15):
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        small = cv2.resize(bgr, (sw, sh))
        _YUNET_NET.setInputSize((sw, sh))
        try:
            _, faces = _YUNET_NET.detect(small)
        except Exception:
            faces = None
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (int(v) for v in f[:4])
                if fw > 0 and fh > 0:
                    # map back to original coordinates
                    inv = 1.0 / scale
                    x0, y0 = int(x * inv), int(y * inv)
                    boxes.append((x0, y0, max(1, int(fw * inv)), max(1, int(fh * inv))))
        if boxes:
            break
    return boxes


def _detect_cascade(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if _CASCADE is None:
        return []
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = _CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
    except Exception:
        return []


def detect_faces(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes for every face in the image."""
    boxes = _detect_yunet(bgr)
    if not boxes:
        boxes = _detect_cascade(bgr)
    if not boxes:
        h, w = bgr.shape[:2]
        boxes = [(0, 0, w, h)]
    return boxes


# --------------------------------------------------------------------------- #
# Fallback descriptor (used when SFace model is absent)
# --------------------------------------------------------------------------- #
def _dhash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    resized = cv2.resize(gray, (size + 1, size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.float32)


def _color_hist(bgr: np.ndarray, bins: int = 24) -> np.ndarray:
    hists = []
    for i in range(3):
        hist = cv2.calcHist([bgr], [i], None, [bins], [0, 256])
        total = hist.sum() or 1.0
        hists.append((hist / total).flatten().astype(np.float32))
    return np.concatenate(hists)


def _fallback_feature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (96, 96))
    gvec = _dhash(gray, 16)
    cvec = _color_hist(crop, 24)
    return np.concatenate([gvec, cvec]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def _align_face(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Return the aligned face crop expected by SFace."""
    x, y, w, h = box
    crop = bgr[y:y + h, x:x + w]
    if crop.size == 0:
        crop = bgr
    crop = cv2.resize(crop, (112, 112))
    return crop


def encode_face(bgr: np.ndarray, box: Tuple[int, int, int, int]) -> dict:
    x, y, w, h = box
    align = _align_face(bgr, box)

    embedding = None
    sface_dim = 0
    if _SFACE_NET is not None:
        try:
            feat = _SFACE_NET.feature(align)
            embedding = feat.reshape(-1).astype(np.float32)
            sface_dim = int(embedding.shape[0])
        except Exception:
            embedding = None

    if embedding is None:
        feat = _fallback_feature(align)
        sface_dim = 0

    face_id = hashlib.sha256(feat.tobytes()).hexdigest()
    hx = face_id if sface_dim == 0 else \
        hashlib.sha256((b"sface:" + feat.tobytes())).hexdigest()

    return {
        "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "encoding": base64.b64encode(feat.tobytes()).decode(),
        "dim": int(feat.shape[0]),
        "model": "sface" if embedding is not None else "fallback",
        "face_id": hx,
    }


def faces_from_bytes(data: bytes) -> List[dict]:
    bgr, _, _ = decode_image(data)
    faces = []
    for box in detect_faces(bgr):
        try:
            face = encode_face(bgr, box)
            faces.append(face)
        except Exception:
            continue
    return faces


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _decode_feat(encoding: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(encoding), dtype=np.float32)


def _sface_cosine(e1: np.ndarray, e2: np.ndarray) -> float:
    denom = (np.linalg.norm(e1) * np.linalg.norm(e2)) or 1.0
    return float(np.dot(e1, e2) / denom)


def _fallback_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    g = 256
    ga, gb = e1[:g], e2[:g]
    ca, cb = e1[g:], e2[g:]
    bit_diff = int(np.count_nonzero(ga != gb)) / ga.shape[0]
    denom = (np.linalg.norm(ca) * np.linalg.norm(cb)) or 1.0
    cos = float(np.dot(ca, cb) / denom)
    bit_score = max(0.0, 1.0 - 1.6 * bit_diff)
    return max(0.0, cos), 0.55 * bit_score + 0.45 * max(0.0, cos)


def similarity(input_face: dict, other: dict) -> float:
    """0..1 similarity between two face encodings."""
    a = _decode_feat(input_face["encoding"])
    b = _decode_feat(other["encoding"])
    if a.shape != b.shape:
        return 0.0
    if input_face.get("model") == "sface" and other.get("model") == "sface":
        return round(_sface_cosine(a, b), 4)
    _, score = _fallback_similarity(a, b)
    return round(score, 4)


def is_match(input_face: dict, other: dict,
             threshold: float = _FR_COSINE_THRESHOLD) -> Tuple[bool, float]:
    """Decide if two encodings are the same person.

    Returns (bool, score). For SFace embeddings we use the model's documented
    cosine threshold; for the fallback descriptor a 0.72 score threshold.
    """
    score = similarity(input_face, other)
    if input_face.get("model") == "sface" and other.get("model") == "sface":
        return score >= threshold, score
    return score >= 0.72, score


def best_match(input_faces: List[dict], candidate: dict) -> dict:
    """Match a list of input faces against a single candidate face."""
    best = {"score": 0.0, "face_index": -1, "input_face_id": ""}
    for i, f in enumerate(input_faces):
        s = similarity(f, candidate)
        if s > best["score"]:
            best = {"score": s, "face_index": i, "input_face_id": f["face_id"]}
    return best