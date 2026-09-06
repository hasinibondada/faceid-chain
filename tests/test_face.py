"""Tests for the face detection / matching module."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.face.matcher import encode_face, detect_faces, faces_from_bytes
from app.face.matcher import decode_image, similarity
from scripts.make_sample import make_face
import numpy as np


def _face(box=(80, 80, 320, 320)):
    return encode_face(np.zeros((512, 512, 3), dtype=np.uint8), box)


def test_same_face_matches():
    f1 = _face()
    f2 = _face()
    assert f1["face_id"] == f2["face_id"]
    assert similarity(f1, f2) > 0.9


def test_blank_synthetic_face_detected():
    path = make_face()
    with open(path, "rb") as fh:
        data = fh.read()
    faces = faces_from_bytes(data)
    assert len(faces) >= 1


if __name__ == "__main__":
    test_same_face_matches()
    print("ok  test_same_face_matches")
    test_blank_synthetic_face_detected()
    print("ok  test_blank_synthetic_face_detected")
    print("all face tests passed")