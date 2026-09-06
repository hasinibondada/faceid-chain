"""Generate a synthetic test image containing a drawn "face".

Used for local testing of the pipeline's face detection + search + blockchain
flow without needing a real photo on disk.
"""
import io
import os
import sys

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_face.png")


def make_face(path: str = OUT, size: int = 512) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (size, size), "#cfe3f5")
    d = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r = size // 4
    # head
    d.ellipse([cx - r, cy - int(r * 1.25), cx + r, cy + int(r * 1.35)], fill="#e8b58a")
    # hair
    d.arc([cx - r - 4, cy - int(r * 1.35), cx + r + 4, cy + int(r * 0.15)],
          start=180, end=360, fill="#3d2b1f", width=size // 30)
    # eyes
    ey = cy + 2
    for ex in (cx - r // 3, cx + r // 3):
        d.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill="#ffffff")
        d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill="#2b2b2b")
    # nose
    d.line([cx, ey + 10, cx, ey + 34], fill="#d99e70", width=4)
    # mouth
    d.arc([cx - 34, ey + 30, cx + 34, ey + 70], start=20, end=160, fill="#b05b43", width=5)
    # ears
    d.ellipse([cx - r - 18, cy + 6, cx - r + 6, cy + 44], fill="#e8b58a")
    d.ellipse([cx + r - 6, cy + 6, cx + r + 18, cy + 44], fill="#e8b58a")

    img.save(path)
    return path


if __name__ == "__main__":
    print(make_face())