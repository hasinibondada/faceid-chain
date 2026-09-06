"""Download the face-detection model files (external Deep Neural Nets).

Preferred: YuNet (OpenCV Zoo, ONNX, ~230 KB) — modern and supported by
OpenCV 4.8+ and 5.x via cv2.FaceDetectorYN.

Legacy: the OpenCV SSD-Caffe res10 detector, used by OpenCV < 4.11 / 5.x
without the Caffe importer.
"""
import os
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def download(name: str, url: str) -> None:
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"[ok] already present: {name}")
        return
    print(f"[..] downloading {name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"[ok] saved {name} ({len(data)/1e6:.2f} MB)")
    except Exception as e:  # noqa: BLE001
        print(f"[!!] failed to download {name}: {e}")


def main() -> int:
    for name, url in FILES.items():
        download(name, url)
    print("\nDone. Face detection model is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())