"""Face detection and recognition using OpenCV HOG detector + pixel-based comparison."""

import cv2
import numpy as np
import os
from typing import Optional

import config

# Pre-built HOG face detector (built into OpenCV, no extra models needed)
_detector: Optional[cv2.CascadeClassifier] = None


def _get_detector() -> cv2.CascadeClassifier:
    global _detector
    if _detector is None:
        data_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
        cascade_paths = [
            os.path.join(data_dir, "haarcascade_frontalface_default.xml"),
            os.path.join(data_dir, "haarcascade_frontalface_alt2.xml"),
            os.path.join(data_dir, "haarcascade_frontalface_alt.xml"),
        ]
        for path in cascade_paths:
            try:
                det = cv2.CascadeClassifier(path)
                if not det.empty():
                    _detector = det
                    print(f"[face] Using Haar cascade: {os.path.basename(path)}")
                    return _detector
            except Exception:
                continue
        raise RuntimeError("No Haar cascade files found — run download_cascades.py")
    return _detector


_known_cache: Optional[tuple[list[int], list[str], list[np.ndarray]]] = None


def load_known_encodings(force_reload: bool = False) -> tuple[list[int], list[str], list[np.ndarray]]:
    """Load all known visitors and their face embeddings from the database or cache."""
    global _known_cache
    if _known_cache is None or force_reload:
        import database
        visitors = database.get_all_visitors()
        ids = [v["id"] for v in visitors]
        names = [v["name"] for v in visitors]
        embeddings = [v["embedding"] for v in visitors]
        _known_cache = (ids, names, embeddings)
    return _known_cache


def register_known_encoding(visitor_id: int, name: str, encoding: np.ndarray) -> None:
    """Add a newly registered visitor directly into the in-memory face cache."""
    global _known_cache
    if _known_cache is not None:
        ids, names, embeddings = _known_cache
        ids.append(visitor_id)
        names.append(name)
        embeddings.append(encoding)
    else:
        load_known_encodings(force_reload=True)


def capture_face_encoding(frame: np.ndarray) -> tuple[Optional[np.ndarray], Optional[tuple]]:
    """Detect face using Haar cascade, return (encoding, bbox) or (None, None)."""
    detector = _get_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None, None

    # Pick the largest face
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]

    # Crop face region with some padding
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face_roi = frame[y1:y2, x1:x2]
    face_roi = cv2.resize(face_roi, (80, 80))

    # Build encoding: normalized RGB pixels (80*80*3 = 19200 dims)
    encoding = face_roi.astype(np.float64) / 255.0
    encoding = encoding.flatten()
    # Normalise to zero-mean unit-variance
    encoding = (encoding - np.mean(encoding)) / (np.std(encoding) + 1e-8)

    bbox = (y1, x2, y2, x1)  # top, right, bottom, left
    return encoding, bbox


def compare_faces(new_encoding: np.ndarray, known_encodings: list[np.ndarray]) -> tuple:
    """Compare new encoding against known encodings. Returns (best_match_index, cosine_distance)."""
    if not known_encodings:
        return None, float("inf")

    new_norm = new_encoding / (np.linalg.norm(new_encoding) + 1e-8)
    distances = []
    for ek in known_encodings:
        ek_norm = ek / (np.linalg.norm(ek) + 1e-8)
        sim = float(np.dot(new_norm, ek_norm))
        distances.append(1.0 - sim)

    min_idx = int(np.argmin(distances))
    return min_idx, distances[min_idx]


def open_camera(camera_index: int | None = None) -> cv2.VideoCapture:
    """Open camera with retry and warm-up. Non-blocking with timeout."""
    import time

    idx = camera_index if camera_index is not None else config.FACE_CAMERA_INDEX
    widths = [640, 800, 320]
    heights = [480, 600, 240]

    cap = cv2.VideoCapture(idx)

    for attempt in range(3):
        if not cap.isOpened():
            print(f"[face] Camera not open (attempt {attempt+1}/15), retrying in 2s...")
            time.sleep(2)
            continue

        for _ in range(5):
            cap.read()

        for w, h in zip(widths, heights):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.3)
            ok, test_frame = cap.read()
            if ok and test_frame is not None and test_frame.size > 0:
                print(f"[face] Camera {idx} opened at {w}x{h}.")
                return cap

        ok, test_frame = cap.read()
        if ok and test_frame is not None and test_frame.size > 0:
            print(f"[face] Camera {idx} opened at default resolution.")
            return cap

        print(f"[face] Camera opened but no frames (attempt {attempt+1}/15), retrying...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(idx)

    raise RuntimeError(f"Could not get frames from camera {idx}")


def find_available_camera(max_index: int = 5) -> int | None:
    """Scan for available camera indices. Returns first working index or None."""
    import time
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            for _ in range(3):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None and frame.size > 0:
                print(f"[face] Found working camera at index {idx}")
                return idx
        time.sleep(0.5)
    return None
