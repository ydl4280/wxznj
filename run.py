import base64
import io
import os
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request
from paddleocr import PaddleOCR


app = Flask(__name__)

# Global OCR object for warm reuse (avoid reloading model every request).
OCR_ENGINE = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
    use_gpu=False,
)

ALNUM_PATTERN = re.compile(r"[A-Z0-9]+")
MARD_PATTERN = re.compile(r"^[A-Z]{1,2}[0-9]{1,3}$")


def make_error(debug_id: str, message: str, status_code: int = 400):
    return jsonify(
        {
            "success": False,
            "debugId": debug_id,
            "message": message,
        }
    ), status_code


def decode_base64_image(b64: str) -> np.ndarray:
    raw = b64.strip()
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[-1]
    image_bytes = base64.b64decode(raw)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image base64")
    return image


def download_image(url: str, timeout_sec: int = 20) -> np.ndarray:
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    image_bytes = resp.content
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image content from url")
    return image


def smooth_profile(profile: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1:
        return profile
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile, kernel, mode="same")


def collect_peaks(profile: np.ndarray, threshold: float) -> List[int]:
    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if profile[i] >= threshold and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            peaks.append(i)
    return peaks


def estimate_spacing_from_peaks(peaks: List[int], min_gap: int = 4, max_gap: int = 200) -> Optional[float]:
    if len(peaks) < 2:
        return None
    gaps = []
    for i in range(1, len(peaks)):
        gap = peaks[i] - peaks[i - 1]
        if min_gap <= gap <= max_gap:
            gaps.append(gap)
    if not gaps:
        return None
    return float(np.median(gaps))


def estimate_origin_from_peaks(peaks: List[int], spacing: float) -> float:
    if not peaks or spacing <= 0:
        return 0.0
    mods = [int(round(p % spacing)) for p in peaks]
    if not mods:
        return 0.0
    hist: Dict[int, int] = {}
    for m in mods:
        hist[m] = hist.get(m, 0) + 1
    origin_mod = max(hist.items(), key=lambda x: x[1])[0]
    return float(origin_mod)


def estimate_axis(
    axis_len: int,
    profile: np.ndarray,
    expected_count: Optional[int],
) -> Tuple[float, float, int]:
    if expected_count and expected_count > 0:
        spacing = axis_len / float(expected_count)
        return 0.0, spacing, int(expected_count)

    smoothed = smooth_profile(profile, 9)
    threshold = float(np.mean(smoothed) + np.std(smoothed) * 0.9)
    peaks = collect_peaks(smoothed, threshold)
    spacing = estimate_spacing_from_peaks(peaks)

    if not spacing or spacing <= 2:
        # Fallback: default around 52-grid if no reliable peaks.
        spacing = max(8.0, axis_len / 52.0)

    origin = estimate_origin_from_peaks(peaks, spacing)
    count = max(1, int(round((axis_len - origin) / spacing)))
    return origin, spacing, count


def estimate_grid(image: np.ndarray, rows: Optional[int], cols: Optional[int]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_x = np.abs(sobel_x)
    edge_y = np.abs(sobel_y)

    profile_x = np.sum(edge_x, axis=0)
    profile_y = np.sum(edge_y, axis=1)

    h, w = gray.shape[:2]
    origin_x, cell_w, final_cols = estimate_axis(w, profile_x, cols)
    origin_y, cell_h, final_rows = estimate_axis(h, profile_y, rows)

    return {
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "cell_w": float(cell_w),
        "cell_h": float(cell_h),
        "rows": int(final_rows),
        "cols": int(final_cols),
    }


def normalize_text(text: str) -> str:
    if not text:
        return ""
    up = text.upper()
    tokens = ALNUM_PATTERN.findall(up)
    if not tokens:
        return ""
    merged = "".join(tokens)
    return merged


def to_cell_index(center: float, origin: float, cell_size: float) -> int:
    if cell_size <= 0:
        return -1
    return int(np.floor((center - origin) / cell_size))


def run_full_image_ocr(image: np.ndarray):
    # PaddleOCR accepts numpy BGR image directly.
    result = OCR_ENGINE.ocr(image, cls=False)
    lines = result[0] if result and len(result) > 0 else []
    return lines


def map_ocr_to_grid(lines, grid_info):
    rows = grid_info["rows"]
    cols = grid_info["cols"]
    origin_x = grid_info["origin_x"]
    origin_y = grid_info["origin_y"]
    cell_w = grid_info["cell_w"]
    cell_h = grid_info["cell_h"]

    cells: Dict[Tuple[int, int], Dict] = {}

    for item in lines:
        if not item or len(item) < 2:
            continue
        box = item[0]
        text_info = item[1]
        if not text_info or len(text_info) < 2:
            continue

        text_raw = str(text_info[0] or "")
        conf = float(text_info[1] or 0.0)
        text = normalize_text(text_raw)
        if not text:
            continue
        if not MARD_PATTERN.match(text):
            # Keep only likely color-code style tokens.
            continue

        pts = np.array(box, dtype=np.float32)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        col = to_cell_index(cx, origin_x, cell_w)
        row = to_cell_index(cy, origin_y, cell_h)

        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue

        key = (row, col)
        prev = cells.get(key)
        if prev is None or conf > prev["conf"]:
            cells[key] = {
                "row": row + 1,
                "col": col + 1,
                "text": text,
                "conf": round(conf, 4),
            }

    mapped = list(cells.values())
    mapped.sort(key=lambda x: (x["row"], x["col"]))
    return mapped


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/count", methods=["POST"])
def api_count():
    started = time.time()
    debug_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    action = str(body.get("action") or "parse_grid")
    if action == "inc":
        # Keep compatibility with cloud console sample call.
        return jsonify(
            {
                "success": True,
                "action": "inc",
                "value": 1,
                "debugId": debug_id,
            }
        )

    rows = body.get("rows")
    cols = body.get("cols")
    rows = int(rows) if isinstance(rows, int) and rows > 0 else None
    cols = int(cols) if isinstance(cols, int) and cols > 0 else None

    image = None
    image_transport = str(body.get("imageTransport") or "")

    try:
        if body.get("image"):
            image = decode_base64_image(body.get("image"))
            image_transport = "base64"
        elif body.get("imageUrl"):
            image = download_image(body.get("imageUrl"))
            image_transport = image_transport or "cloudFileUrl"
        else:
            return make_error(debug_id, "missing image or imageUrl", 400)
    except Exception as exc:
        return make_error(debug_id, f"load image failed: {exc}", 400)

    try:
        h, w = image.shape[:2]
        grid_info = estimate_grid(image, rows, cols)
        ocr_lines = run_full_image_ocr(image)
        mapped_cells = map_ocr_to_grid(ocr_lines, grid_info)

        elapsed_ms = int((time.time() - started) * 1000)
        return jsonify(
            {
                "success": True,
                "debugId": debug_id,
                "action": action,
                "imageTransport": image_transport,
                "image": {
                    "width": int(w),
                    "height": int(h),
                },
                "grid": {
                    "rows": int(grid_info["rows"]),
                    "cols": int(grid_info["cols"]),
                    "origin": {
                        "x": round(grid_info["origin_x"], 2),
                        "y": round(grid_info["origin_y"], 2),
                    },
                    "cellSize": {
                        "x": round(grid_info["cell_w"], 4),
                        "y": round(grid_info["cell_h"], 4),
                    },
                },
                "rows": int(grid_info["rows"]),
                "cols": int(grid_info["cols"]),
                "cells": mapped_cells,
                "message": "ok",
                "elapsedMs": elapsed_ms,
            }
        )
    except Exception as exc:
        return make_error(debug_id, f"ocr failed: {exc}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
