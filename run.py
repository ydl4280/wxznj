import base64
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

MAX_IMAGE_EDGE = 2200
VALID_CODE_RE = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")
ALNUM_RE = re.compile(r"[A-Z0-9]+")

OCR_ENGINE = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
    use_gpu=False,
)


def make_error(debug_id: str, message: str, status_code: int = 400):
    return jsonify(
        {
            "success": False,
            "debugId": debug_id,
            "message": message,
        }
    ), status_code


def parse_positive_int(value) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def resize_if_needed(image: np.ndarray, max_edge: int = MAX_IMAGE_EDGE) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image
    scale = float(max_edge) / float(max(h, w))
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def decode_base64_image(b64: str) -> np.ndarray:
    raw = str(b64 or "").strip()
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[-1]
    image_bytes = base64.b64decode(raw)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image base64")
    return resize_if_needed(image)


def download_image(url: str, timeout_sec: int = 20) -> np.ndarray:
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    image_bytes = resp.content
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid image content from url")
    return resize_if_needed(image)


def order_quad_points(quad: np.ndarray) -> np.ndarray:
    pts = quad.astype(np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_perspective_quad(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = float(image.shape[0] * image.shape[1])
    min_area = img_area * 0.20
    best = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best = approx
            best_area = area
    return best


def perspective_correct_if_needed(image: np.ndarray, enabled: bool) -> Tuple[np.ndarray, Dict]:
    if not enabled:
        return image, {"applied": False, "reason": "disabled"}

    quad = detect_perspective_quad(image)
    if quad is None:
        return image, {"applied": False, "reason": "no_quad"}

    ordered = order_quad_points(quad)
    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])
    target_w = int(round(max(width_a, width_b)))
    target_h = int(round(max(height_a, height_b)))
    if target_w < 50 or target_h < 50:
        return image, {"applied": False, "reason": "small_target"}

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (target_w, target_h))
    return warped, {"applied": True, "targetWidth": target_w, "targetHeight": target_h}


def build_grid_line_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    h, w = gray.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 70), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 70)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    grid = cv2.bitwise_or(horizontal, vertical)
    grid = cv2.dilate(grid, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return grid


def compute_grid_bbox(grid_mask: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = grid_mask.shape[:2]
    non_zero = cv2.findNonZero(grid_mask)
    if non_zero is None:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(non_zero)
    if bw * bh < 0.20 * w * h:
        return 0, 0, w, h
    pad = int(round(min(bw, bh) * 0.01))
    x = max(0, x - pad)
    y = max(0, y - pad)
    bw = min(w - x, bw + pad * 2)
    bh = min(h - y, bh + pad * 2)
    return x, y, bw, bh


def collect_line_centers(profile: np.ndarray, threshold: float) -> List[int]:
    active = np.where(profile >= threshold)[0]
    if active.size == 0:
        return []
    centers = []
    start = int(active[0])
    prev = int(active[0])
    for idx in active[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        centers.append(int(round((start + prev) / 2.0)))
        start = idx
        prev = idx
    centers.append(int(round((start + prev) / 2.0)))
    return centers


def robust_spacing(centers: List[int]) -> Optional[float]:
    if len(centers) < 2:
        return None
    diffs = [centers[i] - centers[i - 1] for i in range(1, len(centers))]
    diffs = [d for d in diffs if d > 2]
    if not diffs:
        return None
    return float(np.median(diffs))


def detect_axis_geometry(
    grid_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    axis: str,
    expected_count: Optional[int],
) -> Tuple[float, float, int]:
    x, y, w, h = bbox
    roi = grid_mask[y : y + h, x : x + w]
    if axis == "x":
        profile = np.sum((roi > 0).astype(np.float32), axis=0)
        axis_len = w
        axis_offset = x
    else:
        profile = np.sum((roi > 0).astype(np.float32), axis=1)
        axis_len = h
        axis_offset = y

    profile_max = float(np.max(profile)) if profile.size > 0 else 0.0
    centers_local = []
    if profile_max > 0:
        centers_local = collect_line_centers(profile, max(2.0, profile_max * 0.35))
        if len(centers_local) < 3:
            centers_local = collect_line_centers(profile, max(1.0, profile_max * 0.20))
    centers = [axis_offset + c for c in centers_local]

    if expected_count and expected_count > 0:
        if len(centers) >= 2:
            start = float(centers[0])
            end = float(centers[-1])
            if end <= start:
                start = float(axis_offset)
                end = float(axis_offset + axis_len)
        else:
            start = float(axis_offset)
            end = float(axis_offset + axis_len)
        spacing = max(1.0, (end - start) / float(expected_count))
        return start, spacing, int(expected_count)

    spacing = robust_spacing(centers)
    if spacing and spacing > 0 and len(centers) >= 2:
        start = float(centers[0])
        end = float(centers[-1])
        count = max(1, int(round((end - start) / spacing)))
        count = min(300, max(1, count))
        spacing = max(1.0, (end - start) / float(count))
        return start, spacing, count

    fallback_count = max(20, min(80, int(round(axis_len / 24.0))))
    fallback_spacing = max(1.0, axis_len / float(fallback_count))
    return float(axis_offset), float(fallback_spacing), int(fallback_count)


def detect_grid_geometry(
    image: np.ndarray, expected_rows: Optional[int], expected_cols: Optional[int]
) -> Dict:
    grid_mask = build_grid_line_mask(image)
    bbox = compute_grid_bbox(grid_mask)
    origin_x, cell_w, cols = detect_axis_geometry(grid_mask, bbox, "x", expected_cols)
    origin_y, cell_h, rows = detect_axis_geometry(grid_mask, bbox, "y", expected_rows)
    rows = min(300, max(1, rows))
    cols = min(300, max(1, cols))
    return {
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "cell_w": max(1.0, float(cell_w)),
        "cell_h": max(1.0, float(cell_h)),
        "rows": int(rows),
        "cols": int(cols),
        "bbox": {
            "x": int(bbox[0]),
            "y": int(bbox[1]),
            "width": int(bbox[2]),
            "height": int(bbox[3]),
        },
    }


def normalize_ocr_text(text: str) -> str:
    up = str(text or "").upper()
    tokens = ALNUM_RE.findall(up)
    if not tokens:
        return ""
    merged = "".join(tokens)
    if VALID_CODE_RE.match(merged):
        return merged
    if len(merged) >= 2 and any(ch.isalpha() for ch in merged) and any(ch.isdigit() for ch in merged):
        return merged
    return ""


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(blur, -1, sharpen_kernel)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def run_full_image_ocr(image: np.ndarray):
    processed = preprocess_for_ocr(image)
    try:
        result = OCR_ENGINE.ocr(processed, cls=False)
        return result[0] if result and len(result) > 0 and result[0] else []
    except Exception:
        return []


def to_cell_index(center: float, origin: float, cell_size: float) -> int:
    if cell_size <= 0:
        return -1
    return int(np.floor((center - origin) / cell_size))


def sample_cell_average_rgb(image: np.ndarray, row: int, col: int, grid: Dict) -> List[int]:
    h, w = image.shape[:2]
    origin_x = float(grid["origin_x"])
    origin_y = float(grid["origin_y"])
    cell_w = float(grid["cell_w"])
    cell_h = float(grid["cell_h"])
    x0 = int(round(origin_x + col * cell_w))
    y0 = int(round(origin_y + row * cell_h))
    x1 = int(round(origin_x + (col + 1) * cell_w))
    y1 = int(round(origin_y + (row + 1) * cell_h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return [255, 255, 255]
    avg_bgr = region.reshape(-1, 3).mean(axis=0)
    return [int(round(avg_bgr[2])), int(round(avg_bgr[1])), int(round(avg_bgr[0]))]


def map_ocr_lines_to_grid(lines, grid: Dict, image: np.ndarray) -> List[Dict]:
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    origin_x = float(grid["origin_x"])
    origin_y = float(grid["origin_y"])
    cell_w = float(grid["cell_w"])
    cell_h = float(grid["cell_h"])

    cells: Dict[Tuple[int, int], Dict] = {}
    for item in lines:
        if not item or len(item) < 2:
            continue
        box = item[0]
        text_info = item[1]
        if not text_info or len(text_info) < 2:
            continue
        raw_text = str(text_info[0] or "")
        conf = float(text_info[1] or 0.0)
        if conf < 0.20:
            continue
        code = normalize_ocr_text(raw_text)
        if not code:
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
            avg_color = sample_cell_average_rgb(image, row, col, grid)
            cells[key] = {
                "row": row + 1,
                "col": col + 1,
                "text": code,
                "code": code,
                "conf": round(conf, 4),
                "avgColor": avg_color,
            }

    mapped = list(cells.values())
    mapped.sort(key=lambda x: (x["row"], x["col"]))
    return mapped


def build_code_matrix(rows: int, cols: int, cells: List[Dict]) -> List[List[str]]:
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        row = int(cell["row"]) - 1
        col = int(cell["col"]) - 1
        if 0 <= row < rows and 0 <= col < cols:
            matrix[row][col] = str(cell.get("code") or "")
    return matrix


def build_material_stats(cells: List[Dict]) -> List[Dict]:
    counter: Dict[str, Dict] = {}
    for cell in cells:
        code = str(cell.get("code") or "").strip().upper()
        if not code:
            continue
        if code not in counter:
            counter[code] = {
                "code": code,
                "hex": "",
                "count": 0,
            }
        counter[code]["count"] += 1
    return sorted(counter.values(), key=lambda item: (-item["count"], item["code"]))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


def handle_count_request():
    started = time.time()
    debug_id = f"req_{uuid.uuid4().hex[:12]}"
    try:
        body = request.get_json(silent=True) or {}
    except Exception:
        body = {}

    action = str(body.get("action") or "parse_grid")
    if action == "inc":
        return jsonify(
            {
                "success": True,
                "action": "inc",
                "value": 1,
                "debugId": debug_id,
            }
        )

    rows = parse_positive_int(body.get("rows"))
    cols = parse_positive_int(body.get("cols"))
    enable_perspective = parse_bool(body.get("perspectiveCorrection"), True)
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
        source_h, source_w = image.shape[:2]
        corrected_image, perspective_meta = perspective_correct_if_needed(image, enable_perspective)
        parse_h, parse_w = corrected_image.shape[:2]
        grid = detect_grid_geometry(corrected_image, rows, cols)
        ocr_lines = run_full_image_ocr(corrected_image)
        cells = map_ocr_lines_to_grid(ocr_lines, grid, corrected_image)
        code_matrix = build_code_matrix(int(grid["rows"]), int(grid["cols"]), cells)
        material_stats = build_material_stats(cells)

        elapsed_ms = int((time.time() - started) * 1000)
        return jsonify(
            {
                "success": True,
                "debugId": debug_id,
                "action": action,
                "message": "ok",
                "elapsedMs": elapsed_ms,
                "imageTransport": image_transport,
                "image": {
                    "width": int(source_w),
                    "height": int(source_h),
                    "parsedWidth": int(parse_w),
                    "parsedHeight": int(parse_h),
                },
                "perspective": perspective_meta,
                "grid": {
                    "rows": int(grid["rows"]),
                    "cols": int(grid["cols"]),
                    "origin": {
                        "x": round(float(grid["origin_x"]), 2),
                        "y": round(float(grid["origin_y"]), 2),
                    },
                    "cellSize": {
                        "x": round(float(grid["cell_w"]), 4),
                        "y": round(float(grid["cell_h"]), 4),
                    },
                    "bbox": grid["bbox"],
                },
                "rows": int(grid["rows"]),
                "cols": int(grid["cols"]),
                "cells": cells,
                "codeMatrix": code_matrix,
                "materialStats": material_stats,
            }
        )
    except Exception as exc:
        import traceback

        print("[ERROR] parse pipeline failed")
        print(traceback.format_exc())
        return make_error(debug_id, f"parse failed: {str(exc)}", 500)


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "ok": True,
            "service": "cloud_ocr_service",
            "pipeline": [
                "read-image",
                "perspective-correction",
                "detect-grid",
                "ocr-grid-codes",
                "map-text-to-cells",
                "output-code-matrix-and-materials",
            ],
            "routes": [
                "/api/count",
                "/count",
                "/api/parse-grid",
                "/api/parse_grid",
                "/api/grid/parse",
                "/api/ocr/grid",
            ],
        }
    )


@app.route("/api/count", methods=["POST"])
@app.route("/api/count/", methods=["POST"])
@app.route("/count", methods=["POST"])
@app.route("/count/", methods=["POST"])
@app.route("/api/parse-grid", methods=["POST"])
@app.route("/api/parse_grid", methods=["POST"])
@app.route("/api/grid/parse", methods=["POST"])
@app.route("/api/ocr/grid", methods=["POST"])
def api_count():
    return handle_count_request()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
