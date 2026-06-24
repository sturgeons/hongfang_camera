import time
import json
import os
import shutil
import cv2
import numpy as np
import subprocess
import requests
from requests.auth import HTTPDigestAuth
from socket import *
from flask import Flask, Response, render_template, jsonify
import threading
from collections import deque

from tag_tracker import TagTracker

app = Flask(__name__, template_folder='templates')

udpSocket = socket(AF_INET, SOCK_DGRAM)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


UDP_HOST = os.getenv("UDP_HOST", "192.168.81.20")
UDP_PORT = _env_int("UDP_PORT", 6789)
addr = (UDP_HOST, UDP_PORT)

# 摄像头配置
CAMERA_IP = os.getenv("CAMERA_IP", "192.168.81.82")
CAMERA_USER = os.getenv("CAMERA_USER", "admin")
CAMERA_PASS = os.getenv("CAMERA_PASS", "Zf123456789!")

# 101=主码流  102=子码流(720p, 采集解码更快)
SNAPSHOT_CHANNEL = _env_int("SNAPSHOT_CHANNEL", 102)
snapshot_url = f"http://{CAMERA_IP}/ISAPI/Streaming/channels/{SNAPSHOT_CHANNEL}/picture"
RTSP_STREAM = os.getenv("RTSP_STREAM", "sub")
RTSP_CODEC = os.getenv("RTSP_CODEC", "h264")
camera_path = os.getenv(
    "CAMERA_RTSP_URL",
    f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}/{RTSP_CODEC}/ch1/{RTSP_STREAM}/av_stream",
)

CAPTURE_BACKEND = os.getenv("CAPTURE_BACKEND", "ffmpeg").lower()
ARUCO_DICTIONARY = os.getenv("ARUCO_DICTIONARY", "DICT_6X6_1000")
USE_SNAPSHOT_MODE = _env_bool("USE_SNAPSHOT_MODE", False)
RTSP_FLUSH_FRAMES = _env_int("RTSP_FLUSH_FRAMES", 1)
USE_GPU_DECODE = _env_bool("USE_GPU_DECODE", False)

# 核心性能策略：采集端直接输出算法需要的低分辨率灰度帧。
# 车辆进出只依赖纵向轨迹，不需要对 720p/1080p 整帧做 ArUco 检测。
PROCESS_WIDTH = _env_int("PROCESS_WIDTH", 640)
PROCESS_HEIGHT = _env_int("PROCESS_HEIGHT", 360)
DETECT_WIDTH = _env_int("DETECT_WIDTH", 480)
DETECT_PADDING = _env_int("DETECT_PADDING", 24)
ROI_X_MIN = _env_float("ROI_X_MIN", 0.0)
ROI_X_MAX = _env_float("ROI_X_MAX", 1.0)
ROI_Y_MIN = _env_float("ROI_Y_MIN", 0.0)
ROI_Y_MAX = _env_float("ROI_Y_MAX", 1.0)
MOTION_GATE = _env_bool("MOTION_GATE", True)
MOTION_MIN_AREA = _env_int("MOTION_MIN_AREA", 80)
MOTION_THRESHOLD = _env_int("MOTION_THRESHOLD", 18)
MOTION_EXPAND = _env_float("MOTION_EXPAND", 0.18)
MOTION_FULL_SCAN_INTERVAL = _env_int("MOTION_FULL_SCAN_INTERVAL", 0)
MOTION_MAX_AREA_RATIO = _env_float("MOTION_MAX_AREA_RATIO", 0.45)
MOTION_WIDTH = _env_int("MOTION_WIDTH", 160)
MOTION_BG_ALPHA = _env_float("MOTION_BG_ALPHA", 0.02)
IDLE_SCAN_INTERVAL = _env_int("IDLE_SCAN_INTERVAL", 5)
STARTUP_SCAN_FRAMES = _env_int("STARTUP_SCAN_FRAMES", 12)
SCAN_HOLD_SECONDS = _env_float("SCAN_HOLD_SECONDS", 1.2)
SCAN_ROI_MIN_WIDTH = _env_int("SCAN_ROI_MIN_WIDTH", 260)
SCAN_ROI_MIN_HEIGHT = _env_int("SCAN_ROI_MIN_HEIGHT", 180)
SCAN_FULL_EVERY = _env_int("SCAN_FULL_EVERY", 8)
EDGE_SCAN_INTERVAL = _env_int("EDGE_SCAN_INTERVAL", 2)
EDGE_SCAN_HEIGHT_RATIO = _env_float("EDGE_SCAN_HEIGHT_RATIO", 0.38)
CAPTURE_FPS = _env_int("CAPTURE_FPS", 20)
DISPLAY_MAX_WIDTH = _env_int("DISPLAY_MAX_WIDTH", 960)
RENDER_FPS = _env_int("RENDER_FPS", CAPTURE_FPS)
RENDER_INTERVAL = max(0.02, 1.0 / max(1, RENDER_FPS))
JPEG_QUALITY = _env_int("JPEG_QUALITY", 65)
OPENCV_MIN_PERIMETER = _env_float("OPENCV_MIN_PERIMETER", 0.03)
OPENCV_ADAPTIVE_MAX = _env_int("OPENCV_ADAPTIVE_MAX", 15)

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

latest_gray: np.ndarray | None = None
latest_gray_id = 0
gray_lock = threading.Lock()
gray_ready = threading.Event()

latest_jpeg: bytes | None = None
latest_jpeg_id = 0
display_lock = threading.Lock()

capture_running = False
detection_running = False
render_running = False
worker_lock = threading.Lock()
capture_thread: threading.Thread | None = None
detection_thread: threading.Thread | None = None
tag_worker_thread: threading.Thread | None = None
render_thread: threading.Thread | None = None
stats_lock = threading.Lock()
stats = {
    "active_count": 0,
    "total_in": 0,
    "total_out": 0,
    "fps_capture": 0.0,
    "fps_detect": 0.0,
    "fps_tag": 0.0,
    "fps_render": 0.0,
    "capture_interval_ms": 0.0,
    "detect_ms": 0.0,
    "motion_ms": 0.0,
    "tag_ms": 0.0,
    "render_ms": 0.0,
    "frame_age_ms": 0.0,
    "last_detections": 0,
    "queue_depth": 0,
    "tag_queue_depth": 0,
    "dropped_frames": 0,
    "decode_backend": "unknown",
    "detect_backend": "opencv",
    "aruco_dictionary": ARUCO_DICTIONARY,
    "frame_size": f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}",
    "detect_size": "",
    "detect_roi": "",
    "motion_gate": MOTION_GATE,
    "motion_area": 0,
    "motion_ratio": 0.0,
    "scan_hold": SCAN_HOLD_SECONDS,
    "scan_roi_source": "",
    "startup_scan_frames": STARTUP_SCAN_FRAMES,
    "edge_scan_interval": EDGE_SCAN_INTERVAL,
}


class ArucoDetector:
    def __init__(self, dictionary_name: str = ARUCO_DICTIONARY):
        dict_attr = getattr(cv2.aruco, dictionary_name, None)
        if dict_attr is None:
            raise ValueError(f"未知 ArUco 字典: {dictionary_name}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        params.minMarkerPerimeterRate = OPENCV_MIN_PERIMETER
        params.maxMarkerPerimeterRate = 4.0
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = OPENCV_ADAPTIVE_MAX
        params.adaptiveThreshWinSizeStep = 10
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.backend = "opencv"
        self.dictionary = dictionary_name

    def detect(self, gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        gray = np.ascontiguousarray(gray)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return [], None
        return corners, ids


aruco_detector = ArucoDetector()

tag_tracker = TagTracker()

# 检测与渲染分离：检测全速，画面跟随采集帧率刷新
_overlay_lock = threading.Lock()
_overlay_ready = threading.Event()
_overlay = {
    "gray": None,
    "markers": [],
    "tracks": [],
}

_motion_lock = threading.Lock()
_motion_bg: np.ndarray | None = None
_detect_iteration = 0
_active_scan_roi: tuple[int, int, int, int] | None = None
_active_scan_until = 0.0
_scan_submit_count = 0

_tag_task_lock = threading.Lock()
_tag_task_event = threading.Event()
_tag_tasks: deque[dict] = deque(maxlen=8)
_tag_worker_running = False
_tag_count = 0
_tag_t0 = time.time()
_tag_latest_det_count = 0


def post_event(tag_id: int, op: str) -> None:
    payload = json.dumps({"op": op, "code": str(tag_id)})
    udpSocket.sendto(payload.encode('utf-8'), addr)
    print(f"[{op.upper()}] 标签 ID={tag_id}")


def marker_center(corners: np.ndarray) -> tuple[float, float]:
    points = corners.reshape((4, 2))
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    return cx, cy


def draw_zones(display_img: np.ndarray) -> None:
    """绘制进出方向参考线"""
    h = display_img.shape[0]
    top_line = int(h * 0.15)
    bottom_line = int(h * 0.85)

    cv2.line(display_img, (0, top_line), (display_img.shape[1], top_line), (255, 200, 0), 1)
    cv2.line(display_img, (0, bottom_line), (display_img.shape[1], bottom_line), (255, 200, 0), 1)
    cv2.putText(display_img, "IN (top)", (10, top_line - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(display_img, "OUT (bottom)", (10, bottom_line + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _ffmpeg_supports_cuda() -> bool:
    ff = _ffmpeg_bin()
    if not ff:
        return False
    try:
        result = subprocess.run(
            [ff, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "cuda" in (result.stdout + result.stderr).lower()
    except Exception:
        return False


def _clip_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def base_roi(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape[:2]
    x1 = int(w * _clip_ratio(ROI_X_MIN))
    x2 = int(w * _clip_ratio(ROI_X_MAX))
    y1 = int(h * _clip_ratio(ROI_Y_MIN))
    y2 = int(h * _clip_ratio(ROI_Y_MAX))

    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h
    return x1, y1, x2, y2


def clamp_roi(
    roi: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    bx1, by1, bx2, by2 = bounds
    return max(bx1, x1), max(by1, y1), min(bx2, x2), min(by2, y2)


def expand_to_min_size(
    roi: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    width = max(x2 - x1, SCAN_ROI_MIN_WIDTH)
    height = max(y2 - y1, SCAN_ROI_MIN_HEIGHT)
    expanded = (
        cx - width // 2,
        cy - height // 2,
        cx + (width + 1) // 2,
        cy + (height + 1) // 2,
    )
    return clamp_roi(expanded, bounds)


def union_roi(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return clamp_roi(
        (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
        bounds,
    )


def motion_roi(gray: np.ndarray, base: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Find moving region inside configured ROI. Returns None when no motion."""
    global _motion_bg

    motion_t0 = time.perf_counter()

    if not MOTION_GATE:
        return base

    x1, y1, x2, y2 = base
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return base

    motion_scale = min(1.0, MOTION_WIDTH / roi.shape[1]) if roi.shape[1] else 1.0
    if motion_scale < 1.0:
        motion_img = cv2.resize(
            roi,
            (max(1, int(roi.shape[1] * motion_scale)), max(1, int(roi.shape[0] * motion_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        motion_img = roi

    small = cv2.GaussianBlur(motion_img, (5, 5), 0)
    with _motion_lock:
        if _motion_bg is None or _motion_bg.shape != small.shape:
            _motion_bg = small.astype(np.float32)
            with stats_lock:
                stats["motion_area"] = 0
                stats["motion_ratio"] = 0.0
                stats["motion_ms"] = round((time.perf_counter() - motion_t0) * 1000, 2)
            return None
        bg_uint8 = cv2.convertScaleAbs(_motion_bg)
        diff = cv2.absdiff(small, bg_uint8)
        cv2.accumulateWeighted(small, _motion_bg, MOTION_BG_ALPHA)

    _, mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    total_area = 0
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < MOTION_MIN_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        boxes.append((bx, by, bx + bw, by + bh))
        total_area += area

    with stats_lock:
        stats["motion_area"] = total_area
        stats["motion_ms"] = round((time.perf_counter() - motion_t0) * 1000, 2)

    if not boxes:
        with stats_lock:
            stats["motion_ratio"] = 0.0
        return None

    inv_motion_scale = 1.0 / motion_scale if motion_scale else 1.0
    mx1 = int(min(b[0] for b in boxes) * inv_motion_scale) + x1
    my1 = int(min(b[1] for b in boxes) * inv_motion_scale) + y1
    mx2 = int(max(b[2] for b in boxes) * inv_motion_scale) + x1
    my2 = int(max(b[3] for b in boxes) * inv_motion_scale) + y1

    base_area = max(1, (x2 - x1) * (y2 - y1))
    box_ratio = ((mx2 - mx1) * (my2 - my1)) / base_area
    with stats_lock:
        stats["motion_ratio"] = round(box_ratio, 3)
    if box_ratio > MOTION_MAX_AREA_RATIO:
        return None

    expand_x = int((mx2 - mx1) * MOTION_EXPAND)
    expand_y = int((my2 - my1) * MOTION_EXPAND)
    mx1 = max(x1, mx1 - expand_x)
    my1 = max(y1, my1 - expand_y)
    mx2 = min(x2, mx2 + expand_x)
    my2 = min(y2, my2 + expand_y)

    if mx2 <= mx1 or my2 <= my1:
        return None
    return mx1, my1, mx2, my2


def detection_window(gray: np.ndarray) -> tuple[np.ndarray | None, int, int, float]:
    """Return cropped/scaled image plus offset and scale back to full frame."""
    global _detect_iteration, _active_scan_roi, _active_scan_until, _scan_submit_count

    _detect_iteration += 1
    base = base_roi(gray)
    now = time.time()

    if _detect_iteration <= STARTUP_SCAN_FRAMES:
        roi = base
        roi_source = "startup-full"
        with stats_lock:
            stats["motion_area"] = 0
            stats["motion_ratio"] = 0.0
    else:
        roi = None
        roi_source = "motion"

    if roi is None:
        motion = motion_roi(gray, base)

        if motion is not None:
            motion = expand_to_min_size(motion, base)
            if _active_scan_roi is not None and now < _active_scan_until:
                motion = union_roi(_active_scan_roi, motion, base)
            _active_scan_roi = motion
            _active_scan_until = now + SCAN_HOLD_SECONDS
            roi = motion
        elif _active_scan_roi is not None and now < _active_scan_until:
            roi = _active_scan_roi
            roi_source = "hold"
        else:
            _active_scan_roi = None

    if roi is None:
        if IDLE_SCAN_INTERVAL > 0 and _detect_iteration % IDLE_SCAN_INTERVAL == 0:
            roi = base
            roi_source = "idle-full"
        elif MOTION_FULL_SCAN_INTERVAL > 0 and _detect_iteration % MOTION_FULL_SCAN_INTERVAL == 0:
            roi = base
            roi_source = "interval-full"
        else:
            with stats_lock:
                stats["detect_size"] = "skipped"
                stats["detect_roi"] = "motion:none"
                stats["scan_roi_source"] = "skipped"
            return None, 0, 0, 1.0

    if SCAN_FULL_EVERY > 0 and _active_scan_roi is not None:
        _scan_submit_count += 1
        if _scan_submit_count % SCAN_FULL_EVERY == 0:
            roi = base
            roi_source = "fallback-full"

    detect_img, offset_x, offset_y, scale = build_detection_crop(gray, roi)
    if detect_img is None:
        return None, 0, 0, 1.0

    with stats_lock:
        stats["detect_size"] = f"{detect_img.shape[1]}x{detect_img.shape[0]}"
        stats["detect_roi"] = f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}"
        stats["scan_roi_source"] = roi_source

    return detect_img, offset_x, offset_y, scale


def build_detection_crop(
    gray: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, float, float, float]:
    x1, y1, x2, y2 = clamp_roi(roi, base_roi(gray))
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0, 1.0

    crop = gray[y1:y2, x1:x2]
    scale = min(1.0, DETECT_WIDTH / crop.shape[1]) if crop.shape[1] else 1.0
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    if DETECT_PADDING > 0:
        crop = cv2.copyMakeBorder(
            crop,
            DETECT_PADDING,
            DETECT_PADDING,
            DETECT_PADDING,
            DETECT_PADDING,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        inv_scale = 1.0 / scale if scale else 1.0
        return crop, x1 - DETECT_PADDING * inv_scale, y1 - DETECT_PADDING * inv_scale, scale

    return crop, x1, y1, scale


def edge_scan_rois(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    if EDGE_SCAN_INTERVAL <= 0 or _detect_iteration % EDGE_SCAN_INTERVAL != 0:
        return []

    x1, y1, x2, y2 = base_roi(gray)
    height = max(1, int((y2 - y1) * _clip_ratio(EDGE_SCAN_HEIGHT_RATIO)))
    top = (x1, y1, x2, min(y2, y1 + height))
    bottom = (x1, max(y1, y2 - height), x2, y2)
    return [top, bottom]


def map_corners_to_frame(
    corners: list[np.ndarray],
    offset_x: int,
    offset_y: int,
    scale: float,
) -> list[np.ndarray]:
    if not corners:
        return []

    mapped: list[np.ndarray] = []
    inv_scale = 1.0 / scale if scale else 1.0
    for marker_corners in corners:
        c = marker_corners.astype(np.float32).copy()
        if scale != 1.0:
            c *= inv_scale
        c[:, :, 0] += offset_x
        c[:, :, 1] += offset_y
        mapped.append(c)
    return mapped


def publish_overlay(gray: np.ndarray, markers: list | None = None) -> None:
    with _overlay_lock:
        _overlay["gray"] = gray
        if markers is not None:
            _overlay["markers"] = markers
        _overlay["tracks"] = list(tag_tracker.active_tracks().values())
    _overlay_ready.set()


def submit_tag_task(gray: np.ndarray, detect_img: np.ndarray, offset_x: int, offset_y: int, scale: float) -> bool:
    with _tag_task_lock:
        _tag_tasks.append({
            "gray": gray,
            "detect_img": detect_img,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "scale": scale,
            "submitted_at": time.time(),
        })
        with stats_lock:
            stats["tag_queue_depth"] = len(_tag_tasks)
        _tag_task_event.set()
    return True


def consume_tracker_events(events) -> None:
    for event in events:
        post_event(event.tag_id, event.op)
        with stats_lock:
            if event.op == "in":
                stats["total_in"] += 1
            else:
                stats["total_out"] += 1

    with stats_lock:
        stats["active_count"] = len(tag_tracker.active_tracks())


def process_tag_task(task: dict) -> int:
    global _tag_count, _tag_t0, _tag_latest_det_count

    detect_t0 = time.perf_counter()
    corners, ids = aruco_detector.detect(task["detect_img"])
    corners = map_corners_to_frame(corners, task["offset_x"], task["offset_y"], task["scale"])
    tag_ms = (time.perf_counter() - detect_t0) * 1000

    detections: list[tuple[int, float, float]] = []
    markers: list[tuple[int, float, float, np.ndarray]] = []
    if corners and ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            cx, cy = marker_center(marker_corners)
            detections.append((int(marker_id), cx, cy))
            markers.append((int(marker_id), cx, cy, marker_corners))

    events = tag_tracker.update(detections)
    consume_tracker_events(events)
    publish_overlay(task["gray"], markers)

    _tag_latest_det_count = len(detections)
    _tag_count += 1
    elapsed = time.time() - _tag_t0
    if elapsed >= 2.0:
        with stats_lock:
            stats["fps_tag"] = round(_tag_count / elapsed, 1)
        _tag_count = 0
        _tag_t0 = time.time()

    with stats_lock:
        stats["tag_ms"] = round(tag_ms, 2)
        stats["last_detections"] = len(detections)

    return len(detections)


def tag_worker_func():
    global _tag_worker_running

    while _tag_worker_running:
        _tag_task_event.wait(0.1)
        if not _tag_worker_running:
            break

        with _tag_task_lock:
            task = _tag_tasks.popleft() if _tag_tasks else None
            if not _tag_tasks:
                _tag_task_event.clear()
            with stats_lock:
                stats["tag_queue_depth"] = len(_tag_tasks)

        if task is None:
            continue

        try:
            process_tag_task(task)
        except Exception as e:
            print(f"标签检测错误: {e}")


def detect_and_track(gray: np.ndarray) -> int:
    """轻量帧处理：运动门控 + 异步提交 ArUco ROI。"""
    h, w = gray.shape[:2]
    tag_tracker.set_frame_height(h)
    submitted = False

    detect_img, offset_x, offset_y, scale = detection_window(gray)
    if detect_img is not None:
        submit_tag_task(gray, detect_img, offset_x, offset_y, scale)
        submitted = True

    for roi in edge_scan_rois(gray):
        edge_img, edge_x, edge_y, edge_scale = build_detection_crop(gray, roi)
        if edge_img is not None:
            submit_tag_task(gray, edge_img, edge_x, edge_y, edge_scale)
            submitted = True

    publish_overlay(gray)

    return _tag_latest_det_count


def render_display(det_count: int) -> np.ndarray | None:
    """从缓存状态渲染画面（冷路径，低频调用）。"""
    with _overlay_lock:
        gray = _overlay["gray"]
        markers = list(_overlay["markers"])
        tracks = list(_overlay["tracks"])

    if gray is None:
        return None

    h, w = gray.shape[:2]
    disp_scale = min(1.0, DISPLAY_MAX_WIDTH / w)
    if disp_scale < 1.0:
        dw, dh = int(w * disp_scale), int(h * disp_scale)
        disp = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_LINEAR)
    else:
        disp_scale = 1.0
        disp = gray

    display_img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    draw_zones(display_img)

    with stats_lock:
        roi_text = stats.get("detect_roi", "")

    if roi_text and "," in roi_text:
        try:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi_text.split(",")]
            p1 = (int(rx1 * disp_scale), int(ry1 * disp_scale))
            p2 = (int(rx2 * disp_scale), int(ry2 * disp_scale))
            cv2.rectangle(display_img, p1, p2, (180, 120, 255), 1)
        except Exception:
            pass

    for marker_id, cx, cy, marker_corners in markers:
        sx, sy = int(cx * disp_scale), int(cy * disp_scale)
        pts = (marker_corners.reshape(4, 2) * disp_scale).astype(np.int32)
        cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
        cv2.circle(display_img, (sx, sy), 5, (0, 255, 255), -1)
        cv2.putText(display_img, f"#{marker_id}", (sx - 20, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for track in tracks:
        if len(track.y_samples) >= 2:
            tail = min(6, len(track.y_samples))
            pts = np.array(
                [[int(track.x_samples[i] * disp_scale), int(track.y_samples[i] * disp_scale)]
                 for i in range(-tail, 0)],
                dtype=np.int32,
            )
            cv2.polylines(display_img, [pts], False, (0, 180, 255), 1)

    with stats_lock:
        det_fps = stats["fps_detect"]
        tag_fps = stats["fps_tag"]
        cap_fps = stats["fps_capture"]
        dsp_fps = stats.get("fps_render", 0.0)
        backend = stats.get("decode_backend", "?")
        active = stats["active_count"]

    status = (
        f"Tag:{det_count} Trk:{active} | "
        f"Cap{cap_fps} Loop{det_fps} Tag{tag_fps} Dsp{dsp_fps} [{backend}]"
    )
    cv2.putText(display_img, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)
    return display_img


_capture_count = 0
_capture_t0 = time.time()
_last_capture_ts = 0.0
_detect_count = 0
_detect_t0 = time.time()


_dropped_frames = 0


def _publish_gray(gray: np.ndarray) -> None:
    """采集线程只发布最新帧，旧帧直接覆盖以保证低延迟。"""
    global latest_gray, latest_gray_id, _capture_count, _capture_t0
    global _dropped_frames, _last_capture_ts

    now = time.time()
    with gray_lock:
        if latest_gray is not None and gray_ready.is_set():
            _dropped_frames += 1
        latest_gray = gray
        latest_gray_id += 1
        _capture_count += 1
        gray_ready.set()

        if _last_capture_ts:
            with stats_lock:
                stats["capture_interval_ms"] = round((now - _last_capture_ts) * 1000, 1)
        _last_capture_ts = now

        elapsed = now - _capture_t0
        if elapsed >= 2.0:
            with stats_lock:
                stats["fps_capture"] = round(_capture_count / elapsed, 1)
                stats["queue_depth"] = 1 if gray_ready.is_set() else 0
                stats["dropped_frames"] = _dropped_frames
            _capture_count = 0
            _capture_t0 = time.time()


def _take_latest_gray(timeout: float = 0.05) -> np.ndarray | None:
    """检测线程消费最新帧，没有新帧时短暂等待。"""
    if not gray_ready.wait(timeout):
        return None

    with gray_lock:
        if latest_gray is None:
            gray_ready.clear()
            return None
        gray = latest_gray
        gray_ready.clear()
        with stats_lock:
            stats["queue_depth"] = 0
            if _last_capture_ts:
                stats["frame_age_ms"] = round((time.time() - _last_capture_ts) * 1000, 1)
        return gray


def _tick_detect_fps(det_count: int) -> None:
    global _detect_count, _detect_t0

    _detect_count += 1
    elapsed = time.time() - _detect_t0
    if elapsed >= 2.0:
        with stats_lock:
            stats["fps_detect"] = round(_detect_count / elapsed, 1)
            stats["last_detections"] = det_count
        _detect_count = 0
        _detect_t0 = time.time()


def _publish_jpeg(jpeg_bytes: bytes) -> None:
    global latest_jpeg, latest_jpeg_id

    with display_lock:
        latest_jpeg = jpeg_bytes
        latest_jpeg_id += 1


_render_count = 0
_render_t0 = time.time()


def render_thread_func():
    global render_running, _render_count, _render_t0

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    print(f"渲染线程: 目标 {RENDER_FPS}fps (间隔 {RENDER_INTERVAL*1000:.0f}ms)")

    while render_running:
        _overlay_ready.wait(timeout=RENDER_INTERVAL)
        _overlay_ready.clear()

        loop_start = time.perf_counter()
        try:
            frame = render_display(_tag_latest_det_count)
            if frame is None:
                continue

            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue

            _publish_jpeg(buffer.tobytes())

            _render_count += 1
            elapsed = time.time() - _render_t0
            if elapsed >= 2.0:
                with stats_lock:
                    stats["fps_render"] = round(_render_count / elapsed, 1)
                    stats["render_ms"] = round((time.perf_counter() - loop_start) * 1000, 2)
                _render_count = 0
                _render_t0 = time.time()
        except Exception as e:
            print(f"渲染错误: {e}")

        spent = time.perf_counter() - loop_start
        if spent < RENDER_INTERVAL:
            time.sleep(RENDER_INTERVAL - spent)


def detection_thread_func():
    global detection_running

    print(
        f"检测线程: ArUco {aruco_detector.dictionary} "
        f"frame={PROCESS_WIDTH}x{PROCESS_HEIGHT} display<={DISPLAY_MAX_WIDTH}px"
    )

    while detection_running:
        gray = _take_latest_gray()
        if gray is None:
            continue

        try:
            detect_t0 = time.perf_counter()
            det_count = detect_and_track(gray)
            detect_ms = (time.perf_counter() - detect_t0) * 1000
            with stats_lock:
                stats["detect_ms"] = round(detect_ms, 2)
        except Exception as e:
            print(f"检测错误: {e}")
            time.sleep(0.01)
            continue

        _tick_detect_fps(det_count)


def capture_thread_func():
    if USE_SNAPSHOT_MODE:
        capture_thread_snapshot()
        return

    if CAPTURE_BACKEND == "ffmpeg":
        if USE_GPU_DECODE and _ffmpeg_supports_cuda():
            with stats_lock:
                stats["decode_backend"] = "GPU"
            capture_thread_rtsp_ffmpeg(gpu=True)
            return
        if _ffmpeg_bin():
            with stats_lock:
                stats["decode_backend"] = "FFmpeg"
            capture_thread_rtsp_ffmpeg(gpu=False)
            return

    with stats_lock:
        stats["decode_backend"] = "OpenCV"
    capture_thread_opencv()


def capture_thread_snapshot():
    global capture_running

    auth = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
    session = requests.Session()
    session.auth = auth

    print(f"使用HTTP快照模式: {snapshot_url}")
    print("提示: 快照模式延迟较高，快速移动时易出现残影，建议改用 RTSP")
    error_count = 0

    while capture_running:
        try:
            response = session.get(
                snapshot_url,
                params={"time": int(time.time() * 1000)},
                headers={"Cache-Control": "no-cache"},
                timeout=2,
            )

            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                if frame is not None:
                    if frame.shape[1] != PROCESS_WIDTH or frame.shape[0] != PROCESS_HEIGHT:
                        frame = cv2.resize(
                            frame,
                            (PROCESS_WIDTH, PROCESS_HEIGHT),
                            interpolation=cv2.INTER_AREA,
                        )
                    _publish_gray(frame)
                    error_count = 0
                else:
                    error_count += 1
            else:
                error_count += 1
                if error_count % 10 == 1:
                    print(f"快照获取失败: HTTP {response.status_code}")

        except Exception as e:
            error_count += 1
            if error_count % 10 == 1:
                print(f"快照获取失败: {e}")
            time.sleep(0.1)

        time.sleep(0.08)

    session.close()


def capture_thread_opencv():
    """OpenCV RTSP（无 FFmpeg 时的回退方案）。"""
    global capture_running
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    )

    while capture_running:
        cap = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            time.sleep(3)
            continue

        print(f"OpenCV RTSP: {camera_path}")

        for _ in range(RTSP_FLUSH_FRAMES * 2):
            cap.grab()

        while capture_running:
            for _ in range(RTSP_FLUSH_FRAMES):
                cap.grab()

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                print("RTSP 断流，重连...")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] != PROCESS_WIDTH or gray.shape[0] != PROCESS_HEIGHT:
                gray = cv2.resize(
                    gray,
                    (PROCESS_WIDTH, PROCESS_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            _publish_gray(gray)

        cap.release()
        time.sleep(1)


def _build_ffmpeg_cmd(gpu: bool) -> list[str]:
    ff = _ffmpeg_bin()
    cmd = [ff, "-hide_banner", "-loglevel", "warning"]
    if gpu:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend([
        "-rtsp_transport", "tcp",
        "-rtsp_flags", "prefer_tcp",
        # HEVC 子码流：勿 discardcorrupt / reorder_queue_size=0，否则丢参考帧报 RPS 错
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-err_detect", "ignore_err",
        "-probesize", "5000000",
        "-analyzeduration", "5000000",
        "-i", camera_path,
        "-an",
        "-vf", f"fps={CAPTURE_FPS},scale={PROCESS_WIDTH}:{PROCESS_HEIGHT},format=gray",
        "-pix_fmt", "gray",
        "-vsync", "drop",
        "-f", "rawvideo",
        "-",
    ])
    return cmd


def _drain_stream(stream) -> None:
    """防止 stderr 缓冲区塞满导致 ffmpeg 卡死。"""
    if stream is None:
        return
    try:
        while capture_running:
            chunk = stream.read(4096)
            if not chunk:
                break
    except Exception:
        pass


def capture_thread_rtsp_ffmpeg(gpu: bool = False):
    """FFmpeg 拉流；gpu=True 时用 NVIDIA NVDEC 硬解 H.265。"""
    global capture_running

    ff = _ffmpeg_bin()
    if not ff:
        print("未找到 ffmpeg，回退 OpenCV")
        capture_thread_opencv()
        return

    width = PROCESS_WIDTH
    height = PROCESS_HEIGHT
    frame_size = width * height
    gpu_failures = 0

    while capture_running:
        mode = "CUDA 硬解" if gpu else "CPU 软解"
        print(f"FFmpeg {mode}: {camera_path}")

        try:
            process = subprocess.Popen(
                _build_ffmpeg_cmd(gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size,
            )
        except Exception as e:
            print(f"ffmpeg 启动失败: {e}")
            if gpu:
                gpu = False
                continue
            capture_thread_opencv()
            return

        threading.Thread(target=_drain_stream, args=(process.stderr,), daemon=True).start()

        error_count = 0
        got_first_frame = False
        connect_deadline = time.time() + 15.0

        while capture_running:
            try:
                raw_frame = process.stdout.read(frame_size)
            except Exception:
                break

            if len(raw_frame) == frame_size:
                error_count = 0
                got_first_frame = True
                gray = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width))
                _publish_gray(gray)
                continue

            if process.poll() is not None:
                break

            if not got_first_frame and time.time() < connect_deadline:
                # HEVC 需等待首个关键帧，期间 read 可能返回空
                time.sleep(0.05)
                continue

            error_count += 1
            if error_count > 60:
                print("ffmpeg 读帧超时，重连...")
                break
            time.sleep(0.02)

        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()

        if not capture_running:
            break

        if gpu and not got_first_frame:
            gpu_failures += 1
            if gpu_failures >= 2:
                print("CUDA 硬解连续失败，回退 FFmpeg CPU 软解")
                gpu = False
                with stats_lock:
                    stats["decode_backend"] = "FFmpeg"
        elif not got_first_frame and not gpu:
            print("FFmpeg 软解失败，回退 OpenCV")
            capture_thread_opencv()
            return

        time.sleep(1)


def start_capture_thread():
    global capture_running, capture_thread
    with worker_lock:
        if capture_thread is not None and capture_thread.is_alive():
            return
        capture_running = True
        capture_thread = threading.Thread(target=capture_thread_func, daemon=True)
        capture_thread.start()
    print("帧捕获线程已启动")


def start_detection_thread():
    global detection_running, detection_thread
    with worker_lock:
        if detection_thread is not None and detection_thread.is_alive():
            return
        detection_running = True
        detection_thread = threading.Thread(target=detection_thread_func, daemon=True)
        detection_thread.start()
    print("检测线程已启动")


def start_tag_worker_thread():
    global _tag_worker_running, tag_worker_thread
    with worker_lock:
        if tag_worker_thread is not None and tag_worker_thread.is_alive():
            return
        _tag_worker_running = True
        tag_worker_thread = threading.Thread(target=tag_worker_func, daemon=True)
        tag_worker_thread.start()
    print("标签识别线程已启动")


def start_render_thread():
    global render_running, render_thread
    with worker_lock:
        if render_thread is not None and render_thread.is_alive():
            return
        render_running = True
        render_thread = threading.Thread(target=render_thread_func, daemon=True)
        render_thread.start()
    print("渲染线程已启动")


def start_workers():
    start_tag_worker_thread()
    start_render_thread()
    start_capture_thread()
    start_detection_thread()


def generate_frames():
    global latest_jpeg, latest_jpeg_id

    if not capture_running or not detection_running:
        start_workers()

    while latest_jpeg is None:
        time.sleep(0.02)

    last_sent_id = -1

    while True:
        with display_lock:
            if latest_jpeg is None or latest_jpeg_id == last_sent_id:
                time.sleep(0.001)
                continue
            data = latest_jpeg
            frame_id = latest_jpeg_id

        last_sent_id = frame_id
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    with stats_lock:
        current_stats = dict(stats)

    events = [
        {
            "tag_id": e.tag_id,
            "op": e.op,
            "op_label": "入库" if e.op == "in" else "出库",
            "displacement": round(e.displacement, 1),
            "duration": round(e.duration, 2),
            "timestamp": e.timestamp,
        }
        for e in tag_tracker.recent_events
    ]

    return jsonify({
        **current_stats,
        "events": events,
        "rejected_short": tag_tracker.rejected_short,
        "rejected_static": tag_tracker.rejected_static,
    })


@app.route('/api/benchmark')
def api_benchmark():
    frame = np.full((PROCESS_HEIGHT, PROCESS_WIDTH), 255, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    tag_size = max(48, min(PROCESS_WIDTH, PROCESS_HEIGHT) // 5)
    tag = cv2.aruco.generateImageMarker(aruco_dict, 7, tag_size)
    y = (PROCESS_HEIGHT - tag_size) // 2
    x = (PROCESS_WIDTH - tag_size) // 2
    frame[y:y + tag_size, x:x + tag_size] = tag

    iterations = 200
    start = time.perf_counter()
    detections = 0
    for _ in range(iterations):
        _, ids = aruco_detector.detect(frame)
        detections += 0 if ids is None else len(ids)
    elapsed = time.perf_counter() - start

    return jsonify({
        "backend": aruco_detector.backend,
        "dictionary": aruco_detector.dictionary,
        "frame_size": f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}",
        "iterations": iterations,
        "detections": detections,
        "fps": round(iterations / elapsed, 1),
        "avg_ms": round((elapsed / iterations) * 1000, 3),
    })


def _benchmark_detector_on_frame(
    detector: ArucoDetector,
    frame: np.ndarray,
    detect_width: int,
    iterations: int,
) -> dict:
    h, w = frame.shape[:2]
    scale = min(1.0, detect_width / w)
    if scale < 1.0:
        sample = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = frame

    detections = 0
    start = time.perf_counter()
    for _ in range(iterations):
        _, ids = detector.detect(sample)
        detections += 0 if ids is None else len(ids)
    elapsed = time.perf_counter() - start

    return {
        "backend": detector.backend,
        "detect_size": f"{sample.shape[1]}x{sample.shape[0]}",
        "iterations": iterations,
        "detections": detections,
        "fps": round(iterations / elapsed, 1),
        "avg_ms": round((elapsed / iterations) * 1000, 2),
    }


@app.route('/api/benchmark/latest')
def api_benchmark_latest():
    with gray_lock:
        if latest_gray is None:
            return jsonify({"error": "no frame available"}), 503
        frame = latest_gray.copy()

    widths = sorted({PROCESS_WIDTH, DETECT_WIDTH, 360, 320}, reverse=True)
    iterations = 20
    results = []
    for width in widths:
        results.append(_benchmark_detector_on_frame(aruco_detector, frame, width, iterations))

    return jsonify({
        "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "dictionary": aruco_detector.dictionary,
        "results": results,
    })


if __name__ == '__main__':
    gpu_ok = _ffmpeg_supports_cuda()
    print("启动 ArUco 车辆监控服务器...")
    print(f"采集: {CAPTURE_BACKEND} {PROCESS_WIDTH}x{PROCESS_HEIGHT}@{CAPTURE_FPS}")
    print(f"GPU: {'FFmpeg CUDA 可用' if gpu_ok else '未检测到 FFmpeg CUDA'}")
    print(f"检测: OpenCV {aruco_detector.dictionary}")
    print(f"渲染: {RENDER_FPS}fps JPEG q={JPEG_QUALITY}")
    print("摄像头建议: 曝光手动 1/500~1/2000, 关闭 WDR/3DNR, 增益适中")
    print("访问 http://localhost:5000 查看分析画面")
    start_workers()
    time.sleep(1)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
