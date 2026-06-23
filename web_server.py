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

try:
    from pupil_apriltags import Detector as PupilDetector
except Exception:
    PupilDetector = None

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
DETECTOR_BACKEND = os.getenv(
    "DETECTOR_BACKEND",
    "pupil" if PupilDetector is not None else "opencv",
).lower()
USE_SNAPSHOT_MODE = _env_bool("USE_SNAPSHOT_MODE", False)
RTSP_FLUSH_FRAMES = _env_int("RTSP_FLUSH_FRAMES", 1)
USE_GPU_DECODE = _env_bool("USE_GPU_DECODE", False)

# 核心性能策略：采集端直接输出算法需要的低分辨率灰度帧。
# 车辆进出只依赖纵向轨迹，不需要对 720p/1080p 整帧做 AprilTag 检测。
PROCESS_WIDTH = _env_int("PROCESS_WIDTH", 640)
PROCESS_HEIGHT = _env_int("PROCESS_HEIGHT", 360)
DETECT_WIDTH = _env_int("DETECT_WIDTH", 480)
ROI_X_MIN = _env_float("ROI_X_MIN", 0.0)
ROI_X_MAX = _env_float("ROI_X_MAX", 1.0)
ROI_Y_MIN = _env_float("ROI_Y_MIN", 0.0)
ROI_Y_MAX = _env_float("ROI_Y_MAX", 1.0)
CAPTURE_FPS = _env_int("CAPTURE_FPS", 20)
DISPLAY_MAX_WIDTH = _env_int("DISPLAY_MAX_WIDTH", 960)
RENDER_INTERVAL = _env_float("RENDER_INTERVAL", 0.08)
PUPIL_THREADS = min(_env_int("PUPIL_THREADS", 4), os.cpu_count() or 4)
PUPIL_QUAD_DECIMATE = _env_float("PUPIL_QUAD_DECIMATE", 2.0)
PUPIL_DECODE_SHARPENING = _env_float("PUPIL_DECODE_SHARPENING", 0.25)
OPENCV_MIN_PERIMETER = _env_float("OPENCV_MIN_PERIMETER", 0.03)
OPENCV_ADAPTIVE_MAX = _env_int("OPENCV_ADAPTIVE_MAX", 15)
BENCHMARK_PUPIL = _env_bool("BENCHMARK_PUPIL", False)

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

latest_gray: np.ndarray | None = None
latest_gray_id = 0
gray_lock = threading.Lock()
gray_ready = threading.Event()

latest_display = None
latest_display_id = 0
display_lock = threading.Lock()

capture_running = False
detection_running = False
worker_lock = threading.Lock()
capture_thread: threading.Thread | None = None
detection_thread: threading.Thread | None = None
stats_lock = threading.Lock()
stats = {
    "active_count": 0,
    "total_in": 0,
    "total_out": 0,
    "fps_capture": 0.0,
    "fps_detect": 0.0,
    "capture_interval_ms": 0.0,
    "detect_ms": 0.0,
    "render_ms": 0.0,
    "frame_age_ms": 0.0,
    "last_detections": 0,
    "queue_depth": 0,
    "dropped_frames": 0,
    "decode_backend": "unknown",
    "detect_backend": DETECTOR_BACKEND,
    "frame_size": f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}",
    "detect_size": "",
    "detect_roi": "",
}


class AprilTagDetector:
    def __init__(self, backend: str):
        self.backend = backend
        self._opencv_detector = None
        self._pupil_detector = None

        if backend == "pupil":
            if PupilDetector is None:
                raise RuntimeError("pupil-apriltags is not installed")
            self._pupil_detector = PupilDetector(
                families="tag36h11",
                nthreads=PUPIL_THREADS,
                quad_decimate=PUPIL_QUAD_DECIMATE,
                quad_sigma=0.0,
                refine_edges=False,
                decode_sharpening=PUPIL_DECODE_SHARPENING,
            )
            return

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        params.minMarkerPerimeterRate = OPENCV_MIN_PERIMETER
        params.maxMarkerPerimeterRate = 4.0
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = OPENCV_ADAPTIVE_MAX
        params.adaptiveThreshWinSizeStep = 10
        self._opencv_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.backend = "opencv"

    def detect(self, gray: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        gray = np.ascontiguousarray(gray)
        if self.backend == "pupil":
            results = self._pupil_detector.detect(gray)
            if not results:
                return [], None
            corners = [det.corners.reshape(1, 4, 2).astype(np.float32) for det in results]
            ids = np.array([int(det.tag_id) for det in results], dtype=np.int32).reshape(-1, 1)
            return corners, ids

        corners, ids, _ = self._opencv_detector.detectMarkers(gray)
        if ids is None:
            return [], None
        return corners, ids


try:
    april_detector = AprilTagDetector(DETECTOR_BACKEND)
except Exception as exc:
    print(f"检测器 {DETECTOR_BACKEND} 初始化失败，回退 OpenCV: {exc}")
    april_detector = AprilTagDetector("opencv")
    DETECTOR_BACKEND = "opencv"
    stats["detect_backend"] = "opencv"

tag_tracker = TagTracker()

# 检测与渲染分离：检测全速，画面按 RENDER_INTERVAL 刷新
_overlay_lock = threading.Lock()
_overlay = {
    "gray": None,
    "markers": [],
    "tracks": [],
}


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


def detection_window(gray: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    """Return cropped/scaled image plus offset and scale back to full frame."""
    h, w = gray.shape[:2]
    x1 = int(w * _clip_ratio(ROI_X_MIN))
    x2 = int(w * _clip_ratio(ROI_X_MAX))
    y1 = int(h * _clip_ratio(ROI_Y_MIN))
    y2 = int(h * _clip_ratio(ROI_Y_MAX))

    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h

    crop = gray[y1:y2, x1:x2]
    scale = min(1.0, DETECT_WIDTH / crop.shape[1]) if crop.shape[1] else 1.0
    if scale < 1.0:
        detect_img = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detect_img = crop

    with stats_lock:
        stats["detect_size"] = f"{detect_img.shape[1]}x{detect_img.shape[0]}"
        stats["detect_roi"] = f"{x1},{y1},{x2},{y2}"

    return detect_img, x1, y1, scale


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


def detect_and_track(gray: np.ndarray) -> int:
    """纯检测+追踪，不做画面渲染。"""
    h, w = gray.shape[:2]
    tag_tracker.set_frame_height(h)

    detect_img, offset_x, offset_y, scale = detection_window(gray)
    corners, ids = april_detector.detect(detect_img)
    corners = map_corners_to_frame(corners, offset_x, offset_y, scale)
    detections: list[tuple[int, float, float]] = []
    markers: list[tuple[int, float, float, np.ndarray]] = []

    if corners and ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            cx, cy = marker_center(marker_corners)
            detections.append((int(marker_id), cx, cy))
            markers.append((int(marker_id), cx, cy, marker_corners))

    events = tag_tracker.update(detections)
    for event in events:
        post_event(event.tag_id, event.op)
        with stats_lock:
            if event.op == "in":
                stats["total_in"] += 1
            else:
                stats["total_out"] += 1

    with stats_lock:
        stats["active_count"] = len(tag_tracker.active_tracks())

    with _overlay_lock:
        _overlay["gray"] = gray
        _overlay["markers"] = markers
        _overlay["tracks"] = list(tag_tracker.active_tracks().values())

    return len(detections)


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
        cap_fps = stats["fps_capture"]
        backend = stats.get("decode_backend", "?")
        active = stats["active_count"]

    status = (
        f"Tag:{det_count} Trk:{active} | "
        f"Cap{cap_fps} Det{det_fps} [{backend}]"
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


def _publish_display(display_img: np.ndarray) -> None:
    global latest_display, latest_display_id

    with display_lock:
        latest_display = display_img
        latest_display_id += 1


def detection_thread_func():
    """检测全速跑最新帧；画面按 RENDER_INTERVAL 刷新。"""
    global detection_running

    print(
        f"检测线程: {april_detector.backend} "
        f"frame={PROCESS_WIDTH}x{PROCESS_HEIGHT} display<={DISPLAY_MAX_WIDTH}px"
    )

    last_render = 0.0

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

        now = time.time()
        if now - last_render >= RENDER_INTERVAL:
            try:
                render_t0 = time.perf_counter()
                frame = render_display(det_count)
                render_ms = (time.perf_counter() - render_t0) * 1000
                with stats_lock:
                    stats["render_ms"] = round(render_ms, 2)
                if frame is not None:
                    _publish_display(frame)
            except Exception as e:
                print(f"渲染错误: {e}")
            last_render = now


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


def start_workers():
    start_capture_thread()
    start_detection_thread()


def generate_frames():
    global latest_display, latest_display_id

    if not capture_running or not detection_running:
        start_workers()

    while latest_display is None:
        time.sleep(0.05)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    last_sent_id = -1

    while True:
        with display_lock:
            if latest_display is None or latest_display_id == last_sent_id:
                time.sleep(0.005)
                continue
            frame = latest_display.copy()
            frame_id = latest_display_id

        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue

        last_sent_id = frame_id
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


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
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_size = max(48, min(PROCESS_WIDTH, PROCESS_HEIGHT) // 5)
    tag = cv2.aruco.generateImageMarker(aruco_dict, 7, tag_size)
    y = (PROCESS_HEIGHT - tag_size) // 2
    x = (PROCESS_WIDTH - tag_size) // 2
    frame[y:y + tag_size, x:x + tag_size] = tag

    iterations = 200
    start = time.perf_counter()
    detections = 0
    for _ in range(iterations):
        _, ids = april_detector.detect(frame)
        detections += 0 if ids is None else len(ids)
    elapsed = time.perf_counter() - start

    return jsonify({
        "backend": april_detector.backend,
        "frame_size": f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}",
        "iterations": iterations,
        "detections": detections,
        "fps": round(iterations / elapsed, 1),
        "avg_ms": round((elapsed / iterations) * 1000, 3),
    })


def _benchmark_detector_on_frame(
    detector: AprilTagDetector,
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

    detectors = [april_detector]
    if BENCHMARK_PUPIL and april_detector.backend != "pupil" and PupilDetector is not None:
        try:
            detectors.append(AprilTagDetector("pupil"))
        except Exception:
            pass

    for detector in detectors:
        for width in widths:
            results.append(_benchmark_detector_on_frame(detector, frame, width, iterations))

    return jsonify({
        "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "results": results,
    })


if __name__ == '__main__':
    gpu_ok = _ffmpeg_supports_cuda()
    print("启动 AprilTag 车辆监控服务器...")
    print(f"采集: {CAPTURE_BACKEND} {PROCESS_WIDTH}x{PROCESS_HEIGHT}@{CAPTURE_FPS}")
    print(f"GPU: {'FFmpeg CUDA 可用' if gpu_ok else '未检测到 FFmpeg CUDA'}")
    print(f"检测: {april_detector.backend} qd={PUPIL_QUAD_DECIMATE if april_detector.backend == 'pupil' else '-'}")
    print("摄像头建议: 曝光手动 1/500~1/2000, 关闭 WDR/3DNR, 增益适中")
    print("访问 http://localhost:5000 查看分析画面")
    start_workers()
    time.sleep(1)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
