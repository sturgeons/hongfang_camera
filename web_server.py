import time
import json
import cv2
import cv2 as cv
import numpy as np
import subprocess
import requests
from requests.auth import HTTPDigestAuth
from socket import *
from flask import Flask, Response, render_template, jsonify
from collections import deque
import threading

from tag_tracker import TagTracker

app = Flask(__name__, template_folder='templates')

udpSocket = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.81.20', 6789)

# 摄像头配置
CAMERA_IP = "192.168.81.82"
CAMERA_USER = "admin"
CAMERA_PASS = "Zf123456789!"

# 101=主码流(分辨率高, 利于识别)  102=子码流(延迟略低)
SNAPSHOT_CHANNEL = 101
snapshot_url = f"http://{CAMERA_IP}/ISAPI/Streaming/channels/{SNAPSHOT_CHANNEL}/picture"
camera_path = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}/h264/ch1/main/av_stream"

USE_SNAPSHOT_MODE = False
RTSP_FLUSH_FRAMES = 1  # 过多丢帧会导致快速通过的车辆漏扫

latest_gray = None
latest_gray_id = 0
gray_queue: deque[np.ndarray] = deque(maxlen=40)
gray_lock = threading.Lock()

latest_display = None
latest_display_id = 0
display_lock = threading.Lock()

capture_running = False
detection_running = False
stats_lock = threading.Lock()
stats = {
    "active_count": 0,
    "total_in": 0,
    "total_out": 0,
    "fps_capture": 0.0,
    "fps_detect": 0.0,
    "last_detections": 0,
    "queue_depth": 0,
}

# AprilTag 36h11 检测器（预创建，每帧复用）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector_params = cv2.aruco.DetectorParameters()
detector_params.adaptiveThreshConstant = 5
detector_params.minMarkerPerimeterRate = 0.008
detector_params.maxMarkerPerimeterRate = 4.0
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv.aruco.ArucoDetector(aruco_dict, detector_params)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

tag_tracker = TagTracker()


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


def preprocess_for_detection(gray: np.ndarray) -> np.ndarray:
    """提升低对比度场景下的标签边缘可见度。"""
    return clahe.apply(gray)


def ana_image(gray: np.ndarray) -> tuple[np.ndarray, int]:
    if gray is None or gray.size == 0:
        return gray, 0

    h, w = gray.shape[:2]
    tag_tracker.set_frame_height(h)

    detect_gray = preprocess_for_detection(gray)
    corners, ids, _ = detector.detectMarkers(detect_gray)

    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_zones(display_img)

    detections: list[tuple[int, float, float]] = []

    if corners is not None and len(corners) > 0 and ids is not None:
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            cx, cy = marker_center(marker_corners)
            detections.append((int(marker_id), cx, cy))

            cv2.circle(display_img, (int(cx), int(cy)), 6, (0, 255, 255), -1)
            cv2.putText(display_img, f"#{marker_id}", (int(cx) - 20, int(cy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

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
        det_fps = stats["fps_detect"]
        cap_fps = stats["fps_capture"]

    for track in tag_tracker.active_tracks().values():
        if len(track.y_samples) >= 2:
            tail = 6
            xs = track.x_samples[-tail:]
            ys = track.y_samples[-tail:]
            pts = np.array([[int(x), int(y)] for x, y in zip(xs, ys)], dtype=np.int32)
            cv2.polylines(display_img, [pts], False, (0, 180, 255), 1)

    status = (
        f"AprilTag: {len(detections)} | Track: {stats['active_count']} | "
        f"Cap {cap_fps}fps Det {det_fps}fps"
    )
    cv2.putText(display_img, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

    return display_img, len(detections)


_capture_count = 0
_capture_t0 = time.time()
_detect_count = 0
_detect_t0 = time.time()


def _publish_gray(gray: np.ndarray) -> None:
    """采集线程发布灰度帧，检测线程按序消费。"""
    global latest_gray, latest_gray_id, _capture_count, _capture_t0

    with gray_lock:
        gray_queue.append(gray)
        latest_gray = gray
        latest_gray_id += 1
        _capture_count += 1
        elapsed = time.time() - _capture_t0
        if elapsed >= 2.0:
            with stats_lock:
                stats["fps_capture"] = round(_capture_count / elapsed, 1)
                stats["queue_depth"] = len(gray_queue)
            _capture_count = 0
            _capture_t0 = time.time()


def _publish_display(display_img: np.ndarray, detections: int) -> None:
    global latest_display, latest_display_id, _detect_count, _detect_t0

    with display_lock:
        latest_display = display_img
        latest_display_id += 1
        _detect_count += 1
        elapsed = time.time() - _detect_t0
        if elapsed >= 2.0:
            with stats_lock:
                stats["fps_detect"] = round(_detect_count / elapsed, 1)
                stats["last_detections"] = detections
            _detect_count = 0
            _detect_t0 = time.time()


def detection_thread_func():
    """独立检测线程：按序处理采集帧，不依赖 Web 显示速度。"""
    global detection_running

    print("检测线程已启动（与 Web 显示解耦）")

    while detection_running:
        with gray_lock:
            if not gray_queue:
                time.sleep(0.001)
                continue
            gray = gray_queue.popleft()

        try:
            processed, det_count = ana_image(gray)
        except Exception as e:
            print(f"检测错误: {e}")
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            det_count = 0

        _publish_display(processed, det_count)

    print("检测线程已停止")


def capture_thread_func():
    if USE_SNAPSHOT_MODE:
        capture_thread_snapshot()
    else:
        capture_thread_rtsp()


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


def capture_thread_rtsp():
    """优先 OpenCV 低延迟 RTSP；失败时回退 FFmpeg。"""
    global capture_running
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    )

    while capture_running:
        cap = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            if capture_thread_rtsp_ffmpeg():
                return
            time.sleep(3)
            continue

        print(f"RTSP 低延迟模式: {camera_path}")

        # 连接后先清空缓冲
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
            _publish_gray(gray)

        cap.release()
        time.sleep(1)


def capture_thread_rtsp_ffmpeg() -> bool:
    global capture_running

    width = 1280
    height = 720

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("OpenCV RTSP 失败且未找到 ffmpeg")
        return False

    while capture_running:
        print(f"FFmpeg RTSP 回退: {camera_path}")

        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-fflags', 'nobuffer+discardcorrupt+flush_packets',
            '-flags', 'low_delay',
            '-max_delay', '0',
            '-reorder_queue_size', '0',
            '-i', camera_path,
            '-vf', 'fps=20',
            '-vsync', 'drop',
            '-an',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-'
        ]

        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
        except Exception as e:
            print(f"ffmpeg 启动失败: {e}")
            return False

        frame_size = width * height
        error_count = 0

        while capture_running:
            try:
                raw_frame = process.stdout.read(frame_size)

                if len(raw_frame) != frame_size:
                    error_count += 1
                    if error_count > 50:
                        break
                    continue

                error_count = 0
                gray = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width))
                _publish_gray(gray)

            except Exception:
                break

        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()

        time.sleep(1)

    return True


def start_capture_thread():
    global capture_running
    capture_running = True
    thread = threading.Thread(target=capture_thread_func, daemon=True)
    thread.start()
    print("帧捕获线程已启动")


def start_detection_thread():
    global detection_running
    detection_running = True
    thread = threading.Thread(target=detection_thread_func, daemon=True)
    thread.start()


def start_workers():
    start_capture_thread()
    start_detection_thread()


def generate_frames():
    global latest_display, latest_display_id

    if not capture_running:
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


if __name__ == '__main__':
    print("启动 AprilTag 车辆监控服务器...")
    print("摄像头建议: 曝光手动 1/500~1/2000, 关闭 WDR/3DNR, 增益适中")
    print("访问 http://localhost:5000 查看分析画面")
    start_workers()
    time.sleep(1)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
