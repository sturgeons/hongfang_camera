import time
import json
import cv2
import cv2 as cv
import numpy as np
import subprocess
import requests
from requests.auth import HTTPDigestAuth
from socket import *
from flask import Flask, Response, render_template
import threading

app = Flask(__name__, template_folder='templates')

udpSocket = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.81.20', 6789)

# 自行设置
rtmpUrl = "rtmp://192.168.81.16:1935/live/livestream"

# 摄像头配置
CAMERA_IP = "192.168.81.82"
CAMERA_USER = "admin"
CAMERA_PASS = "Zf123456789!"

# 方式1: HTTP快照URL（海康摄像头，无残影）
# 101=主码流（高分辨率），102=子码流（低分辨率）
snapshot_url = f"http://{CAMERA_IP}/ISAPI/Streaming/channels/101/picture"

# 方式2: RTSP流（备用）
camera_path = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}/h264/ch1/sub/av_stream"

# 使用快照模式（True）还是RTSP模式（False）
USE_SNAPSHOT_MODE = True

queue = {}

# 用于线程间传递最新帧的变量
latest_frame = None
frame_lock = threading.Lock()
capture_running = False

# 预创建检测器（避免每帧重复创建，大幅提速）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
# 简化参数，提高速度
parameters.adaptiveThreshConstant = 7
parameters.adaptiveThreshWinSizeMin = 5
parameters.adaptiveThreshWinSizeMax = 21
parameters.adaptiveThreshWinSizeStep = 10
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # 关闭角点精化，提速
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

def post_code(obj):
    op = 'none'
    if 'out' not in obj:
        return
    op = 'out' if (obj['_in'] - obj['out']) > 0 else 'in'
    resjson = json.dumps({"op": op, "code": str(obj['_id'])})
    udpSocket.sendto(resjson.encode('utf-8'), addr)

def sharpen_image(gray):
    """锐化图像，减轻拖影/模糊"""
    # 使用Unsharp Mask锐化
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 2.0, blurred, -1.0, 0)
    return sharpened


def ana_image(gray):
    """分析灰度图像，直接返回灰度图，去除颜色处理"""
    if gray is None or gray.size == 0:
        return gray
    
    # 锐化图像减轻拖影
    gray = sharpen_image(gray)
    
    # 直接在灰度图上检测
    corners, ids, _ = detector.detectMarkers(gray)
    
    # 转为3通道灰度图用于显示标记（这样标记可以用彩色显示）
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 画出标志位置
    if corners is not None and len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
        ids_flat = ids.flatten()
        for (mr, marker_id) in zip(corners, ids_flat):
            corner_points = mr.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner_points
            topRight_coord = (int(topRight[0]), int(topRight[1]))
            
            cX = int((topLeft[0] + bottomRight[0]) / 2)
            cY = int((topLeft[1] + bottomRight[1]) / 2)
            cv2.putText(display_img, f"{marker_id}", (cX - 15, cY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if marker_id not in queue:
                queue[marker_id] = {"_id": int(marker_id), "_in": int(topRight_coord[1]), "first_time": time.time()}
            else:
                queue[marker_id]['out'] = int(topRight_coord[1])
            print(f"ID:{marker_id} Y:{topRight_coord[1]}")
    
    # 清理过期的标记
    po = [key for key in queue if (time.time() - queue[key]["first_time"] > 3)]
    for i in po:
        post_code(queue[i])
        queue.pop(i)
    
    return display_img

def capture_thread_func():
    """根据配置选择快照模式或RTSP模式"""
    if USE_SNAPSHOT_MODE:
        capture_thread_snapshot()
    else:
        capture_thread_rtsp()


def capture_thread_snapshot():
    """HTTP快照模式 - 完全无残影"""
    global latest_frame, capture_running
    
    # 使用Digest认证（海康摄像头要求）
    auth = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
    
    # 创建Session复用连接
    session = requests.Session()
    session.auth = auth
    
    print(f"使用HTTP快照模式: {snapshot_url}")
    error_count = 0
    
    while capture_running:
        try:
            # 发送HTTP请求获取JPEG快照
            response = session.get(snapshot_url, timeout=2, stream=True)
            
            if response.status_code == 200:
                # 解码JPEG为numpy数组
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
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
            
        # 控制帧率约10-15fps
        time.sleep(0.05)
    
    session.close()


def capture_thread_rtsp():
    """RTSP模式（备用）"""
    global latest_frame, capture_running
    
    width = 640
    height = 360
    
    while capture_running:
        print(f"正在连接摄像头: {camera_path}")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-fflags', 'nobuffer+discardcorrupt+flush_packets',
            '-flags', 'low_delay',
            '-max_delay', '0',
            '-reorder_queue_size', '0',
            '-skip_frame', 'nointra',
            '-i', camera_path,
            '-vf', 'fps=10',
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
            print("FFmpeg RTSP模式")
        except FileNotFoundError:
            print("找不到ffmpeg，回退到OpenCV...")
            capture_thread_opencv()
            return
        except Exception as e:
            print(f"ffmpeg失败: {e}")
            time.sleep(3)
            continue
        
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
                
                with frame_lock:
                    latest_frame = gray
                    
            except:
                break
        
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
        
        time.sleep(1)


def capture_thread_opencv():
    """备用：OpenCV模式（当ffmpeg不可用时）"""
    global latest_frame, capture_running
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    while capture_running:
        cap = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_path)
            
        if not cap.isOpened():
            time.sleep(3)
            continue
        
        print("OpenCV模式连接成功")
        
        while capture_running:
            # 激进丢帧：丢弃10帧只取1帧
            for _ in range(10):
                cap.grab()
            
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with frame_lock:
                latest_frame = gray
        
        cap.release()


def start_capture_thread():
    """启动帧捕获线程"""
    global capture_running
    capture_running = True
    thread = threading.Thread(target=capture_thread_func, daemon=True)
    thread.start()
    print("帧捕获线程已启动")


def generate_frames():
    """生成视频帧的生成器 - 从最新帧变量读取，避免残影"""
    global latest_frame
    
    # 确保捕获线程已启动
    if not capture_running:
        start_capture_thread()
    
    # 等待第一帧
    while latest_frame is None:
        time.sleep(0.05)
    
    # JPEG编码参数：低质量=快速
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    
    while True:
        # 获取最新帧（灰度图）
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            gray = latest_frame
        
        # 处理图像（已经是灰度图）
        try:
            processed_frame = ana_image(gray)
        except Exception as e:
            print(f"处理错误: {e}")
            processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
        if not ret:
            continue
        
        # 以MJPEG格式输出
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("启动Web服务器...")
    print("访问 http://localhost:5000 查看分析画面")
    
    # 预先启动帧捕获线程
    start_capture_thread()
    
    # 等待一下让线程初始化
    time.sleep(1)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

