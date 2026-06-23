"""
颜色标记检测方案 - 最简单的识别方式
特点：只需要彩色贴纸/卡片，无需打印二维码
支持多种颜色，每种颜色代表一个ID
"""
import datetime
import time
import json
import cv2
import numpy as np
from socket import *
from flask import Flask, Response, render_template
import os

app = Flask(__name__, template_folder='templates')

udpSocket = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.81.20', 6789)

camera_path = "rtsp://admin:Zf123456789!@192.168.81.82/h264/ch1/sub/av_stream"

queue = {}

# 定义颜色范围 (HSV) - 可以根据实际颜色调整
# 格式: {ID: (颜色名, HSV下限, HSV上限, 显示颜色BGR)}
COLOR_RANGES = {
    1: ("红色", np.array([0, 100, 100]), np.array([10, 255, 255]), (0, 0, 255)),
    2: ("红色2", np.array([160, 100, 100]), np.array([180, 255, 255]), (0, 0, 255)),
    3: ("绿色", np.array([35, 100, 100]), np.array([85, 255, 255]), (0, 255, 0)),
    4: ("蓝色", np.array([100, 100, 100]), np.array([130, 255, 255]), (255, 0, 0)),
    5: ("黄色", np.array([20, 100, 100]), np.array([35, 255, 255]), (0, 255, 255)),
    6: ("紫色", np.array([130, 100, 100]), np.array([160, 255, 255]), (255, 0, 255)),
    7: ("橙色", np.array([10, 100, 100]), np.array([20, 255, 255]), (0, 165, 255)),
}

# 最小检测面积（过滤噪点）
MIN_AREA = 1000

def post_code(obj):
    op = 'none'
    if 'out' not in obj:
        return
    op = 'out' if (obj['_in'] - obj['out']) > 0 else 'in'
    resjson = json.dumps({"op": op, "code": str(obj['_id'])})
    udpSocket.sendto(resjson.encode('utf-8'), addr)

def detect_color_markers(image):
    """检测彩色标记"""
    if image is None or image.size == 0:
        return image
    
    display_img = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detected_count = 0
    
    for marker_id, (color_name, lower, upper, display_color) in COLOR_RANGES.items():
        # 创建颜色掩码
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查是否接近正方形（可选，提高精度）
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            detected_count += 1
            
            # 画出检测框
            cv2.rectangle(display_img, (x, y), (x + w, y + h), display_color, 3)
            
            # 显示ID和颜色名
            center_x, center_y = x + w // 2, y + h // 2
            cv2.putText(display_img, f"ID:{marker_id}", (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
            
            # 追踪逻辑
            track_key = f"{marker_id}_{x//100}"  # 按位置区分同色多个标记
            
            if track_key not in queue:
                queue[track_key] = {
                    "_id": marker_id,
                    "_in": center_y,
                    "first_time": time.time(),
                    "color": color_name
                }
            else:
                queue[track_key]['out'] = center_y
            
            print(f"检测到 {color_name} ID:{marker_id}, 位置: ({center_x}, {center_y})")
    
    # 清理队列
    po = []
    for key in queue:
        if (time.time() - queue[key]["first_time"] > 3):
            post_code(queue[key])
            po.append(key)
    for i in po:
        queue.pop(i)
    
    # 显示状态
    status_text = f"Color markers: {detected_count}"
    cv2.putText(display_img, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 显示颜色图例
    y_offset = 60
    for marker_id, (color_name, _, _, display_color) in list(COLOR_RANGES.items())[:5]:
        cv2.rectangle(display_img, (10, y_offset), (30, y_offset + 20), display_color, -1)
        cv2.putText(display_img, f"ID{marker_id}", (35, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
    
    return display_img

def generate_frames():
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    while True:
        print(f"正在连接摄像头: {camera_path}")
        cap = cv2.VideoCapture(camera_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_path)
            
        if not cap.isOpened():
            print("无法打开摄像头，5秒后重试...")
            time.sleep(5)
            continue
        
        print("摄像头连接成功！")
        
        while True:
            for _ in range(3):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                processed_frame = detect_color_markers(frame)
            except Exception as e:
                print(f"处理错误: {e}")
                processed_frame = frame
            
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("启动颜色标记检测服务器...")
    print("支持的颜色: 红、绿、蓝、黄、紫、橙")
    print("访问 http://localhost:5000 查看")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

