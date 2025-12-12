"""
AprilTag 检测方案 - 比 ArUco 更容易识别
特点：边框更粗、对比度要求更低、检测更稳定
"""
import datetime
import time
import json
import cv2
import numpy as np
from socket import *
import cv2 as cv
from flask import Flask, Response, render_template
import os

app = Flask(__name__, template_folder='templates')

udpSocket = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.81.20', 6789)

# 摄像头设置
camera_path = "rtsp://admin:Zf123456789!@192.168.81.82/h264/ch1/sub/av_stream"

queue = {}

def post_code(obj):
    op = 'none'
    if 'out' not in obj:
        return
    op = 'out' if (obj['_in'] - obj['out']) > 0 else 'in'
    resjson = json.dumps({"op": op, "code": str(obj['_id'])})
    udpSocket.sendto(resjson.encode('utf-8'), addr)

def ana_image_apriltag(image):
    """使用 AprilTag 检测"""
    if image is None or image.size == 0:
        return image
    
    display_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # AprilTag 检测器 - 使用 tag36h11 字典（最常用，识别率最高）
    # 可选字典: tag16h5, tag25h9, tag36h11, tagCircle21h7, tagStandard41h12
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    
    parameters = cv2.aruco.DetectorParameters()
    # AprilTag 优化参数
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # 画出检测结果
    cv2.aruco.drawDetectedMarkers(display_img, corners, ids)
    
    if corners is not None and len(corners) > 0:
        ids_flat = ids.flatten()
        for (mr, marker_id) in zip(corners, ids_flat):
            corner_points = mr.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corner_points
            topRight_coord = (int(topRight[0]), int(topRight[1]))
            
            cX = int((topLeft[0] + bottomRight[0]) / 2)
            cY = int((topLeft[1] + bottomRight[1]) / 2)
            cv2.putText(display_img, f"ID:{marker_id}", (cX - 30, cY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if marker_id not in queue:
                queue[marker_id] = {"_id": int(marker_id), "_in": int(topRight_coord[1]), "first_time": time.time()}
            else:
                queue[marker_id]['out'] = int(topRight_coord[1])
            print(f"AprilTag 检测到 ID: {marker_id}, 位置: {topRight_coord}")
    
    # 清理队列
    po = []
    for key in queue:
        if (time.time() - queue[key]["first_time"] > 3):
            post_code(queue[key])
            po.append(key)
    for i in po:
        queue.pop(i)
    
    status_text = f"AprilTag 检测: {len(corners) if corners is not None else 0} 个"
    cv2.putText(display_img, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
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
                processed_frame = ana_image_apriltag(frame)
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
    print("启动 AprilTag 检测服务器...")
    print("访问 http://localhost:5000 查看")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

