import datetime
import time
import json
import cv2
import subprocess
import numpy
from socket import *
import queue
import threading
import cv2 as cv
import subprocess as sp


udpSocket = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.81.20', 6789)
command = ""


# 自行设置
rtmpUrl = "rtmp://192.168.81.16:1935/live/livestream"
camera_path = "rtsp://admin:Zf123456789!@192.168.81.82/h265/ch1/sub/av_stream"

def post_code(obj):
    op = 'none'
    if('out' not in obj): return
    op = 'out' if (obj['_in'] - obj['out']) > 0 else 'in'
    resjson = json.dumps({"op": op,"code": str(obj['_id'])})
    udpSocket.sendto(resjson.encode('utf-8'), addr)

queue={}
def ana_image(image):
    cv2.cuda.setDevice(0)
    gpu_frame=cv2.cuda_GpuMat()
    gpu_frame.upload(image)
    gpu_gray=cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    gpu_color_img = cv2.cuda.cvtColor(gpu_gray, cv2.COLOR_GRAY2BGR)
    img=gpu_color_img.download()
    #设置预定义的字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    #使用默认值初始化检测器参数
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    #使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)
    #画出标志位置
    cv2.aruco.drawDetectedMarkers(img, corners,ids)
    if len(corners)>0:
        ids=ids.flatten()
        for (mr,id) in zip(corners,ids):
            corners=mr.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            if id not in queue:
                queue[id]={"_id":id,"_in":int(topRight[1]),"first_time":time.time()}
            else:
                queue[id]['out']=int(topRight[1])
            print('-----')
            print(id)
            print(topRight)
    po=[]
    for key in queue:
        if(time.time()-queue[key]["first_time"]>3):
            post_code(queue[key])
            po.append(key)
    for i in po:
        queue.pop(i)

if __name__ == '__main__':
    print("开始运行---------------")
    # ffmpeg command 保存进程参数
    while True:
        # 打开RTSP流，也可以用0，调用本地视频流，并取出视频流的帧率、帧宽、帧高
        cap = cv2.VideoCapture(camera_path,cv2.CAP_GSTREAMER)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()  # 从视频流中获取一帧
            if not ret:
                break
                # raise IOError("could't open webcamera or video")
            # 处理代码(使用AI算法)
            # writer.write(frame)  #视频保存
            image = ana_image(frame)
            cv2.imshow('Video', frame)  # 显示处理结果
            # 推流代码
            # pipe.stdin.write(image.tobytes())
            # 按下q键退出
            if cv2.waitKey(1) == ord('q'):
                break

        # 释放视频流
        cap.release()
        # 关闭窗口
        cv2.destroyAllWindows()
        # 关闭进程
        # pipe.stdin.close()
        # pipe.wait()
