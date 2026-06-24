from web_server import app, start_workers
import time


def main():
    print("启动 ArUco 车辆监控...")
    print("判定规则: 上→下=入库, 下→上=出库")
    print("摄像头建议: 曝光手动 1/500~1/2000, 关闭 WDR/3DNR")
    print("访问 http://localhost:5000 查看分析画面")

    start_workers()
    time.sleep(1)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
