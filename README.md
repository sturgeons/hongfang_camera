AprilTag vehicle monitor
========================

Monitors a camera stream for AprilTag 36h11 labels on vehicles. A tag moving
from the top of the image to the bottom is reported as `in`; bottom to top is
reported as `out`.

Run:

```bash
uv run python main.py
```

The web UI is served at `http://localhost:5000`.

Performance defaults
--------------------

The runtime is tuned for low latency:

- FFmpeg is the default capture backend.
- Frames are converted to grayscale and resized to `640x360` before detection.
- `pupil-apriltags` is the default detector when installed; OpenCV remains a
  fallback.
- Only the newest frame is kept; old frames are dropped when detection lags.

Useful environment variables:

```bash
CAMERA_IP=192.168.81.82
CAMERA_USER=admin
CAMERA_PASS=...
UDP_HOST=192.168.81.20
UDP_PORT=6789

CAPTURE_BACKEND=ffmpeg      # ffmpeg or opencv
DETECTOR_BACKEND=pupil      # pupil or opencv
PROCESS_WIDTH=640
PROCESS_HEIGHT=360
DETECT_WIDTH=480
CAPTURE_FPS=20
RTSP_STREAM=sub
RTSP_CODEC=h264
PUPIL_QUAD_DECIMATE=2.0
PUPIL_THREADS=4

# Optional detection crop as frame ratios.
ROI_X_MIN=0.0
ROI_X_MAX=1.0
ROI_Y_MIN=0.0
ROI_Y_MAX=1.0
```

If tags are too small to detect reliably, raise `PROCESS_WIDTH` and
`PROCESS_HEIGHT` first, for example `960x540`. If latency or CPU load is too
high, lower `DETECT_WIDTH` first, for example `360` or `320`. If the camera
view contains large areas where tags never appear, set the ROI variables to
crop detection to the driving lane.

With the `pupil` detector, `PUPIL_QUAD_DECIMATE` is the main speed/accuracy
knob. Use `2.0` first; if tags are missed, try `1.5` or `1.0`. If speed is
still too low and tags are large, try `3.0`.

Diagnostics
-----------

Use `http://localhost:5000/api/stats` to separate capture and detection
problems:

- `fps_capture`: frames actually received from the camera pipeline.
- `fps_detect`: frames processed by the detection thread.
- `capture_interval_ms`: time between received frames.
- `detect_ms`: AprilTag detection and tracking time for one frame.
- `frame_age_ms`: how old the latest frame was when detection started.

Use `http://localhost:5000/api/benchmark` to test AprilTag detector speed
without the camera. If this is fast but `fps_capture` is low, C++ will not fix
the main bottleneck; the problem is RTSP, camera encoding, network, or FFmpeg
decode settings.

Use `http://localhost:5000/api/benchmark/latest` after the service has received
camera frames. It benchmarks the latest real frame at multiple detection
resolutions and, when available, compares OpenCV and `pupil-apriltags`. This is
the best way to pick `DETECT_WIDTH` and `DETECTOR_BACKEND` for the actual scene.
