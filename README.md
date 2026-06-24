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
DETECT_PADDING=24
CAPTURE_FPS=20
RTSP_STREAM=sub
RTSP_CODEC=h264
PUPIL_QUAD_DECIMATE=2.0
PUPIL_THREADS=4

# Motion-gated detection. Only moving regions are scanned for AprilTags.
MOTION_GATE=true
MOTION_MIN_AREA=80
MOTION_THRESHOLD=18
MOTION_EXPAND=0.18
MOTION_FULL_SCAN_INTERVAL=0
MOTION_MAX_AREA_RATIO=0.45
MOTION_BG_ALPHA=0.02
IDLE_SCAN_INTERVAL=5
STARTUP_SCAN_FRAMES=12
SCAN_HOLD_SECONDS=1.2
SCAN_ROI_MIN_WIDTH=260
SCAN_ROI_MIN_HEIGHT=180
SCAN_FULL_EVERY=8
EDGE_SCAN_INTERVAL=2
EDGE_SCAN_HEIGHT_RATIO=0.38

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

The default detector is motion-gated. This avoids scanning the entire complex
scene when only a vehicle-sized region matters. If the purple ROI rectangle in
the web view is too small or misses the vehicle, raise `MOTION_EXPAND`; if it
reacts to too much background noise, raise `MOTION_MIN_AREA` or
`MOTION_THRESHOLD`. If static compression noise still creates a nearly full
frame ROI, lower `MOTION_MAX_AREA_RATIO`. To disable this strategy, set
`MOTION_GATE=false`.

For stuttering vehicles or unstable motion boxes, keep scan windows forgiving:
`SCAN_HOLD_SECONDS` keeps scanning the last vehicle ROI after motion pauses,
`SCAN_ROI_MIN_WIDTH/HEIGHT` prevent the ROI from becoming too tight, and
`SCAN_FULL_EVERY` periodically scans the configured full ROI while motion is
active to recover tags that were just outside the motion box.
`IDLE_SCAN_INTERVAL` performs a low-rate fallback scan when no motion is
detected, which helps with vehicles that stop before the tag is read.
`STARTUP_SCAN_FRAMES` forces full scans immediately after startup so the first
vehicle is not missed while the motion background model is warming up.
`EDGE_SCAN_INTERVAL` scans the top and bottom bands regularly. This protects
against tags near the entry/exit edges where motion gating may not produce a
stable box.
`DETECT_PADDING` adds a white border around cropped detection images. Keep it
enabled for tags close to the image edge; AprilTag detectors need visible quiet
space around the black tag border.

Diagnostics
-----------

Use `http://localhost:5000/api/stats` to separate capture and detection
problems:

- `fps_capture`: frames actually received from the camera pipeline.
- `fps_detect`: frames processed by the lightweight motion loop.
- `fps_tag`: AprilTag detector jobs completed per second.
- `capture_interval_ms`: time between received frames.
- `detect_ms`: lightweight motion loop time for one frame.
- `motion_ms`: motion ROI calculation time.
- `tag_ms`: native AprilTag detector time for one ROI.
- `render_ms`: display overlay rendering time, now handled by a separate thread.
- `frame_age_ms`: how old the latest frame was when detection started.

Use `http://localhost:5000/api/benchmark` to test AprilTag detector speed
without the camera. If this is fast but `fps_capture` is low, C++ will not fix
the main bottleneck; the problem is RTSP, camera encoding, network, or FFmpeg
decode settings.

Use `http://localhost:5000/api/benchmark/latest` after the service has received
camera frames. It benchmarks the latest real frame at multiple detection
resolutions and, when available, compares OpenCV and `pupil-apriltags`. This is
the best way to pick `DETECT_WIDTH` and `DETECTOR_BACKEND` for the actual scene.
