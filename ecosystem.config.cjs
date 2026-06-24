module.exports = {
  apps: [
    {
      name: "hongfang-camera",
      cwd: "/root/hongfang_camera",
      script: ".venv/bin/python",
      args: "main.py",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "512M",
      env: {
        JPEG_QUALITY: "55",
        RENDER_FPS: "20",
        CAPTURE_FPS: "20",
      },
    },
  ],
};
