import os

# Ultralytics pip-installs missing packages at runtime by default; everything it needs is pinned in uv.lock instead.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
