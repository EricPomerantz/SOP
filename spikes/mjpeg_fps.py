"""Throwaway: count MJPEG parts from /video.mjpg for 10 s while a run plays (spec §2 criterion 6)."""

import time

import httpx

SECONDS = 10
frames, start = 0, time.time()
with httpx.stream("GET", "http://localhost:8000/video.mjpg", timeout=None) as r:
    for chunk in r.iter_bytes():
        frames += chunk.count(b"--frame")
        if time.time() - start >= SECONDS:
            break
print(f"{frames / SECONDS:.1f} fps from /video.mjpg")
