import asyncio
import threading

import cv2
import numpy as np
import pytest

from sopvision.server.frames import FrameBuffer, encode_jpeg


@pytest.mark.parametrize("w,h,ew,eh", [(1920, 1080, 1280, 720), (640, 480, 640, 480), (1280, 960, 960, 720)])
def test_encode_fits_without_upscaling(w, h, ew, eh):  # Review Focus 5
    out = cv2.imdecode(np.frombuffer(encode_jpeg(np.zeros((h, w, 3), np.uint8)), np.uint8), 1)
    assert out.shape[:2] == (eh, ew)


async def test_frame_buffer_hands_newer_frames_from_another_thread():
    buf = FrameBuffer()
    threading.Thread(target=lambda: buf.put(b"one")).start()
    seq, data = await asyncio.wait_for(buf.next(0), 1.0)
    assert (seq, data) == (1, b"one")
    waiter = asyncio.create_task(buf.next(seq))
    await asyncio.sleep(0.05)
    assert not waiter.done()
    threading.Thread(target=lambda: buf.put(b"two")).start()
    assert await asyncio.wait_for(waiter, 1.0) == (2, b"two")


async def test_waiters_wake_as_soon_as_a_frame_arrives():
    # Polling with asyncio.sleep can't do this on Windows: sleeps round up to ~15 ms there.
    import time

    buf = FrameBuffer()
    latencies = []
    for i in range(10):
        put_at = {}

        def later():
            time.sleep(0.02)
            put_at["t"] = time.perf_counter()
            buf.put(b"frame")

        threading.Thread(target=later).start()
        await asyncio.wait_for(buf.next(i), 1.0)
        latencies.append(time.perf_counter() - put_at["t"])
    assert max(latencies) < 0.008, [round(x * 1000, 1) for x in latencies]
