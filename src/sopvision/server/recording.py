"""Recording a camera into an SOP's dataset (timeline format), with torque presses, for training and evaluation.

No detector is involved: recordings are what a new SOP's detector is trained from. While recording, the dashboard's
video stream shows a preview, and every "Torque reached" press is stored on the video's timeline (sources/recorder.py).
On stop: <dataset>/runs/<id>/video.mp4, torque.csv, meta.json, and the run listed under its split in splits.yaml.
"""

import json
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import yaml

from sopvision.config import ConfigError, SopConfig
from sopvision.recordings import dataset_root
from sopvision.server.frames import FrameBuffer, encode_jpeg
from sopvision.settings import Settings
from sopvision.sources.latency import camera_delay
from sopvision.sources.recorder import Recorder
from sopvision.sources.registry import SourceSpec, open_source

PREVIEW_WIDTH = 960
PREVIEW_INTERVAL = 0.1  # seconds between preview frames
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def add_to_split(splits_file: Path, split: str, run_id: str) -> None:
    splits = yaml.safe_load(splits_file.read_text(encoding="utf-8")) if splits_file.exists() else {}
    splits = splits or {}
    runs = list(splits.get(split) or [])
    if run_id not in runs:
        runs.append(run_id)
    splits[split] = runs
    splits_file.write_text(yaml.safe_dump({k: splits.get(k, []) for k in ("train", "val", "test")} |
                                          {k: v for k, v in splits.items() if k not in ("train", "val", "test")},
                                          sort_keys=False), encoding="utf-8")


class RecordingSession:
    def __init__(self, settings: Settings, sop: SopConfig, sop_file: str, spec: SourceSpec, split: str, run_id: str,
                 frames: FrameBuffer, fps: float = 25.0, source=None):
        if sop.dataset.format != "timeline":
            raise ConfigError(f"SOP {sop.sop} keeps its recordings in the '{sop.dataset.format}' format; recording "
                              "writes the timeline format only")
        if split not in ("train", "val", "test"):
            raise ConfigError(f"split must be train, val or test, not '{split}'")
        if not _RUN_ID.match(run_id):
            raise ConfigError("run name: letters, digits, '.', '_' and '-' only")
        self.root = dataset_root(sop, settings)
        self.folder = self.root / "runs" / run_id
        if self.folder.exists():
            raise ConfigError(f"{self.folder} already exists; pick another name")
        self.sop, self.sop_file, self.spec, self.split, self.run_id = sop, sop_file, spec, split, run_id
        self.frames = frames
        self.delay_s = camera_delay(spec, settings)
        self.recorder = Recorder(self.folder / "video.mp4", fps=fps, delay_s=self.delay_s)
        self.source = source or open_source(spec, settings)
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._reader: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.recorder.start_thread()
        self._reader = threading.Thread(target=self._read, name="recording-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        last_preview = 0.0
        try:
            for frame in self.source.frames():
                self.recorder.offer(frame.image)
                now = time.monotonic()
                if now - last_preview >= PREVIEW_INTERVAL:
                    last_preview = now
                    h, w = frame.image.shape[:2]
                    small = cv2.resize(frame.image, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
                    self.frames.put(encode_jpeg(small))
        except Exception as e:  # noqa: BLE001 — reported in status and on stop
            self.error = f"{type(e).__name__}: {e}"

    def torque(self, ok: bool, torque_nm: float | None = None) -> float | None:
        press = self.recorder.press(ok, torque_nm)
        return None if press is None else press.time_s

    def status(self) -> dict:
        return {"run_id": self.run_id, "sop": self.sop.sop, "split": self.split, "source": self.spec.id,
                "folder": str(self.folder), "seconds": round(self.recorder.video_time() or 0.0, 1),
                "frames": self.recorder.written, "camera_delay_ms": round(self.delay_s * 1000, 1),
                "presses": [{"time_s": p.time_s, "ok": p.ok} for p in self.recorder.presses], "error": self.error}

    def _halt(self) -> dict:
        self.source.close()
        if self._reader is not None:
            self._reader.join(timeout=5)
        return self.recorder.stop()

    def discard(self) -> dict:
        """Stop and throw the run away: its folder is deleted and splits.yaml is left as it was."""
        summary = self._halt()
        if self.folder.exists() and self.folder.parent == self.root / "runs":
            shutil.rmtree(self.folder)
        return {"run_id": self.run_id, "discarded": True, "seconds": summary["seconds"]}

    def stop(self) -> dict:
        summary = self._halt()
        if summary["frames"] == 0:
            raise ConfigError(f"No frames came from {self.spec.label}; nothing was recorded"
                              + (f" ({self.error})" if self.error else ""))
        self.recorder.write_torque_csv(self.folder / "torque.csv")
        meta = {"run_id": self.run_id, "sop": self.sop.sop, "sop_file": self.sop_file, "split": self.split,
                "source": self.spec.id, "started_at": self.started_at, "fps": self.recorder.fps,
                "camera_delay_ms": round(self.delay_s * 1000, 1), **summary,
                "note": "steps.csv (step_id,start_s,end_s) is added by hand; torque.csv times are on the video's timeline"}
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        add_to_split(self.root / "splits.yaml", self.split, self.run_id)
        return meta | {"folder": str(self.folder)}
