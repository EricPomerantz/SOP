"""YOLO detector + ByteTrack behind a small protocol, so the model can change without touching consumers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sopvision.config import ConfigError
from sopvision.sources.base import Frame


@dataclass(frozen=True)
class Detection:
    cls: str
    xyxy: tuple[float, float, float, float]  # pixels
    conf: float
    track_id: int | None


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...

    def reset(self) -> None: ...


def _ultralytics_yolo(weights):
    from ultralytics import YOLO

    return YOLO(str(weights))


class YoloDetector:
    def __init__(self, weights: Path, classes: list[str], imgsz: int = 1280, tracker: str = "bytetrack.yaml",
                 model_factory=_ultralytics_yolo):
        self.weights, self.imgsz, self.tracker = Path(weights), imgsz, tracker
        self._factory = model_factory
        self.model = model_factory(self.weights)
        names = list(self.model.names.values())
        if names != list(classes):
            raise ConfigError(f"Model classes {names} don't match SOP detector.classes {list(classes)}")
        self.classes = names

    def detect(self, frame: Frame) -> list[Detection]:
        result = self.model.track(frame.image, persist=True, imgsz=self.imgsz, tracker=self.tracker, verbose=False)[0]
        boxes = result.boxes
        xyxy, cls, conf = boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ids = boxes.id.tolist() if boxes.id is not None else [None] * len(xyxy)
        return [Detection(self.classes[int(c)], tuple(float(v) for v in b), float(s), None if i is None else int(i))
                for b, c, s, i in zip(xyxy, cls, conf, ids)]

    def reset(self) -> None:
        """Forget tracks (new run): reload so ByteTrack starts fresh."""
        self.model = self._factory(self.weights)
