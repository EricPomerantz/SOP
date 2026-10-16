"""Importing an SOP's detector: a trained Ultralytics detection model, uploaded or picked from models/.

Each import is a new file (models/<sop id>/detector-<time>.pt), so replacing a published SOP's detector in its draft
never changes what live runs use until the draft is published.
"""

import shutil
import time
from pathlib import Path

from sopvision.builder.drafts import DraftError, SopFile, write

MODELS = Path("models")


def _ultralytics_names(weights: Path) -> tuple[str, list[str]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.task, list(model.names.values())


def read_classes(weights: Path, loader=None) -> list[str]:
    loader = loader or _ultralytics_names
    try:
        task, names = loader(Path(weights))
    except Exception as e:  # noqa: BLE001 — anything Ultralytics can't load is "not a model"
        raise DraftError(f"Not an Ultralytics model ({type(e).__name__}: {e})") from None
    if task != "detect":
        raise DraftError(f"This is a {task} model; the SOP needs a detection model (boxes)")
    if not names:
        raise DraftError("The model has no class names")
    return names


def existing_models(loader=None) -> list[dict]:
    """Detection models already under models/, with their classes, for picking instead of uploading."""
    out = []
    for path in sorted(MODELS.glob("*/*.pt")):
        try:
            out.append({"path": str(path), "classes": read_classes(path, loader)})
        except DraftError:
            continue
    return out


def apply_detector(sop_file: SopFile, weights: Path, classes: list[str]) -> tuple[SopFile, list[str]]:
    """Point the draft at `weights`: classes from the model; anchor, tools and roles kept where the classes still exist.
    Returns the draft and warnings."""
    data = dict(sop_file.data)
    detector = dict(data.get("detector") or {})
    detector["weights"], detector["classes"] = str(weights), list(classes)
    detector["roles"] = {c: r for c, r in (detector.get("roles") or {}).items() if c in classes}
    data["detector"] = detector
    layout = dict(data.get("layout") or {})
    if layout.get("anchor") not in classes:
        layout["anchor"] = ""
    layout["tools"] = [t for t in layout.get("tools") or [] if t in classes]
    data["layout"] = layout
    warnings = []
    if "hand" not in classes:
        warnings.append("The model has no 'hand' class: the app won't see hands (occlusion, who's working).")
    return write(sop_file, data), warnings


def import_upload(sop_file: SopFile, upload: Path, loader=None) -> tuple[SopFile, list[str]]:
    """Check an uploaded file, keep it as the SOP's next detector, and point the draft at it."""
    classes = read_classes(upload, loader)
    folder = MODELS / sop_file.id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"detector-{time.strftime('%Y%m%d-%H%M%S')}.pt"
    shutil.move(str(upload), target)
    return apply_detector(sop_file, target, classes)


def import_existing(sop_file: SopFile, path: str, loader=None) -> tuple[SopFile, list[str]]:
    weights = Path(path)
    if weights.parent.parent != MODELS or weights.suffix != ".pt" or not weights.is_file():
        raise DraftError(f"{path}: pick a model from models/<folder>/<name>.pt")
    return apply_detector(sop_file, weights, read_classes(weights, loader))
