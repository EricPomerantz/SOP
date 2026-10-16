"""How far an SOP has got through the builder's stages, read from the files (nothing extra is stored)."""

from pathlib import Path

import yaml

from sopvision.builder.drafts import SopFile, validate
from sopvision.decide import laya_registry
from sopvision.settings import Settings


def recordings(settings: Settings, data: dict) -> list[dict]:
    """The SOP's recorded runs (timeline format) with their split and whether they have a step timeline."""
    dataset = data.get("dataset") or {}
    root = settings.datasets_dir / (dataset.get("dir") or data.get("sop") or "")
    try:
        splits = yaml.safe_load((root / "splits.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        splits = {}
    split_of = {str(run): split for split, runs in splits.items() for run in runs or []}
    out = []
    for folder in sorted((root / "runs").glob("*")):
        if (folder / "video.mp4").exists():
            out.append({"id": folder.name, "split": split_of.get(folder.name), "steps": (folder / "steps.csv").exists(),
                        "draft": (folder / "steps.draft.csv").exists(), "torque": (folder / "torque.csv").exists()})
    return out


def checklist(settings: Settings, sop_file: SopFile, published: SopFile | None, loaded: bool) -> dict:
    data = sop_file.data
    detector = data.get("detector") or {}
    layout = data.get("layout") or {}
    roles = detector.get("roles") or {}
    runs = recordings(settings, data)
    counts = {s: sum(r["split"] == s for r in runs) for s in ("train", "val", "test")}
    timed = {s: sum(r["split"] == s and r["steps"] for r in runs) for s in ("train", "val", "test")}
    checkpoint = laya_registry.load(settings.laya_registry).get(sop_file.id)
    return {
        "errors": validate(data),
        "detector": bool(detector.get("classes")) and Path(str(detector.get("weights", ""))).exists(),
        "roles": bool(layout.get("anchor")) and any(r == "part" for r in roles.values()),
        "layout": bool(layout.get("slots")),
        "steps": bool(data.get("steps")),
        "recordings": counts,
        "timelines": timed,
        "laya_checkpoint": checkpoint,
        "published": published is not None,
        "loaded": loaded,  # the running app has it (in the top bar)
        "draft": sop_file.draft,
    }
