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


STAGES = [("detector", "Detector"), ("recordings", "Recordings"), ("layout", "Slots"), ("steps", "Steps"),
          ("timelines", "Timelines"), ("train", "Decision model")]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def stages(data: dict, errors: list[str], runs: list[dict], checkpoint: str | None) -> list[dict]:
    """What each builder stage still needs, as sentences for the user; `done` when nothing is left."""
    detector = data.get("detector") or {}
    layout = data.get("layout") or {}
    roles = detector.get("roles") or {}
    counts = {s: sum(r["split"] == s for r in runs) for s in ("train", "val", "test")}
    untimed = sum(not r["steps"] for r in runs)
    blockers: dict[str, list[str]] = {s: [] for s, _ in STAGES}
    if not (detector.get("classes") and Path(str(detector.get("weights", ""))).exists()):
        blockers["detector"].append("Import a detector model")
    if not layout.get("anchor"):
        blockers["detector"].append("Pick the anchor class")
    if not any(r == "part" for r in roles.values()):
        blockers["detector"].append("Mark at least one class as a part")
    blockers["detector"] += [e for e in errors if e.startswith(("detector", "layout: anchor", "layout: tool"))
                             and not (e.startswith("layout: anchor") and not layout.get("anchor"))]
    blockers["recordings"] += [f"Record at least one {s} run" for s, n in counts.items() if n == 0]
    if not layout.get("slots"):
        blockers["layout"].append("Add at least one slot")
    blockers["layout"] += [e for e in errors if e.startswith(("slot", "layout.slots"))]
    if not data.get("steps"):
        blockers["steps"].append("Add at least one step")
    blockers["steps"] += [e for e in errors if e.startswith(("step", "duplicate step", "groups", "order"))]
    if not runs:
        blockers["timelines"].append("Record some runs first")
    elif untimed:
        blockers["timelines"].append(f"{_plural(untimed, 'run')} still need{'s' if untimed == 1 else ''} a timeline")
    if not checkpoint:
        blockers["train"].append("Train a decision model")
    return [{"id": s, "title": t, "done": not blockers[s], "blockers": blockers[s]} for s, t in STAGES]


def publish_check(data: dict, errors: list[str]) -> dict:
    """The same checks `drafts.publish` makes, as blockers the builder can show before the user presses Publish."""
    blockers = list(errors)
    if not Path(str((data.get("detector") or {}).get("weights", ""))).exists():
        blockers.append("Import the detector model first")
    if not data.get("steps"):
        blockers.append("Add at least one step")
    return {"ready": not blockers, "blockers": blockers}


def checklist(settings: Settings, sop_file: SopFile, published: SopFile | None, loaded: bool) -> dict:
    data = sop_file.data
    detector = data.get("detector") or {}
    layout = data.get("layout") or {}
    roles = detector.get("roles") or {}
    runs = recordings(settings, data)
    counts = {s: sum(r["split"] == s for r in runs) for s in ("train", "val", "test")}
    timed = {s: sum(r["split"] == s and r["steps"] for r in runs) for s in ("train", "val", "test")}
    checkpoint = laya_registry.load(settings.laya_registry).get(sop_file.id)
    errors = validate(data)
    return {
        "errors": errors,
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
        "stages": stages(data, errors, runs, checkpoint),
        "publish": publish_check(data, errors),
    }
