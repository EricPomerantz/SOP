"""SOP drafts: the YAML the builder edits, kept under sops/drafts/ until it's published into sops/.

A draft may be incomplete (no detector yet, no steps): it is saved as it is and its validation errors are reported,
so the builder can be filled in stage by stage. Publishing needs a valid SOP whose weights exist. Editing a published
SOP copies it into drafts/ under the same file name; publishing it back bumps the version and keeps the previous file
in sops/.history/. Saving through the builder rewrites the YAML, so hand-written comments don't survive it.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from sopvision.config import SopConfig, _describe
from sopvision.settings import Settings

SOP_ID = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
DEFAULT_OFF_SOP = "Any action that is not one of the steps above."


class DraftError(ValueError):
    """A builder request that can't be done as asked; the message says why."""


@dataclass(frozen=True)
class SopFile:
    id: str
    path: Path
    draft: bool
    data: dict


def _read(path: Path) -> dict | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def scan(settings: Settings) -> list[SopFile]:
    """Every SOP file, published and draft, valid or not (unreadable YAML is skipped)."""
    out = []
    for draft, folder in ((False, settings.sops_dir), (True, settings.sops_dir / "drafts")):
        for path in sorted(folder.glob("*.yaml")):
            data = _read(path)
            if data is not None:
                out.append(SopFile(str(data.get("sop") or path.stem), path, draft, data))
    return out


def find(settings: Settings, sop_id: str) -> tuple[SopFile | None, SopFile | None]:
    """(draft, published) for an SOP id."""
    files = [f for f in scan(settings) if f.id == sop_id]
    return next((f for f in files if f.draft), None), next((f for f in files if not f.draft), None)


def current(settings: Settings, sop_id: str) -> SopFile:
    """What the builder shows for an SOP: its draft if it has one, else the published file."""
    draft, published = find(settings, sop_id)
    if draft is None and published is None:
        raise DraftError(f"No SOP {sop_id}")
    return draft or published


def validate(data: dict) -> list[str]:
    try:
        SopConfig.model_validate(data)
    except ValidationError as e:
        return _describe(e).split("; ")
    return []


def recordings_dir_of(settings: Settings, sop_id: str) -> str:
    """The dataset folder (under datasets_dir) an SOP's recordings live in, for a new SOP that reuses them."""
    sf = current(settings, sop_id)
    folder = (sf.data.get("dataset") or {}).get("dir") or sop_id
    if not (settings.datasets_dir / folder / "runs").is_dir():
        raise DraftError(f"{sop_id} has no recordings to reuse (datasets/{folder}/runs)")
    return folder


def new_draft(settings: Settings, sop_id: str, name: str, description: str, dataset_dir: str | None = None) -> SopFile:
    if not SOP_ID.match(sop_id):
        raise DraftError("SOP id: lower-case letters, digits and '_', starting with a letter (2-48 characters)")
    draft, published = find(settings, sop_id)
    if draft or published:
        raise DraftError(f"An SOP {sop_id} already exists")
    path = settings.sops_dir / "drafts" / f"{sop_id}.yaml"
    if path.exists():
        raise DraftError(f"{path} already exists")
    data = {
        "sop": sop_id, "name": name or sop_id, "version": 1, "description": description or name or sop_id,
        "detector": {"weights": f"models/{sop_id}/best.pt", "classes": []},
        "layout": {"anchor": "", "match_radius": 0.1, "slots": {}},
        "steps": [], "order": "strict", "off_sop": DEFAULT_OFF_SOP, "idle": "No one is working.",
        "decisions": {"accept_confidence": 0.8, "hold_seconds": 1.0, "evidence_only_after": 1.0},
        "dataset": {"format": "timeline", "dir": dataset_dir or sop_id},
    }
    return write(SopFile(sop_id, path, True, data), data)


def write(sop_file: SopFile, data: dict) -> SopFile:
    if not sop_file.draft:
        raise DraftError(f"{sop_file.id} is published: start editing it first (a draft copy)")
    if data.get("sop") != sop_file.id:
        raise DraftError(f"The SOP id can't change (was {sop_file.id})")
    sop_file.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sop_file.path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    tmp.replace(sop_file.path)
    return SopFile(sop_file.id, sop_file.path, True, data)


def start_edit(settings: Settings, sop_id: str) -> SopFile:
    """A draft copy of a published SOP (or the draft already being edited)."""
    draft, published = find(settings, sop_id)
    if draft is not None:
        return draft
    if published is None:
        raise DraftError(f"No SOP {sop_id}")
    path = settings.sops_dir / "drafts" / published.path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(published.path, path)
    return SopFile(sop_id, path, True, dict(published.data))


def discard(settings: Settings, sop_id: str) -> None:
    draft, _ = find(settings, sop_id)
    if draft is None:
        raise DraftError(f"{sop_id} has no draft")
    draft.path.unlink()


def publish(settings: Settings, sop_id: str) -> Path:
    """Move the draft into sops/ (checked first). Returns the published file."""
    draft, published = find(settings, sop_id)
    if draft is None:
        raise DraftError(f"{sop_id} has no draft to publish")
    errors = validate(draft.data)
    if errors:
        raise DraftError("Fix these first: " + "; ".join(errors))
    data = dict(draft.data)
    weights = Path(data["detector"]["weights"])
    if not weights.exists():
        raise DraftError(f"Detector weights {weights} not found: import the detector first")
    if not data.get("steps"):
        raise DraftError("Add at least one step first")
    target = published.path if published else settings.sops_dir / draft.path.name
    if published is not None:
        old = int(published.data.get("version") or 1)
        data["version"] = max(int(data.get("version") or 1), old + 1)
        history = settings.sops_dir / ".history"
        history.mkdir(exist_ok=True)
        shutil.copyfile(published.path, history / f"{published.path.stem}.v{old}.yaml")
    elif target.exists():
        raise DraftError(f"{target} exists and belongs to another SOP")
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    tmp.replace(target)
    draft.path.unlink()
    return target


NUMBERED_SLOT = re.compile(r"^[a-z]+_(\d+)$")


def generate_steps(data: dict) -> dict:
    """Steps from the layout: a place step per slot (any order), then, with a tool, a torque step per slot in slot
    order as a sequence. Returns the `steps`, `order` and `groups` fields; the builder shows them for editing.

    Slots numbered `<word>_<n>` (hole_1, hole_2, ...) give short ids and labels (`place_1`, "Place 1"), in number
    order, so the ids match timelines labelled by slot number; any other naming keeps the whole slot name."""
    slots = list((data.get("layout") or {}).get("slots") or {})
    tool = bool((data.get("layout") or {}).get("tools"))
    numbers = [m.group(1) for s in slots if (m := NUMBERED_SLOT.match(s))]
    if len(numbers) == len(slots) and len(set(numbers)) == len(slots):
        short = {s: n for s, n in zip(slots, numbers)}
        slots = sorted(slots, key=lambda s: int(short[s]))
    else:
        short = {s: s for s in slots}
    places = [{"id": f"place_{short[s]}", "label": f"Place {_words(short[s])}", "text": f"Install the part in {_words(s)}.",
               "evidence": {s: "installed"}} for s in slots]
    if not tool:
        return {"steps": places, "order": "strict", "groups": None}
    torques = [{"id": f"torque_{short[s]}", "label": f"Torque {_words(short[s])}",
                "text": f"Torque the part in {_words(s)} with the {_words(data['layout']['tools'][0])}.",
                "evidence": {s: "torqued"}} for s in slots]
    return {"steps": places + torques, "order": "groups",
            "groups": [[p["id"] for p in places], {"sequence": [t["id"] for t in torques]}]}


def _words(name: str) -> str:
    return name.replace("_", " ")
