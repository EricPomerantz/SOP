"""SOP configuration: one YAML file per SOP, validated at load time (spec §5)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(Exception):
    """A configuration or environment problem, with a message naming what to fix."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlotSpec(_Model):
    at: tuple[float, float]
    # The part class that fills the slot, or a list of classes any of which does (e.g. a bolt in any colour).
    filled_by: str | list[str]
    empty_as: str | None = None
    match_radius: float | None = None

    @property
    def fills(self) -> list[str]:
        return [self.filled_by] if isinstance(self.filled_by, str) else list(self.filled_by)


class Layout(_Model):
    anchor: str
    match_radius: float
    slots: dict[str, SlotSpec]
    # Detector classes that are tools (e.g. a torque wrench): located like hands (the slot they're at), never parts.
    tools: list[str] = []
    # How far each new anchor detection moves the reference box (0-1). Low smooths detection jitter but lags when the
    # camera moves; 1 follows every frame. A camera that shakes (mounted on the work frame) needs a high value.
    anchor_follow: float = Field(0.1, gt=0.0, le=1.0)
    # Seconds an installed slot must read empty before the scene says its part left (None: the usual 0.5 s settle).
    # Longer rides out a detector that briefly mistakes a part for its empty slot, at the cost of later removal alerts.
    removal_seconds: float | None = Field(None, gt=0.0)


class StepSpec(_Model):
    id: str
    text: str
    label: str | None = None
    # installed / empty: what the camera sees in the slot. torqued: a torque-OK result attributed to the slot.
    evidence: dict[str, Literal["installed", "empty", "torqued"]] = {}
    # The step's last action isn't visible (e.g. a latch): evidence only counts once hands have left the part.
    hands_off: bool = False


class DetectorSpec(_Model):
    weights: Path
    classes: list[str]
    autolabel_prompts: dict[str, str] = {}


class DatasetSpec(_Model):
    """Where this SOP's recordings are and how their step timelines are written (sopvision.recordings)."""

    # timeline: <dir>/runs/<id>/{video.mp4, steps.csv, torque.csv}, <dir>/splits.yaml
    # clips:    the server_fan layout: raw/Install_<n>.MP4 cut into per-action clips under train/ and test/
    format: Literal["timeline", "clips"] = "timeline"
    dir: str | None = None  # under DATASETS_DIR (timeline; default: the SOP id). clips read SOP_DATA_ROOT.
    actions: dict[int, str] = {}  # clips: action number in the clip names -> step id (or off_sop)


class DecisionSpec(_Model):
    accept_confidence: float = 0.8
    hold_seconds: float = 1.0
    evidence_only_after: float = 3.0
    hands_off_seconds: float = 0.5


class SequenceGroup(_Model):
    """Steps in a group that must happen in exactly this order (a plain list in `groups` allows any order)."""

    sequence: list[str]


class SopConfig(_Model):
    sop: str
    name: str | None = None  # display name for the dashboard; defaults to the id
    version: int
    description: str
    detector: DetectorSpec
    layout: Layout
    steps: list[StepSpec]
    order: Literal["strict", "any", "groups"] = "strict"
    groups: list[list[str] | SequenceGroup] | None = None
    off_sop: str
    idle: str
    decisions: DecisionSpec = DecisionSpec()
    dataset: DatasetSpec = DatasetSpec()

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "SopConfig":
        classes = set(self.detector.classes)
        if self.layout.anchor not in classes:
            raise ValueError(f"layout: anchor '{self.layout.anchor}' is not a detector class")
        for name, slot in self.layout.slots.items():
            if not slot.fills:
                raise ValueError(f"slot {name}: filled_by names no class")
            for field, values in (("filled_by", slot.fills), ("empty_as", [slot.empty_as])):
                for value in values:
                    if value is not None and value not in classes:
                        raise ValueError(f"slot {name}: {field} '{value}' is not a detector class")
            if not all(0.0 <= c <= 1.0 for c in slot.at):
                raise ValueError(f"slot {name}: coordinates must be within [0, 1]")
        part_classes = {c for slot in self.layout.slots.values() for c in [*slot.fills, slot.empty_as] if c}
        for tool in self.layout.tools:
            if tool not in classes:
                raise ValueError(f"layout: tool '{tool}' is not a detector class")
            if tool in part_classes or tool in (self.layout.anchor, "hand"):
                raise ValueError(f"layout: tool '{tool}' is also a part, the anchor or the hand class")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id '{step.id}'")
            seen.add(step.id)
            for slot in step.evidence:
                if slot not in self.layout.slots:
                    raise ValueError(f"step {step.id}: evidence slot '{slot}' is not declared")
        for action, step_id in self.dataset.actions.items():
            if step_id not in seen and step_id != "off_sop":
                raise ValueError(f"dataset: action {action} maps to unknown step '{step_id}'")
        if self.order == "groups":
            flat = [s for group in self.groups or [] for s in _group_steps(group)]
            if sorted(flat) != sorted(seen):
                raise ValueError("groups must contain every step exactly once")
        return self

    @property
    def display_name(self) -> str:
        return self.name or self.sop

    def step(self, step_id: str) -> StepSpec:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def step_label(self, step_id: str) -> str:
        return self.step(step_id).label or step_id

    def slot_radius(self, slot: str) -> float:
        r = self.layout.slots[slot].match_radius
        return self.layout.match_radius if r is None else r

    def must_precede(self, step_id: str) -> set[str]:
        """Steps that must be done before `step_id` is completed, per the order mode."""
        if self.order == "any":
            return set()
        if self.order == "groups":
            earlier: set[str] = set()
            for group in self.groups or []:
                steps = _group_steps(group)
                if step_id in steps:
                    # A sequence group also needs the steps before this one in the sequence.
                    return earlier | (set(steps[: steps.index(step_id)]) if isinstance(group, SequenceGroup) else set())
                earlier |= set(steps)
            raise KeyError(step_id)
        ids = [s.id for s in self.steps]
        return set(ids[: ids.index(step_id)])


def _group_steps(group: "list[str] | SequenceGroup") -> list[str]:
    return group.sequence if isinstance(group, SequenceGroup) else list(group)


def _describe(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        msg = err["msg"].removeprefix("Value error, ")
        loc = ".".join(str(p) for p in err["loc"])
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)


def load_sop(path: Path) -> SopConfig:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path}: SOP file not found") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: invalid YAML: {e}") from None
    try:
        return SopConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path.name}: {_describe(e)}") from None


def list_sops(directory: Path) -> list[SopConfig]:
    return [load_sop(p) for p in sorted(Path(directory).glob("*.yaml"))]
