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
    filled_by: str
    empty_as: str | None = None
    match_radius: float | None = None


class Layout(_Model):
    anchor: str
    match_radius: float
    slots: dict[str, SlotSpec]


class StepSpec(_Model):
    id: str
    text: str
    label: str | None = None
    evidence: dict[str, Literal["installed", "empty"]] = {}
    # The step's last action isn't visible (e.g. a latch): evidence only counts once hands have left the part.
    hands_off: bool = False


class DetectorSpec(_Model):
    weights: Path
    classes: list[str]
    autolabel_prompts: dict[str, str] = {}


class DecisionSpec(_Model):
    accept_confidence: float = 0.8
    # Per-answer thresholds, each defaulting to accept_confidence. Decision models' scales differ (Kev's step
    # confidence runs far below Jev's), so a model's own values usually come from Settings, not the SOP file.
    step_confidence: float | None = Field(default=None, ge=0, le=1)     # current step: activate it, or warn off-SOP
    done_confidence: float | None = Field(default=None, ge=0, le=1)     # "active step done" (still needs evidence)
    anomaly_confidence: float | None = Field(default=None, ge=0, le=1)  # anomaly warning
    hold_seconds: float = 1.0
    evidence_only_after: float = 3.0
    hands_off_seconds: float = 0.5

    @property
    def step_threshold(self) -> float:
        return self.accept_confidence if self.step_confidence is None else self.step_confidence

    @property
    def done_threshold(self) -> float:
        return self.accept_confidence if self.done_confidence is None else self.done_confidence

    @property
    def anomaly_threshold(self) -> float:
        return self.accept_confidence if self.anomaly_confidence is None else self.anomaly_confidence


class SopConfig(_Model):
    sop: str
    name: str | None = None  # display name for the dashboard; defaults to the id
    version: int
    description: str
    detector: DetectorSpec
    layout: Layout
    steps: list[StepSpec]
    order: Literal["strict", "any", "groups"] = "strict"
    groups: list[list[str]] | None = None
    off_sop: str
    idle: str
    decisions: DecisionSpec = DecisionSpec()

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "SopConfig":
        classes = set(self.detector.classes)
        if self.layout.anchor not in classes:
            raise ValueError(f"layout: anchor '{self.layout.anchor}' is not a detector class")
        for name, slot in self.layout.slots.items():
            for field in ("filled_by", "empty_as"):
                value = getattr(slot, field)
                if value is not None and value not in classes:
                    raise ValueError(f"slot {name}: {field} '{value}' is not a detector class")
            if not all(0.0 <= c <= 1.0 for c in slot.at):
                raise ValueError(f"slot {name}: coordinates must be within [0, 1]")
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id '{step.id}'")
            seen.add(step.id)
            for slot in step.evidence:
                if slot not in self.layout.slots:
                    raise ValueError(f"step {step.id}: evidence slot '{slot}' is not declared")
        if self.order == "groups":
            flat = [s for group in self.groups or [] for s in group]
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
                if step_id in group:
                    return earlier
                earlier |= set(group)
            raise KeyError(step_id)
        ids = [s.id for s in self.steps]
        return set(ids[: ids.index(step_id)])


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
