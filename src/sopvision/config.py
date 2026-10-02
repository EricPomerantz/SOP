"""SOP configuration: one YAML file per SOP, validated at load time (spec §5)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


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


class DetectorSpec(_Model):
    weights: Path
    classes: list[str]
    autolabel_prompts: dict[str, str] = {}


class DecisionSpec(_Model):
    accept_confidence: float = 0.8
    hold_seconds: float = 1.0
    evidence_only_after: float = 3.0


class SopConfig(_Model):
    sop: str
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
