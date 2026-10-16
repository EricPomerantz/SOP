"""The jobs the app can queue, and the command lines each runs: requests name a kind and its parameters, never commands.

Each builder checks its parameters and returns (title, steps, result paths). Paths are the worker's, which runs in
the app's working directory with the same mounts (docker-compose.yml, service `worker`).
"""

from collections.abc import Callable
from pathlib import Path

from sopvision.config import ConfigError, load_sop
from sopvision.jobs.store import Step
from sopvision.settings import Settings

PY = "python"


class UnknownJobKind(ValueError):
    pass


def _sop_file(settings: Settings, params: dict, check: bool = True) -> Path:
    """The SOP file a job is for: `<name>.yaml` (published) or `drafts/<name>.yaml`, under the SOPs folder."""
    name = str(params.get("sop_file") or "").removeprefix(f"{settings.sops_dir}/")  # the builder's `file` works too
    base = Path(name).name
    path = settings.sops_dir / ("drafts" if name.startswith("drafts/") else "") / base
    if not base.endswith(".yaml") or name not in (base, f"drafts/{base}") or not path.is_file():
        raise ConfigError(f"No SOP file '{name}' in {settings.sops_dir} or its drafts")
    if check:
        load_sop(path)  # an invalid SOP fails here, not half way through the job
    return path


def draft_timelines(settings: Settings, params: dict) -> tuple[str, list[Step], dict]:
    """The tracker's step timeline and a browser preview for each recording (sopvision.builder.timelines)."""
    path = _sop_file(settings, params)
    sop = load_sop(path)
    argv = [PY, "-m", "sopvision.builder.timelines", "--sop", str(path)]
    for run in params.get("runs") or []:
        argv += ["--run", str(run)]
    return f"Draft step timelines for {sop.display_name}", [Step("draft timelines", argv)], {}


def previews(settings: Settings, params: dict) -> tuple[str, list[Step], dict]:
    """Browser-playable copies of recordings, for the timeline page (no detector, the SOP may be incomplete)."""
    path = _sop_file(settings, params, check=False)
    argv = [PY, "-m", "sopvision.builder.timelines", "--sop", str(path), "--previews-only"]
    for run in params.get("runs") or []:
        argv += ["--run", str(run)]
    runs = params.get("runs") or []
    return f"Prepare video{'s' if len(runs) != 1 else ''} for labelling" + (f" ({', '.join(map(str, runs))})" if runs else ""), \
        [Step("prepare video", argv)], {}


def autofit(settings: Settings, params: dict) -> tuple[str, list[Step], dict]:
    """Slot positions suggested from the recordings (sopvision.builder.autofit)."""
    import yaml

    path = _sop_file(settings, params, check=False)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    argv = [PY, "-m", "sopvision.builder.autofit", "--sop", str(path)]
    threshold = params.get("threshold")
    if threshold is not None:
        try:
            argv += ["--threshold", str(min(max(float(threshold), 0.01), 0.5))]
        except (TypeError, ValueError):
            raise ConfigError(f"threshold must be a number, not {threshold!r}") from None
    sop_id = data.get("sop") or path.stem
    return f"Suggest slots for {data.get('name') or sop_id}", [Step("auto-fit slots", argv)], \
        {"autofit": f"data/builder/{sop_id}/autofit.json"}


def train_laya(settings: Settings, params: dict) -> tuple[str, list[Step], dict]:
    """The SOP's own Laya checkpoint, trained, benchmarked and replayed (sopvision.builder.training)."""
    from sopvision.builder import training

    path = _sop_file(settings, params)
    sop = load_sop(path)
    version = training.next_version(sop.sop)
    try:
        epochs = min(max(int(params.get("epochs") or 12), 1), 40)
    except (TypeError, ValueError):
        raise ConfigError(f"epochs must be a whole number, not {params.get('epochs')!r}") from None
    steps = []
    for stage in training.STAGES:
        argv = [PY, "-m", "sopvision.builder.training", stage, "--sop", str(path), "--version", str(version)]
        steps.append(Step(stage, argv + (["--epochs", str(epochs)] if stage == "train" else [])))
    p = training.paths(sop.sop, version)
    return f"Train the decision model for {sop.display_name} ({p.name})", steps, \
        {"checkpoint": p.model.name, "report": str(p.eval / "report.json")}


KINDS: dict[str, Callable[[Settings, dict], tuple[str, list[Step], dict]]] = {
    "draft_timelines": draft_timelines,
    "autofit": autofit,
    "previews": previews,
    "train_laya": train_laya,
}


def build(kind: str, settings: Settings, params: dict) -> tuple[str, list[Step], dict]:
    if kind not in KINDS:
        raise UnknownJobKind(f"Unknown job kind '{kind}'; one of {', '.join(KINDS)}")
    return KINDS[kind](settings, params)
