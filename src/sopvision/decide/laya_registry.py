"""Which Laya checkpoint answers for which SOP: the registry the Laya service and the app both read.

    data/training/models/registry.json  {"sops": {"<sop id>": "<checkpoint folder under data/training/models>"}}

The Laya service (docker/laya_serve.py) registers each named checkpoint as `ckpt-<folder>`; the app sends that name
as the request's `model` for the SOP's runs. An SOP the registry doesn't name gets `settings.laya_model` (the shared
checkpoint the service was started with). Activating a checkpoint rewrites the file and asks the service to reload.
"""

import json
import os
from pathlib import Path

from sopvision.settings import Settings


def model_name(checkpoint: str) -> str:
    return f"ckpt-{Path(checkpoint).name}".lower()


def load(path: Path) -> dict[str, str]:
    """SOP id -> checkpoint folder; empty when there's no registry yet."""
    try:
        return dict(json.loads(Path(path).read_text(encoding="utf-8")).get("sops", {}))
    except FileNotFoundError:
        return {}


def laya_model_for(settings: Settings, sop_id: str | None) -> str:
    checkpoint = load(settings.laya_registry).get(sop_id) if sop_id else None
    return model_name(checkpoint) if checkpoint else settings.laya_model


def set_checkpoint(path: Path, sop_id: str, checkpoint: str | None) -> dict[str, str]:
    """Point an SOP at a checkpoint folder (None: back to the shared checkpoint). Written atomically."""
    sops = load(path)
    if checkpoint is None:
        sops.pop(sop_id, None)
    else:
        sops[sop_id] = Path(checkpoint).name
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"sops": dict(sorted(sops.items()))}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return sops
