"""Start the local Decider: make sure the pinned snapshot is in the HF cache, then serve it.

First start: downloads DECIDER_REPO at DECIDER_REVISION (8.4 GB for decider-4b) into HF_HOME, the hf_cache volume.
Later starts: the snapshot is found locally and nothing is fetched. The revision is a commit SHA, so the weights
(and the `decider/` inference code shipped next to them, which is what gets served) cannot change underneath us.

    python docker/decider_serve.py                 serve on DECIDER_PORT (8001)
    python docker/decider_serve.py --fetch-only    just make sure the snapshot is cached (also used for the vision model)
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULTS = {
    "Mapika/decider-4b": "eb5fbdfc9448473ec25e399882912863afbdb70e",  # v2.1
    "Mapika/decider-2b-vision": "863e290863655f1d6b69324d77d09ac972d21609",
}


def ensure_snapshot(repo: str, revision: str) -> Path:
    try:
        path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        if (path / "model.safetensors").exists():
            print(f"[decider] {repo}@{revision[:8]} found in cache: {path}", flush=True)
            return path
    except Exception:  # noqa: BLE001 — not cached (or only partly): download, which resumes partial files
        pass
    print(f"[decider] downloading {repo}@{revision[:8]} into {os.environ.get('HF_HOME')} (first start only)", flush=True)
    return Path(snapshot_download(repo, revision=revision))


def main() -> None:
    repo = os.environ.get("DECIDER_REPO", "Mapika/decider-4b")
    revision = os.environ.get("DECIDER_REVISION") or DEFAULTS[repo]
    path = ensure_snapshot(repo, revision)
    if "--fetch-only" in sys.argv:
        return
    os.environ["DECIDER_MODEL"] = str(path)
    sys.path.insert(0, str(path))  # the snapshot's own decider/ package, matched to these weights
    import uvicorn

    uvicorn.run("decider.serve:app", host="0.0.0.0", port=int(os.environ.get("DECIDER_PORT", "8001")), log_level="info")


if __name__ == "__main__":
    main()
