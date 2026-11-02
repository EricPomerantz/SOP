"""Demo assets (detectors, recordings, the fine-tuned Laya) kept in a private Hugging Face repo, so a fresh clone runs.

Git doesn't carry them (models/, data/, datasets/ and the recordings are gitignored and large). The HF repo holds them at
the same paths they have here, so fetching into the repo root puts everything where the app looks.

    docker compose run --rm demo-assets fetch          # what the app needs to start + the lug-bolt test runs (~3.4 GB)
    docker compose run --rm demo-assets fetch --all    # + every lug-bolt run (~6.7 GB)
    docker compose run --rm demo-assets publish        # upload this machine's copies (needs a write token)

HF_TOKEN in .env authenticates. SOP_DEMO_REPO names the repo (default: <your HF user>/sop-demo, created private on the
first publish). Fetching again downloads only what is missing or changed. ./start.sh offers the fetch when files are
missing.
"""

import argparse
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import yaml

LAYA = "data/training/models/laya-sop-v2-e12"
LUG_SPLITS = "datasets/lug_bolts/splits.yaml"
LUG_RUNS = "datasets/lug_bolts/runs"

# Uploaded one group per commit, so an interrupted publish keeps the groups already done.
GROUPS = {
    "detectors": ["models/server_fan/*", "models/lug_bolts/*"],
    "server_fan recordings": ["sop-sample-training-data/server_fan/raw/*",
                              "sop-sample-training-data/server_fan/test/*"],
    "mistake scenarios": ["data/scenarios/*"],
    # The weights and configs Laya serves from; checkpoint_latest/ is training-resume state and stays here.
    "laya fine-tune": ["data/training/models/registry.json", f"{LAYA}/model.safetensors", f"{LAYA}/questions.json",
                       f"{LAYA}/rl_agent_config.json", f"{LAYA}/encoder/*", f"{LAYA}/tokenizer/*"],
    "lug_bolts recordings": [LUG_SPLITS, f"{LUG_RUNS}/*"],
}
PUBLISH = [p for group in GROUPS.values() for p in group]
IGNORE = ["*/checkpoint_latest/*", "*/__pycache__/*", ".cache/*"]
CORE = [p for name, group in GROUPS.items() if name != "lug_bolts recordings" for p in group]


def test_runs(splits_text: str) -> list[str]:
    return list((yaml.safe_load(splits_text) or {}).get("test") or [])


def fetch_patterns(scope: str, runs: list[str]) -> list[str]:
    """`core`: everything but the lug-bolt runs outside the test split; `all`: everything published."""
    if scope == "all":
        return PUBLISH
    return CORE + [LUG_SPLITS] + [f"{LUG_RUNS}/{run}/*" for run in runs]


def _selected(path: str) -> bool:
    return any(fnmatch(path, p) for p in PUBLISH) and not any(fnmatch(path, p) for p in IGNORE)


def _repo(api) -> str:
    return os.environ.get("SOP_DEMO_REPO") or f"{api.whoami()['name']}/sop-demo"


def publish(api, root: Path) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and _selected(p.relative_to(root).as_posix())]
    unreadable = [p for p in files if not os.access(p, os.R_OK)]
    if unreadable:
        sys.exit("Can't read (created as root by a container?):\n  " + "\n  ".join(map(str, unreadable))
                 + "\nFix with: sudo chmod a+r <those files>")
    for name, patterns in GROUPS.items():
        if not any(any(fnmatch(p.relative_to(root).as_posix(), pat) for pat in patterns) for p in files):
            sys.exit(f"Nothing to publish for {name} ({', '.join(patterns)}): is this the machine with the demo?")
    repo = _repo(api)
    api.create_repo(repo, private=True, exist_ok=True)
    for name, patterns in GROUPS.items():
        print(f"Uploading {name} ...", flush=True)
        api.upload_folder(repo_id=repo, folder_path=root, allow_patterns=patterns, ignore_patterns=IGNORE,
                          commit_message=f"Demo assets: {name}")
    print(f"Published to https://huggingface.co/{repo} (private)")


def fetch(api, root: Path, scope: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    repo = _repo(api)
    splits = Path(hf_hub_download(repo, LUG_SPLITS, local_dir=root)).read_text()
    print(f"Fetching the demo assets ({scope}) from {repo} ...", flush=True)
    snapshot_download(repo, local_dir=root, allow_patterns=fetch_patterns(scope, test_runs(splits)))
    print("Demo assets in place.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the demo assets into this repo")
    f.add_argument("--all", action="store_true", help="every lug-bolt run, not only the test split")
    sub.add_parser("publish", help="upload this machine's demo assets (needs a write token)")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set: add a Hugging Face access token to .env (read for fetch, write for publish).")
    from huggingface_hub import HfApi

    api, root = HfApi(), Path.cwd()
    if args.cmd == "publish":
        publish(api, root)
    else:
        fetch(api, root, "all" if args.all else "core")


if __name__ == "__main__":
    main()
