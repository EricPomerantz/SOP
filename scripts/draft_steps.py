"""Draft each recording's step timeline (steps.csv) from a replay through the detector and tracker, for a person to
correct; then install the corrected files into the dataset.

    # 1. drafts -> data/<sop dataset dir>/steps/<run>.csv (+ REVIEW.md), owned by whoever owns that folder
    mkdir -p data/lug_bolts/steps
    docker compose run --rm -v ./data/lug_bolts:/app/data/lug_bolts --entrypoint python sopvision \
        scripts/draft_steps.py draft --sop sops/lug_bolts.yaml
    # 2. after correcting them: copy into <dataset>/runs/<run>/steps.csv
    docker compose run --rm -v ./data/lug_bolts:/app/data/lug_bolts --entrypoint python sopvision \
        scripts/draft_steps.py install --sop sops/lug_bolts.yaml

The dashboard's SOP builder does the same per run (sopvision.builder.timelines, steps.draft.csv in the run folder)
and has a page to correct it; this script is the file-based way. How a draft is cut: the tracker's completion times end the steps (a bolt seen seated, a torque result attributed),
and each step runs from the previous completion to its own, so fetching the next bolt counts as working on it (the
server_fan clips' convention). Stretches of IDLE_GAP seconds or more with no hand or tool in view become `idle`, as
does everything after the last completion. Steps the tracker never completed, and deviations it can't see (a bolt
removed, the wheel wiggled, someone reaching across the hub: `off_sop`), are for the reviewer: REVIEW.md lists each
run's alerts next to its draft.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

from sopvision.builder import timelines
from sopvision.builder.timelines import IDLE_GAP, Draft, cut  # noqa: F401 — the cutting lives with the builder
from sopvision.config import SopConfig, load_sop
from sopvision.perception.detector import YoloDetector
from sopvision.recordings import Recording, dataset_root, load_recordings
from sopvision.settings import Settings



def replay(sop: SopConfig, rec: Recording, detector: YoloDetector) -> Draft:
    """The tracker's view of a recording (sopvision.builder.timelines.replay)."""
    return timelines.replay(sop, rec.video, rec.torque_log, rec.source_id, detector)


def write_csv(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_id", "start_s", "end_s"])
        w.writerows(rows)
    _own_like_parent(path)


def _own_like_parent(path: Path) -> None:
    """In the container we run as root: hand the file to whoever owns its folder, so they can edit it."""
    st = path.parent.stat()
    try:
        os.chown(path, st.st_uid, st.st_gid)
    except PermissionError:
        pass


def draft(sop: SopConfig, settings: Settings, out: Path, only: list[str]) -> None:
    detector = YoloDetector(sop.detector.weights, sop.detector.classes)
    out.mkdir(parents=True, exist_ok=True)
    review = [f"# Step timelines to review ({sop.sop})", "",
              "Each `<run>.csv` is a draft: rows `step_id,start_s,end_s`, contiguous, times in the video's seconds.",
              "Check each against the video and fix it, then run `scripts/draft_steps.py install`. Step ids:",
              ", ".join(s.id for s in sop.steps) + ", plus `idle` (nobody working) and `off_sop` (doing something",
              "that isn't an SOP step: a bolt taken out, the wheel wiggled, reaching across the hub).", ""]
    for rec in load_recordings(sop, settings):
        if only and rec.id not in only:
            continue
        target = out / f"{rec.id}.csv"
        if target.exists():
            print(f"{rec.id}: {target} exists, kept (delete it to redraft)")
            continue
        d = replay(sop, rec, detector)
        write_csv(target, d.rows)
        review += [f"## {rec.id} ({rec.split})", ""]
        review += [f"- not completed by the tracker: {', '.join(d.not_done)}"] if d.not_done else []
        review += [f"- {a}" for a in d.alerts] or ["- no alerts"]
        review.append("")
        print(f"{rec.split}:{rec.id}: {len(d.rows)} rows, {len(d.alerts)} alerts" + (f", not done {d.not_done}" if d.not_done else ""))
    (out / "REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    _own_like_parent(out / "REVIEW.md")


def check_contiguous(rows: list[tuple[str, float, float]]) -> None:
    """Rows in file order must each start where the previous one ended: a gap leaves frames unlabelled."""
    for i, (step, a, b) in enumerate(rows):
        if b <= a:
            raise ValueError(f"{step} {a}-{b}: ends before it starts")
        if i and abs(a - rows[i - 1][2]) > 0.01:
            raise ValueError(f"{step} starts at {a} but the row before ends at {rows[i - 1][2]}")


def install(sop: SopConfig, settings: Settings, src: Path) -> int:
    from sopvision.recordings import read_steps

    root, errors = dataset_root(sop, settings), 0
    for path in sorted(src.glob("*.csv")):
        folder = root / "runs" / path.stem
        if not folder.is_dir():
            print(f"{path.name}: no run {path.stem} in {root}", file=sys.stderr)
            errors += 1
            continue
        try:
            read_steps(path, sop)  # unknown step ids or bad numbers fail here, before anything is copied
            with path.open(newline="", encoding="utf-8") as f:
                check_contiguous([(r["step_id"], float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(f)])
        except Exception as e:  # noqa: BLE001 — report every bad file, then stop
            print(f"{path.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        shutil.copyfile(path, folder / "steps.csv")
        print(f"{path.stem}: installed")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["draft", "install"])
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--dir", type=Path, help="drafts folder (default data/<dataset dir>/steps)")
    ap.add_argument("--run", action="append", default=[], help="only these runs (repeatable)")
    args = ap.parse_args()
    sop, settings = load_sop(args.sop), Settings()
    folder = args.dir or Path("data") / (sop.dataset.dir or sop.sop) / "steps"
    if args.cmd == "draft":
        draft(sop, settings, folder, args.run)
        return 0
    return install(sop, settings, folder)


if __name__ == "__main__":
    raise SystemExit(main())
