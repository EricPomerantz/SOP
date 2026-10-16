"""Step timelines (steps.csv) for an SOP's recordings: the tracker's draft, a browser preview, checks before saving.

    python -m sopvision.builder.timelines --sop sops/drafts/<id>.yaml [--run <run id>]... [--previews-only]

For each recorded run this writes, next to its video:
  steps.draft.csv   the tracker's view of the run, for a person to correct on the timeline page
  steps.draft.json  what the tracker couldn't settle: steps it never completed, its alerts
  preview.webm      the video re-encoded for the browser (VP8; the recordings are MPEG-4 Part 2, which browsers
                    don't play), same frames and timing, so its times are the video's

How a draft is cut: the tracker's completion times end the steps (a part seen in its slot, a torque result
attributed), and each step runs from the previous completion to its own, so fetching the next part counts as working
on it. Stretches of IDLE_GAP seconds or more with no hand or tool in view become `idle`, as does everything after
the last completion. Deviations the tracker can't see (a part taken out again, off-SOP handling) are for the person.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sopvision.config import SopConfig, load_sop

IDLE_GAP = 5.0  # seconds with nobody at the work piece that count as idle
PREVIEW_WIDTH = 960
EXTRA_STEPS = ("idle", "off_sop")


@dataclass(frozen=True)
class Draft:
    rows: list[tuple[str, float, float]]  # step_id, start_s, end_s
    alerts: list[str]
    not_done: list[str]


def cut(done: list[tuple[float, str]], present: list[tuple[float, bool]], duration: float) -> list[tuple[str, float, float]]:
    """Contiguous segments ending at each completion; long stretches with nobody present split out as idle."""
    rows, start = [], 0.0
    for t, step in sorted(done) + [(duration, "idle")]:
        idle = _idle_spans(present, start, t)
        for a, b in idle:
            if a > start:
                rows.append((step, start, a))
            rows.append(("idle", a, b))
            start = b
        if t > start:
            rows.append((step, start, t))
        start = t
    merged: list[tuple[str, float, float]] = []
    for step, a, b in rows:  # back-to-back idle pieces become one
        if merged and merged[-1][0] == step == "idle":
            merged[-1] = (step, merged[-1][1], b)
        else:
            merged.append((step, a, b))
    return [(s, round(a, 2), round(b, 2)) for s, a, b in merged if b - a > 0.01]


def _idle_spans(present: list[tuple[float, bool]], start: float, end: float) -> list[tuple[float, float]]:
    spans, since = [], None
    for t, here in present:
        if t < start or t > end:
            continue
        if not here and since is None:
            since = t
        elif here and since is not None:
            if t - since >= IDLE_GAP:
                spans.append((since, t))
            since = None
    if since is not None and end - since >= IDLE_GAP:
        spans.append((since, end))
    return spans


def check_rows(rows: list[tuple[str, float, float]], step_ids: set[str], duration: float | None = None) -> None:
    """A timeline that can be saved: known steps, rows in order each starting where the previous ended (a gap leaves
    frames unlabelled), from 0 to the end of the video when its length is known. Raises ValueError saying what's wrong."""
    if not rows:
        raise ValueError("The timeline is empty")
    known = set(step_ids) | set(EXTRA_STEPS)
    for i, (step, a, b) in enumerate(rows):
        if step not in known:
            raise ValueError(f"Unknown step '{step}'")
        if b <= a:
            raise ValueError(f"{step} {a}-{b}: ends before it starts")
        if i and abs(a - rows[i - 1][2]) > 0.01:
            raise ValueError(f"{step} starts at {a} but the row before ends at {rows[i - 1][2]}")
    if abs(rows[0][1]) > 0.01:
        raise ValueError(f"The first row starts at {rows[0][1]}, not 0")
    if duration is not None and abs(rows[-1][2] - duration) > 0.5:
        raise ValueError(f"The last row ends at {rows[-1][2]}, but the video is {duration:.2f} s long")


def read_rows(path: Path) -> list[tuple[str, float, float]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [(r["step_id"].strip(), float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(f)]


def write_rows(path: Path, rows: list[tuple[str, float, float]]) -> None:
    tmp = Path(path).with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_id", "start_s", "end_s"])
        w.writerows((s, round(a, 2), round(b, 2)) for s, a, b in rows)
    tmp.replace(path)


def replay(sop: SopConfig, video: Path, torque_log: Path | None, source_id: str, detector) -> Draft:
    """The tracker's view of a recording: no decision model, completions from evidence and torque results alone."""
    from sopvision.pipeline import RunCore
    from sopvision.scene.builder import SceneBuilder
    from sopvision.sop.torque import read_torque_log
    from sopvision.sources.file import FileSource

    detector.reset()
    core = RunCore(sop, "draft", source_id, None)
    builder = SceneBuilder(sop)
    core.mark_torqued = builder.mark_torqued
    if torque_log is not None and core.torque is not None:
        for result in read_torque_log(torque_log):
            core.on_torque(result)
    present, t = [], 0.0
    for frame in FileSource(video, mode="lockstep").frames():
        scene, events = builder.update(frame.t, detector.detect(frame))
        core.on_scene(scene, events)
        present.append((frame.t, bool(scene.hands) or bool(scene.tools)))
        t = frame.t
    core.finish()
    steps = core.tracker.steps()
    done = [(s.done_t, s.id) for s in steps if s.status == "done" and s.done_t is not None]
    alerts = [f"{a.t:.1f}s {a.level} {a.kind}: {a.message}" for a in core.alerts if a.level in ("warning", "violation")]
    return Draft(cut(done, present, t), alerts, [s.id for s in steps if s.status != "done"])


def video_info(video: Path) -> tuple[float, float]:
    """(fps, duration in seconds) as the app reads the video (frame index / fps)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, round(frames / fps, 2) if frames > 0 else 0.0


def write_preview(video: Path, out: Path, width: int = PREVIEW_WIDTH) -> None:
    """Every frame, smaller, as VP8 WebM at the same frame rate: times on the page are the video's times."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer, tmp = None, out.with_name(out.stem + ".tmp.webm")
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            h, w = image.shape[:2]
            if writer is None:
                size = (width, int(round(h * width / w / 2)) * 2) if w > width else (w, h)
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("OpenCV can't write VP8 WebM here")
            writer.write(cv2.resize(image, size) if size != (w, h) else image)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if tmp.exists():
        tmp.replace(out)


def draft_runs(sop: SopConfig, settings, only: list[str], detector=None, previews: bool = True) -> list[str]:
    from sopvision.perception.detector import YoloDetector
    from sopvision.recordings import load_recordings

    detector = detector or YoloDetector(sop.detector.weights, sop.detector.classes)
    done = []
    for rec in load_recordings(sop, settings):
        if only and rec.id not in only:
            continue
        folder = rec.video.parent
        d = replay(sop, rec.video, rec.torque_log, rec.source_id, detector)
        write_rows(folder / "steps.draft.csv", d.rows)
        (folder / "steps.draft.json").write_text(json.dumps({"alerts": d.alerts, "not_done": d.not_done}, indent=1) + "\n",
                                                 encoding="utf-8")
        print(f"{rec.split}:{rec.id}: {len(d.rows)} rows, {len(d.alerts)} alerts"
              + (f", not done {d.not_done}" if d.not_done else ""), flush=True)
        if previews and not (folder / "preview.webm").exists():
            write_preview(rec.video, folder / "preview.webm")
            print(f"{rec.id}: preview written", flush=True)
        done.append(rec.id)
    return done


def write_previews(sop_file: Path, settings, only: list[str]) -> list[str]:
    """Browser previews only (no detector, no valid SOP needed): for labelling runs before anything is drafted."""
    import yaml

    data = yaml.safe_load(Path(sop_file).read_text(encoding="utf-8")) or {}
    root = Path(settings.datasets_dir) / ((data.get("dataset") or {}).get("dir") or data.get("sop") or "")
    done = []
    for video in sorted((root / "runs").glob("*/video.mp4")):
        run = video.parent.name
        if only and run not in only:
            continue
        if not (video.parent / "preview.webm").exists():
            write_preview(video, video.parent / "preview.webm")
            print(f"{run}: preview written", flush=True)
        done.append(run)
    return done


def main() -> int:
    from sopvision.settings import Settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sop", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], help="only these runs (repeatable)")
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--previews-only", action="store_true", help="only the browser previews (no draft)")
    args = ap.parse_args()
    if args.previews_only:
        print(f"{len(write_previews(args.sop, Settings(), args.run))} previews ready", flush=True)
        return 0
    runs = draft_runs(load_sop(args.sop), Settings(), args.run, previews=not args.no_previews)
    print(f"{len(runs)} runs drafted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
