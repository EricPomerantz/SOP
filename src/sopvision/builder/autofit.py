"""Suggest slot positions from the SOP's recordings: where parts and empty-slot markers sit on the anchor's box.

    python -m sopvision.builder.autofit --sop sops/drafts/<id>.yaml [--every 1.0] [--threshold 0.08]

About one frame a second from every recorded run goes through the detector. Each part or empty-marker detection is
put in that frame's anchor-box coordinates (0-1 across and down, as slot positions are), so an anchor that moves
between frames or runs (a chassis slid across the bench, a shaking camera) still gives one tight cloud per slot.
Frames whose anchor box is far from its usual size are dropped: a half-hidden or half-out-of-view anchor shrinks its
box and would shift every point. The points are clustered (average linkage, cut at `threshold`); clusters seen in
fewer than `min_share` of the frames are dropped as noise. Per cluster: the median position, its spread (90th
percentile distance from the median), a suggested match radius, and the classes seen there (-> filled_by, empty_as).

A rotating anchor can't be represented (slots are fixed fractions of an axis-aligned box): when the box's aspect
ratio varies a lot, or clusters smear, the result carries a warning. Writes data/builder/<sop id>/autofit.json and
one sample frame per run (frames/<run>.jpg) with its anchor box, for the layout editor.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SIZE_TOLERANCE = 0.15  # an anchor box within ±15% of the median width and height counts
ASPECT_WARNING = 0.2  # p90-p10 spread of the box aspect ratio, relative to its median
SPREAD_WARNING = 0.06  # a cluster's spread above this is worth checking
MAX_POINTS = 3000  # clustering is quadratic in points


@dataclass
class FrameObs:
    run: str
    t: float
    anchor: tuple[float, float, float, float]  # pixels
    points: list[tuple[str, float, float]]  # class, u, v (anchor-box units)


@dataclass
class Fit:
    slots: list[dict] = field(default_factory=list)
    match_radius: float = 0.1
    frames_used: int = 0
    frames_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def fit(frames: list[FrameObs], part_classes: set[str], empty_classes: set[str], threshold: float = 0.08,
        min_share: float = 0.1, seed: int = 0) -> Fit:
    from scipy.cluster.hierarchy import fcluster, linkage

    out = Fit()
    if not frames:
        out.warnings.append("No frame showed the anchor: check the anchor class and the recordings.")
        return out
    w = np.array([f.anchor[2] - f.anchor[0] for f in frames])
    h = np.array([f.anchor[3] - f.anchor[1] for f in frames])
    keep = (np.abs(w / np.median(w) - 1) <= SIZE_TOLERANCE) & (np.abs(h / np.median(h) - 1) <= SIZE_TOLERANCE)
    used = [f for f, k in zip(frames, keep) if k]
    out.frames_used, out.frames_dropped = len(used), len(frames) - len(used)
    aspect = w / h  # over every frame: a turned anchor's frames are also the ones the size check drops
    if (np.percentile(aspect, 90) - np.percentile(aspect, 10)) / np.median(aspect) > ASPECT_WARNING:
        out.warnings.append("The anchor's box changes shape a lot between frames: if the work piece rotates, slot "
                            "positions won't follow it. Keep it in one orientation.")
    if out.frames_dropped > 0.3 * len(frames):
        out.warnings.append(f"{out.frames_dropped} of {len(frames)} frames were dropped because the anchor's box was "
                            "much smaller or larger than usual (partly hidden, out of view, or the camera moved).")
    wanted = part_classes | empty_classes
    pts = [(f_i, c, u, v) for f_i, f in enumerate(used) for c, u, v in f.points if c in wanted]
    if len(pts) < 2:
        out.warnings.append("No parts or empty-slot markers were seen on the anchor: check the class roles.")
        return out
    rng = np.random.default_rng(seed)
    if len(pts) > MAX_POINTS:
        pts = [pts[i] for i in sorted(rng.choice(len(pts), MAX_POINTS, replace=False))]
    xy = np.array([(u, v) for _, _, u, v in pts])
    labels = fcluster(linkage(xy, "average"), t=threshold, criterion="distance")
    sampled_frames = len({f_i for f_i, *_ in pts})
    clusters = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        share = len({pts[i][0] for i in idx}) / sampled_frames
        if share < min_share:
            continue
        centre = np.median(xy[idx], axis=0)
        spread = float(np.percentile(np.linalg.norm(xy[idx] - centre, axis=1), 90))
        seen = Counter(pts[i][1] for i in idx)
        sample = idx if len(idx) <= 200 else rng.choice(idx, 200, replace=False)
        clusters.append({"at": [round(float(centre[0]), 3), round(float(centre[1]), 3)], "spread": round(spread, 4),
                         "share": round(share, 3), "classes": dict(seen.most_common()),
                         "points": [[round(float(a), 3), round(float(b), 3)] for a, b in xy[sample]]})
    # Reading order: top to bottom in rows, left to right within a row.
    clusters.sort(key=lambda c: (round(c["at"][1] / max(threshold, 0.05)), c["at"][0]))
    centres = np.array([c["at"] for c in clusters]) if clusters else np.zeros((0, 2))
    for i, c in enumerate(clusters):
        others = np.delete(centres, i, axis=0)
        nn = float(np.min(np.linalg.norm(others - centres[i], axis=1))) if len(others) else 0.5
        c["nearest"] = round(nn, 3)
        c["radius"] = round(min(0.4 * nn, max(1.5 * c["spread"], 0.05)), 3)
        total = sum(c["classes"].values())
        c["filled_by"] = [k for k, n in c["classes"].items() if k in part_classes and n >= 0.05 * total]
        empties = [k for k in c["classes"] if k in empty_classes]
        c["empty_as"] = empties[0] if empties else None
        c["name"] = f"slot_{i + 1}"
        if c["spread"] > SPREAD_WARNING:
            out.warnings.append(f"{c['name']} is spread out (±{c['spread']:.2f}): check it on the sample frames; it may "
                                "be two slots, or a part that moves relative to the anchor.")
    out.slots = clusters
    if clusters:
        out.match_radius = round(float(np.median([c["radius"] for c in clusters])), 3)
    return out


def _yolo(weights: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names

    def detect(image) -> list[tuple[str, tuple[float, float, float, float], float]]:
        r = model.predict(image, imgsz=1280, verbose=False)[0]
        return [(names[int(c)], tuple(b), float(s)) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                                         r.boxes.conf.tolist())]

    return detect


def collect(videos: dict[str, Path], anchor_class: str, detect, every: float = 1.0, max_per_run: int = 300,
            samples_dir: Path | None = None) -> tuple[list[FrameObs], dict]:
    """Detections from about one frame per `every` seconds of each video; one sample frame per run is saved."""
    import cv2

    frames, samples = [], {}
    for run, video in videos.items():
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step, index, taken = max(1, round(fps * every)), 0, 0
        while taken < max_per_run:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                dets = detect(image)
                anchors = [d for d in dets if d[0] == anchor_class]
                if anchors:
                    a = max(anchors, key=lambda d: d[2])[1]
                    aw, ah = a[2] - a[0], a[3] - a[1]
                    points = [(c, ((b[0] + b[2]) / 2 - a[0]) / aw, ((b[1] + b[3]) / 2 - a[1]) / ah)
                              for c, b, _ in dets if c != anchor_class]
                    frames.append(FrameObs(run, index / fps, tuple(float(v) for v in a), points))
                    if samples_dir is not None and run not in samples:
                        samples_dir.mkdir(parents=True, exist_ok=True)
                        h, w = image.shape[:2]
                        scale = min(1.0, 1280 / w)
                        small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image
                        cv2.imwrite(str(samples_dir / f"{run}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        samples[run] = {"t": round(index / fps, 2), "width": small.shape[1], "height": small.shape[0],
                                        "anchor": [round(v * scale, 1) for v in a]}
                taken += 1
            index += 1
        cap.release()
        print(f"{run}: {taken} frames sampled", flush=True)
    return frames, samples


def output_dir(sop_id: str) -> Path:
    return Path("data/builder") / sop_id


def main() -> int:
    from sopvision.settings import Settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sop", type=Path, required=True, help="the SOP file (a draft is fine)")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--threshold", type=float, default=0.08, help="cluster cut distance (anchor-box units)")
    ap.add_argument("--min-share", type=float, default=0.1, help="drop clusters seen in fewer frames than this share")
    args = ap.parse_args()
    data = yaml.safe_load(args.sop.read_text(encoding="utf-8"))
    sop_id, roles = data["sop"], (data.get("detector") or {}).get("roles") or {}
    anchor = (data.get("layout") or {}).get("anchor")
    if not anchor:
        print("The SOP has no anchor class yet (detector stage)", flush=True)
        return 2
    root = Settings().datasets_dir / ((data.get("dataset") or {}).get("dir") or sop_id)
    videos = {p.parent.name: p for p in sorted((root / "runs").glob("*/video.mp4"))}
    if not videos:
        print(f"No recordings in {root}/runs", flush=True)
        return 2
    out = output_dir(sop_id)
    frames, samples = collect(videos, anchor, _yolo(str(data["detector"]["weights"])), args.every,
                              samples_dir=out / "frames")
    parts = {c for c, r in roles.items() if r == "part"}
    empties = {c for c, r in roles.items() if r == "empty"}
    result = fit(frames, parts, empties, args.threshold, args.min_share)
    out.mkdir(parents=True, exist_ok=True)
    (out / "autofit.json").write_text(json.dumps({
        "sop": sop_id, "threshold": args.threshold, "every": args.every, "slots": result.slots,
        "match_radius": result.match_radius, "frames_used": result.frames_used,
        "frames_dropped": result.frames_dropped, "warnings": result.warnings, "samples": samples,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"{len(result.slots)} slots from {result.frames_used} frames ({result.frames_dropped} dropped)", flush=True)
    for w in result.warnings:
        print(f"warning: {w}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
