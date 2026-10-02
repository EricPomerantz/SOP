"""Reference-frame labelling: the user draws boxes on 3 frames of one run once; everything else is derived.

Step 1 (this file, `draw`): an OpenCV window asks for each group of boxes in turn and saves
data/reference/Install_<run>.json with the boxes in full-resolution pixels.

Usage: uv run python -m training.reference_labels draw --run 4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from sopvision.dataset import load_index
from sopvision.settings import Settings

# frame name -> [(class, how many boxes, instruction)]
REFERENCE_GROUPS: dict[str, list[tuple[str, int, str]]] = {
    "start": [
        ("chassis", 1, "Draw ONE box around the whole server"),
        ("fan_bay_empty", 6, "Draw a box on each of the 6 EMPTY FAN BAYS"),
        ("psu_bay_empty", 2, "2 EMPTY POWER SUPPLY BAYS: grey bays at the top-left and bottom-left edges (not the riser cages)"),
    ],
    "filled": [
        ("fan", 6, "Draw a box on each of the 6 INSTALLED FANS"),
        ("psu", 2, "2 INSTALLED POWER SUPPLIES: grey bays at the left edges, include the black handle"),
    ],
    "covered": [
        ("cover", 1, "Draw ONE box around the COVER that was just installed (not the fixed right-hand panel)"),
    ],
}
# Times in the reference run (Install_4) where nothing blocks the parts.
REFERENCE_TIMES = {4: {"start": 0.3, "filled": 74.8, "covered": 86.0}}
WINDOW_WIDTH = 1600


def display_to_image(rois, scale: float) -> list[tuple[float, float, float, float]]:
    """selectROIs (x, y, w, h) boxes in the scaled window → (x1, y1, x2, y2) in image pixels."""
    return [(x / scale, y / scale, (x + w) / scale, (y + h) / scale) for x, y, w, h in rois]


def transform_box(xyxy, h: np.ndarray) -> tuple[float, float, float, float]:
    """Map a box through a homography; return the axis-aligned bounds of the mapped corners."""
    x1, y1, x2, y2 = xyxy
    corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float64)
    pts = cv2.perspectiveTransform(corners, np.asarray(h, np.float64)).reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def assign_bay_identities(change: np.ndarray) -> list[int]:
    """change[step][bay] = how much each bay changed during that step → bay index for each step."""
    steps, bays = linear_sum_assignment(-np.asarray(change, float))
    order = dict(zip(steps, bays))
    return [int(order[s]) for s in range(len(order))]


def bay_state(step_action: int, actions_by_position: list[int], position: int) -> str:
    """'installed' / 'empty' / 'in_progress' for the bay filled by `step_action`, at a frame in `position`."""
    step_position = actions_by_position.index(step_action) + 1
    if step_position < position:
        return "installed"
    if step_position == position:
        return "in_progress"
    return "empty"


CROP_SIZE = (32, 32)
HIDDEN_FACTOR = 0.6  # a bay is hidden/blocked when it matches neither reference within this share of their separation
HIDDEN_FLOOR = 0.6  # ...or within this absolute diff, which cross-recording noise stays under (measured: p90 0.34)


def appearance_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two crops after resizing and z-normalising (ignores brightness/contrast)."""
    def norm(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, CROP_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)

    return float(np.abs(norm(a) - norm(b)).mean())


def focus_region(empty_crop: np.ndarray, full_crop: np.ndarray, keep: float = 0.5, pad: float = 0.05):
    """Part of a bay box (fractions x1, y1, x2, y2) where the empty and installed looks differ most.

    A fan changes its whole bay; a power supply only changes at its handle end, so comparing just
    this region keeps small changes from being drowned out.
    """
    a = cv2.cvtColor(empty_crop, cv2.COLOR_BGR2GRAY)
    b = cv2.resize(cv2.cvtColor(full_crop, cv2.COLOR_BGR2GRAY), (a.shape[1], a.shape[0]))
    d = cv2.GaussianBlur(cv2.absdiff(a, b), (5, 5), 0).astype(np.float32)
    if d.max() < 10:
        return (0.0, 0.0, 1.0, 1.0)
    ys, xs = np.nonzero(d >= keep * d.max())
    h, w = d.shape
    return (max(0.0, xs.min() / w - pad), max(0.0, ys.min() / h - pad),
            min(1.0, (xs.max() + 1) / w + pad), min(1.0, (ys.max() + 1) / h + pad))


def sub_box(box, focus) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    fx1, fy1, fx2, fy2 = focus
    w, h = x2 - x1, y2 - y1
    return (x1 + fx1 * w, y1 + fy1 * h, x1 + fx2 * w, y1 + fy2 * h)


def decide_bay(state: str, d_empty: float, d_full: float, separation: float, occluded: bool,
               empty_cls: str, full_cls: str) -> tuple[str | None, str | None]:
    """Label for one bay in one frame: (class or None to leave it unlabelled, flag reason or None)."""
    if occluded or min(d_empty, d_full) > max(HIDDEN_FACTOR * separation, HIDDEN_FLOOR):
        return None, None
    looks_full = d_full < d_empty
    if state == "in_progress":
        return (full_cls if looks_full else empty_cls), None
    want_full = state == "installed"
    cls = full_cls if want_full else empty_cls
    if looks_full != want_full:
        return cls, f"{cls}: looks {'installed' if looks_full else 'empty'}"
    return cls, None


def read_frame(video: Path, t: float):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"can't read {video} at {t:.1f}s")
        return frame
    finally:
        cap.release()


NEXT_KEYS = {ord("n"), ord("N"), 13, 10, 32}  # N, Enter, Space
UNDO_KEYS = {ord("u"), ord("U"), 8}  # U, Backspace


def _ask(frame, scale: float, cls: str, n: int, text: str) -> list[tuple[float, float, float, float]]:
    """Drag to draw; each box is kept when the mouse is released. U undoes, N goes to the next round."""
    base = cv2.resize(frame, None, fx=scale, fy=scale)
    title = "Reference labels"
    boxes: list[tuple[int, int, int, int]] = []  # (x, y, w, h) in window pixels
    drag: dict = {"start": None, "cur": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = drag["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
            drag["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
            (x0, y0), drag["start"] = drag["start"], None
            if abs(x - x0) > 4 and abs(y - y0) > 4:
                boxes.append((min(x, x0), min(y, y0), abs(x - x0), abs(y - y0)))

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        shown = base.copy()
        for i, (x, y, w, h) in enumerate(boxes, 1):
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(shown, str(i), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        if drag["start"]:
            cv2.rectangle(shown, drag["start"], drag["cur"], (0, 255, 255), 2)
        done = len(boxes) >= n
        status = f"All {n} drawn - press N for next (U = undo)" if done else f"{len(boxes)}/{n} drawn - U = undo, N = next"
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(shown, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(shown, status, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0) if done else (255, 255, 255), 2)
        cv2.imshow(title, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in UNDO_KEYS and boxes:
            boxes.pop()
        elif key in NEXT_KEYS or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return display_to_image(boxes, scale)


def replace_classes(existing: dict, new_boxes: dict[str, list[dict]], classes: set[str]) -> dict:
    """Swap in freshly drawn boxes for `classes`, keeping every other saved box."""
    for name, boxes in new_boxes.items():
        frame = existing["frames"][name]
        frame["boxes"] = [b for b in frame["boxes"] if b["cls"] not in classes] + boxes
    return existing


def draw(run_no: int, out_dir: Path, only: set[str] | None = None) -> Path:
    run = {r.run: r for r in load_index(Settings().sop_data_root)}[run_no]
    path = out_dir / f"Install_{run_no}.json"
    drawn: dict[str, list[dict]] = {}
    for name, groups in REFERENCE_GROUPS.items():
        groups = [g for g in groups if only is None or g[0] in only]
        if not groups:
            continue
        frame = read_frame(run.raw_path, REFERENCE_TIMES[run_no][name])
        scale = WINDOW_WIDTH / frame.shape[1]
        drawn[name] = []
        for cls, n, text in groups:
            got = _ask(frame, scale, cls, n, text)
            if len(got) != n:
                print(f"note: {cls}: expected {n} boxes, got {len(got)}")
            drawn[name] += [{"cls": cls, "xyxy": [round(v, 1) for v in b]} for b in got]
    if only:
        result = replace_classes(json.loads(path.read_text(encoding="utf-8")), drawn, only)
    else:
        result = {"run": run_no, "frames": {name: {"t": REFERENCE_TIMES[run_no][name], "boxes": boxes}
                                            for name, boxes in drawn.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


FAN_ACTIONS = (1, 2, 3, 4, 5, 6)
PSU_ACTIONS = (7, 8)
COVER_ACTION = 9
MIN_INLIERS = 40
HAND_OCCLUDES = 0.3  # share of a bay covered by a hand box before we leave the bay unlabelled


class Aligner:
    """Homography from a reference frame to another frame, using SIFT features on the server only."""

    def __init__(self, ref_bgr: np.ndarray, roi_xyxy):
        self.sift = cv2.SIFT_create(nfeatures=4000)
        gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        x1, y1, x2, y2 = (int(v) for v in roi_xyxy)
        mask[max(0, y1):y2, max(0, x1):x2] = 255
        self.kp, self.des = self.sift.detectAndCompute(gray, mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def align(self, img_bgr: np.ndarray) -> tuple[np.ndarray | None, int]:
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), None)
        if des is None or len(kp) < MIN_INLIERS:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(self.des, des, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return None, len(good)
        src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        n = int(inliers.sum()) if inliers is not None else 0
        return (h if h is not None and n >= MIN_INLIERS else None), n


def _crop(img: np.ndarray, xyxy) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2] if x2 - x1 > 4 and y2 - y1 > 4 else np.zeros((8, 8, 3), np.uint8)


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _pair_boxes(empty_boxes: list, full_boxes: list) -> list[tuple]:
    """Pair each empty-bay box with the installed-part box nearest to it."""
    cost = np.array([[np.hypot(*np.subtract(_center(e), _center(f))) for f in full_boxes] for e in empty_boxes])
    rows, cols = linear_sum_assignment(cost)
    return [(empty_boxes[r], full_boxes[c]) for r, c in zip(rows, cols)]


def _overlap_share(bay, other) -> float:
    ix = max(0.0, min(bay[2], other[2]) - max(bay[0], other[0]))
    iy = max(0.0, min(bay[3], other[3]) - max(bay[1], other[1]))
    area = max(1e-6, (bay[2] - bay[0]) * (bay[3] - bay[1]))
    return ix * iy / area


class Reference:
    """Everything derived from the user's three reference frames."""

    def __init__(self, ref_json: Path, data_root: Path):
        spec = json.loads(Path(ref_json).read_text(encoding="utf-8"))
        self.run = {r.run: r for r in load_index(data_root)}[spec["run"]]
        frames = spec["frames"]
        self.images = {name: read_frame(self.run.raw_path, f["t"]) for name, f in frames.items()}

        def boxes(frame, cls):
            return [tuple(b["xyxy"]) for b in frames[frame]["boxes"] if b["cls"] == cls]

        self.chassis = boxes("start", "chassis")[0]
        self.cover = boxes("covered", "cover")[0]
        start, filled = self.images["start"], self.images["filled"]
        # bays: (empty box, installed box), ordered by the step that fills them
        self.bays: dict[int, tuple] = {}
        self.focus: dict[int, tuple] = {}  # per bay: the sub-region compared between looks
        for actions, empty_cls, full_cls in ((FAN_ACTIONS, "fan_bay_empty", "fan"), (PSU_ACTIONS, "psu_bay_empty", "psu")):
            pairs = _pair_boxes(boxes("start", empty_cls), boxes("filled", full_cls))
            focuses = [focus_region(_crop(start, e), _crop(filled, e)) for e, _ in pairs]
            order = assign_bay_identities(self._step_changes(actions, [sub_box(e, fo) for (e, _), fo in zip(pairs, focuses)]))
            for action, bay in zip(actions, order):
                self.bays[action], self.focus[action] = pairs[bay], focuses[bay]
        self.separation = {a: appearance_diff(_crop(start, sub_box(e, self.focus[a])), _crop(filled, sub_box(e, self.focus[a])))
                           for a, (e, _) in self.bays.items()}
        self.cover_separation = appearance_diff(_crop(start, self.cover), _crop(self.images["covered"], self.cover))
        self.aligners = [Aligner(start, self.chassis), Aligner(self.images["covered"], self.chassis)]

    def _step_changes(self, actions, empty_boxes) -> np.ndarray:
        """change[step][bay]: how much each bay moved away from its empty look across that step's clip."""
        start = self.images["start"]
        segments = {a: s for a, s in zip(self.run.actions_by_position, self.run.segments)}

        def d_empty(t, box):
            vals = [appearance_diff(_crop(read_frame(self.run.raw_path, t + dt), box), _crop(start, box)) for dt in (-0.2, 0, 0.2)]
            return float(np.median(vals))

        return np.array([[d_empty(segments[a].end + 0.6, b) - d_empty(segments[a].start - 0.6, b) for b in empty_boxes]
                         for a in actions])

    def align(self, img: np.ndarray) -> tuple[np.ndarray | None, int]:
        best = (None, 0)
        for aligner in self.aligners:
            h, n = aligner.align(img)
            if n > best[1]:
                best = (h, n)
            if h is not None:
                return h, n
        return best

    def slot_centers(self) -> dict[str, tuple[float, float]]:
        """Bay centres normalised to the reference chassis box, keyed by slot name (for the SOP layout)."""
        x1, y1, x2, y2 = self.chassis

        def norm(b):
            cx, cy = _center(b)
            return (round((cx - x1) / (x2 - x1), 3), round((cy - y1) / (y2 - y1), 3))

        names = {a: f"fan_bay_{a}" for a in FAN_ACTIONS} | {7: "psu_bay_1", 8: "psu_bay_2"}
        centers = {names[a]: norm(e) for a, (e, _) in self.bays.items()}
        centers["cover"] = norm(self.cover)
        return centers


def label_frame(ref: Reference, img: np.ndarray, actions_by_position: list[int], position: int,
                hands: list[tuple]) -> tuple[list[tuple[str, tuple]] | None, list[str], int]:
    """Boxes (class, xyxy) for one frame, flag reasons, and alignment inliers. None when alignment fails."""
    h, inliers = ref.align(img)
    if h is None:
        return None, [f"alignment failed ({inliers} inliers)"], inliers
    start, filled, covered = ref.images["start"], ref.images["filled"], ref.images["covered"]
    labels = [("chassis", transform_box(ref.chassis, h))] + [("hand", b) for b in hands]
    reasons = []
    for action, (empty_box, full_box) in ref.bays.items():
        e, f = transform_box(empty_box, h), transform_box(full_box, h)
        state = bay_state(action, actions_by_position, position)
        if bay_state(COVER_ACTION, actions_by_position, position) == "installed":
            continue  # under the cover
        occluded = any(_overlap_share(e, hb) > HAND_OCCLUDES for hb in hands)
        empty_cls, full_cls = ("fan_bay_empty", "fan") if action in FAN_ACTIONS else ("psu_bay_empty", "psu")
        here, there = _crop(img, sub_box(e, ref.focus[action])), sub_box(empty_box, ref.focus[action])
        cls, reason = decide_bay(state, appearance_diff(here, _crop(start, there)), appearance_diff(here, _crop(filled, there)),
                                 ref.separation[action], occluded, empty_cls, full_cls)
        if cls:
            labels.append((cls, f if cls == full_cls else e))
        if reason:
            reasons.append(reason)
    c = transform_box(ref.cover, h)
    cover_state = bay_state(COVER_ACTION, actions_by_position, position)
    cls, reason = decide_bay(cover_state, appearance_diff(_crop(img, c), _crop(start, ref.cover)),
                             appearance_diff(_crop(img, c), _crop(covered, ref.cover)), ref.cover_separation,
                             False, "", "cover")
    if cls == "cover":
        labels.append(("cover", c))
    if reason:
        reasons.append(reason)
    return labels, reasons, inliers


def generate(ref_json: Path, frames: Path, out: Path, classes: list[str]) -> None:
    from training.autolabel import Sam3Labeler
    from training.yolo_format import LabeledBox, to_yolo

    settings = Settings()
    ref = Reference(ref_json, settings.sop_data_root)
    hands_model = Sam3Labeler({"hand": "hand"})
    rows = [json.loads(l) for l in (frames / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    flags = []
    for i, row in enumerate(rows):
        img = cv2.imread(str(frames / row["image"]))
        hands = [b.xyxy for b in hands_model.label(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
        labels, reasons, inliers = label_frame(ref, img, row["actions_by_position"], row["position"], hands)
        target = out / Path(row["image"]).with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if labels is None:
            target.unlink(missing_ok=True)  # no label file → frame left out of the dataset
        else:
            h, w = img.shape[:2]
            target.write_text("".join(to_yolo(LabeledBox(c, b, 1.0), w, h, classes) + "\n" for c, b in labels),
                              encoding="utf-8")
        flags.append({"image": row["image"], "ok": not reasons, "reasons": reasons, "inliers": inliers})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} frames")
    (out / "flags.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flags), encoding="utf-8")
    failed = sum(1 for f in flags if f["inliers"] < MIN_INLIERS)
    print(f"labelled {len(flags) - failed}/{len(flags)} frames; {sum(not f['ok'] for f in flags)} flagged; "
          f"{failed} failed alignment")
    print("slot centres:", ref.slot_centers())


def write_slots(ref_json: Path, sop_path: Path, check_run: int, check_out: Path) -> None:
    """Slot centres from the reference → SOP layout; then draw them on a frame of another (rotated) run."""
    from sopvision.calibrate import calibration_targets, draw_layout_check, write_layout
    from sopvision.config import load_sop

    settings = Settings()
    ref = Reference(ref_json, settings.sop_data_root)
    coords = ref.slot_centers()
    write_layout(sop_path, coords)
    print("wrote", sop_path, coords)
    sop = load_sop(sop_path)
    run = {r.run: r for r in load_index(settings.sop_data_root)}[check_run]
    targets = calibration_targets(sop, run)
    target = targets[-2] if len(targets) > 1 else targets[-1]  # most parts in place, final step not yet done
    frame = read_frame(run.raw_path, target.t)
    h, inliers = ref.align(frame)
    if h is None:
        raise RuntimeError(f"can't align Install_{check_run} for the check ({inliers} inliers)")
    check_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(check_out), draw_layout_check(frame, transform_box(ref.chassis, h), sop))
    print("check image:", check_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--run", type=int, default=4)
    d.add_argument("--out", type=Path, default=Path("data/reference"))
    d.add_argument("--only", help="comma-separated classes to redraw, keeping the rest")
    sl = sub.add_parser("slots", help="write slot centres from the reference into the SOP YAML, then draw a check")
    sl.add_argument("--reference", type=Path, default=Path("data/reference/Install_4.json"))
    sl.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
    sl.add_argument("--check-run", type=int, default=9)
    sl.add_argument("--check-out", type=Path, default=Path("data/calibrate_check.jpg"))
    g = sub.add_parser("generate")
    g.add_argument("--reference", type=Path, default=Path("data/reference/Install_4.json"))
    g.add_argument("--frames", type=Path, default=Path("data/frames"))
    g.add_argument("--out", type=Path, default=Path("data/labels_auto"))
    g.add_argument("--sop", type=Path, default=Path("sops/server_fan.yaml"))
    args = parser.parse_args()
    if args.cmd == "draw":
        print("saved", draw(args.run, args.out, set(args.only.split(",")) if args.only else None))
    elif args.cmd == "slots":
        write_slots(args.reference, args.sop, args.check_run, args.check_out)
    else:
        from sopvision.config import load_sop

        generate(args.reference, args.frames, args.out, load_sop(args.sop).detector.classes)


if __name__ == "__main__":
    main()
