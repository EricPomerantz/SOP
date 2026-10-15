"""Sample training frames from the server_fan clips (spec §6.1 step 1).

Usage: docker compose run --rm tools -m training.sample_frames --out data/frames
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from sopvision.dataset import RunIndex, load_index
from sopvision.settings import Settings

PERCENTS = (10, 35, 65, 90)
BOUNDARY_OFFSET = 0.5


@dataclass(frozen=True)
class SampleSpec:
    run: int
    split: str
    position: int
    action: int
    t_in_clip: float
    clip_path: Path
    tag: str
    actions_by_position: tuple[int, ...] = ()

    @property
    def image(self) -> str:
        return f"{self.split}/Install_{self.run}_{self.position:02d}_{self.tag}.jpg"


def plan_samples(runs: list[RunIndex]) -> list[SampleSpec]:
    samples = []
    for run in runs:
        if run.split == "test":  # test runs are never sampled for training
            continue
        seq = tuple(run.actions_by_position)

        def spec(clip, t, tag):
            return SampleSpec(run.run, run.split, clip.position, clip.action, t, clip.path, tag, seq)

        for i, clip in enumerate(run.clips):
            samples += [spec(clip, clip.duration * p / 100, f"p{p}") for p in PERCENTS]
            if i + 1 < len(run.clips):
                nxt = run.clips[i + 1]
                samples.append(spec(clip, max(0.0, clip.duration - BOUNDARY_OFFSET), "bm"))
                samples.append(spec(nxt, min(BOUNDARY_OFFSET, nxt.duration), "bp"))
    return samples


def extract(samples: list[SampleSpec], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    by_clip: dict[Path, list[SampleSpec]] = defaultdict(list)
    for s in samples:
        by_clip[s.clip_path].append(s)
    rows = []
    for clip_path, clip_samples in by_clip.items():
        cap = cv2.VideoCapture(str(clip_path))
        try:
            for s in sorted(clip_samples, key=lambda s: s.t_in_clip):
                cap.set(cv2.CAP_PROP_POS_MSEC, s.t_in_clip * 1000)
                ok, image = cap.read()
                if not ok:
                    continue
                target = out_dir / s.image
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"image": s.image, "run": s.run, "split": s.split, "position": s.position,
                             "action": s.action, "actions_by_position": list(s.actions_by_position)})
        finally:
            cap.release()
    rows.sort(key=lambda r: (r["run"], r["position"], r["image"]))
    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/frames"))
    args = parser.parse_args()
    settings = Settings()
    settings.check_data_root()
    samples = plan_samples(load_index(settings.sop_data_root))
    manifest = extract(samples, args.out)
    n = sum(1 for _ in manifest.open(encoding="utf-8"))
    print(f"wrote {n} frames; manifest {manifest}")


if __name__ == "__main__":
    main()
