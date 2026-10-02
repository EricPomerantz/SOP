# SOP Vision

Live SOP monitoring: a custom YOLO detector turns camera frames into a JSON scene state ("fan bay 3: installed, hand near PSU bay 1"), **Jev** (TypeSafe AI's System One decision model) decides which SOP step is happening and whether it's done, and a web dashboard shows the live checklist, Jev's judgment, a timeline and alerts. The first SOP is a server fan / power-supply / cover install.

- Design: [docs/superpowers/specs/2026-10-01-sop-vision-design.md](docs/superpowers/specs/2026-10-01-sop-vision-design.md)
- Build plan: [docs/superpowers/plans/2026-10-01-sop-vision-phase1.md](docs/superpowers/plans/2026-10-01-sop-vision-phase1.md)
- Findings and numbers: [docs/superpowers/notes/phase0.md](docs/superpowers/notes/phase0.md)
- Demo walkthrough: [docs/demo-script.md](docs/demo-script.md)

## Results

Evaluated on the two held-out recordings (Install_12, Install_13) and two simulated-mistake videos, in lockstep (`sopvision eval`):

| Acceptance criterion | Result |
|---|---|
| All 18 steps detected, in order | ✓ 18/18 |
| Step completion timing | ✓ median 1.13 s, worst 1.44 s after the true step boundary |
| Off-SOP segments handled (no step progress, Jev says idle/off-SOP) | ✓ 4/4 |
| No false violations on clean runs | ✓ |
| Simulated mistakes raise their alerts | ✓ part removed · SOP incomplete |
| Dashboard video ≥ 20 fps | ✓ 23.5 fps (pipeline 29.8 fps on an RTX 4080) |
| Docker gives the same results as native | ✓ `sopvision compare` → parity: OK |

Detector (YOLO26m): mAP50 0.994, mAP50-95 0.975 on the validation recording. Jev answers in ~180–195 ms (median).

## Prerequisites

- Windows 11 with an NVIDIA GPU (developed on an RTX 4080) and a current NVIDIA driver
- [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`) — it installs Python 3.12 for the project
- Node.js 20+ (only to rebuild the dashboard)
- Docker Desktop with the WSL2 backend (only for the container path)
- A TypeSafe API key for Jev, and the `server_fan` dataset in `sop-sample-training-data/`

## Setup

```bash
uv sync
copy .env.example .env      # then put your key after TYPESAFE_API_KEY=
cd web && npm ci && npm run build && cd ..
```

`.env` must be named exactly `.env` (Windows Explorer tends to create `env.env` or `.env.txt`). A Vercel AI Gateway key also works with the two extra lines shown in `.env.example`.

## Run it

**Natively (primary demo path)**

```bash
uv run sopvision serve          # dashboard on http://localhost:8000
uv run sopvision eval           # replays the test runs + scenarios, checks acceptance → runs/eval-native.json
```

**In Docker (second, equivalent path)**

```bash
docker compose up --build                                     # dashboard on http://localhost:8000
docker compose run --rm sopvision eval --out runs/eval-docker.json
uv run sopvision compare runs/eval-native.json runs/eval-docker.json
uv run python scripts/docker_smoke.py                         # image builds, sees the GPU, serves the API
```

The container supports recorded videos and RTSP cameras; the USB webcam works natively only.

## Tests

```bash
uv run pytest                   # unit tests (no network, no GPU)
uv run pytest -m gpu            # real detector weights
uv run pytest -m jev            # real Jev API: contract + violation scenarios
cd web && npx vitest run        # dashboard
```

## Retrain or add equipment

Labels come from three reference frames you box once per setup (zero-shot labellers couldn't find the server parts — see the notes):

```bash
uv run python -m training.sample_frames --out data/frames          # frames spread across every step
uv run python -m training.reference_labels draw --run 4            # box chassis/bays/parts on 3 frames (a window opens)
uv run --group training python -m training.reference_labels generate   # align every frame, label from the step timeline
uv run python -m training.review_sheets                            # contact sheets in data/labels_review/
uv run python -m training.yolo_dataset                             # data/yolo
uv run python -m training.train train                              # models/server_fan/best.pt
uv run python -m training.reference_labels slots                   # write slot centres into sops/server_fan.yaml
uv run sopvision make-scenarios --run 12                           # simulated-mistake videos
```

A new SOP is a new YAML file in `sops/` plus a detector trained the same way.

## Layout

```
sops/                 SOP definitions (steps, slots, order rules, thresholds)
src/sopvision/        sources · perception · scene · decide (Jev) · sop (tracker) · storage · server · evaluation
training/             frame sampling, reference labelling, dataset, training
web/                  React dashboard
tests/                unit tests; tests/jev/ calls the real API
```
