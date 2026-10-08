# SOP Vision

Live SOP monitoring: a custom YOLO detector turns camera frames into a JSON scene state ("fan bay 3: installed, hand near PSU bay 1"). **Jev** (TypeSafe AI's System One decision model) reads that state, names the SOP step under way, and can raise off-SOP and anomaly warnings. A step is ticked off once its part is visibly in place: either Jev holds "done" while the part is visible, or the part stays visible for 1 s. On the test footage the 1 s visual check always got there first (see Results). A web dashboard shows the live checklist, Jev's judgment, a timeline and alerts. The first SOP is a server fan / power-supply / cover install.

- Architecture and flow diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
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
| Off-SOP segments handled (no step progress; any Jev answer is idle/off-SOP) | ✓ 4/4 (Jev answered 0 times in them; see below) |
| No false violations on clean runs | ✓ |
| Simulated mistakes raise their alerts | ✓ part removed · SOP incomplete |
| Dashboard video ≥ 20 fps | ✓ 23.5 fps (pipeline 29.8 fps on an RTX 4080) |
| Docker gives the same results as native | ✓ `sopvision compare` → parity: OK |

Detector (YOLO26m): mAP50 0.994, mAP50-95 0.975 on the validation recording. Jev answers in ~170–195 ms (median).

**What Jev did in these results:**
- **Step confirmation:** all 18 test-run steps were ticked off by visual evidence and none by Jev. For a just-placed part, Jev's "done" confidence sits around the 0.8 threshold, so the 1 s visual check gets there first.
- **Step tracking:** Jev's real job here is naming the step under way, which drives the active step on the dashboard.
- **Off-SOP stretches:** in the test recordings these are an empty bench with nothing changing, so Jev wasn't asked anything during them (0 answers in each of the 4 windows). Criterion 3 passes on "no step progressed" alone.
- **Unverified:** the off-SOP and anomaly warnings work in unit tests with scripted Jev answers, but haven't been tested with real Jev on real footage. That needs a recording with off-SOP work in it.

`sopvision eval` prints these counts with every run (`test-run completions: …`, `off-SOP windows: …`).

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
docker compose --profile test run --rm tests              # unit tests in Docker
docker compose --profile test run --rm tests -m decider   # real local Decider (service must be up)
cd web && npx vitest run        # dashboard
```

## Local Decider (Jev comparison)

[Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) is an open model that serves Jev's own wire format (`POST /v1/systemone`), so the app sends it **byte-identical** requests through the same client. `DECISION_MODE` in `.env` picks who decides:

| `DECISION_MODE` | Decides | Also | Needs |
|---|---|---|---|
| `jev` (default) | Jev | – | `TYPESAFE_API_KEY` |
| `decider` | local Decider | – | the Decider service ready (checked at startup) |
| `compare` | Jev | Decider gets the same request at the same moment; both answers are logged | the key; Decider optional (its errors are logged, Jev is unaffected) |

```bash
docker compose --profile decider up -d decider        # first start downloads 8.4 GB, then loads (~1-2 min)
docker compose run --rm sopvision doctor              # decider: running / model_loaded / gpu / inference
docker compose --profile test run --rm tests -m decider   # smoke test against the real model
DECISION_MODE=compare docker compose up sopvision     # or set it in .env
```

**Where the model lives.** The `decider` service (`docker/decider.Dockerfile`) runs the model's own server from the downloaded snapshot. Weights go to the `hf_cache` Docker volume (`/cache/huggingface`), never into git or the image. The revision is pinned to a commit SHA (`DECIDER_REVISION`, default v2.1 `eb5fbdfc…`), so an upstream push can't change results. On first start it downloads; on later starts it finds the snapshot and makes no network calls. Compiled Triton kernels are cached in the same volume. Behind a TLS-intercepting proxy, set `CORP_CA_DIR` to the folder of your corporate root CA files so the build trusts them (verification stays on).

**Readiness.** The service's `/health` reports ok only after the weights are loaded and CUDA graphs captured; the app additionally sends one real decision and checks the probabilities and that it runs on CUDA (`sopvision.decide.providers.decider_readiness`).

**What gets logged.** Every `decision` event now has `provider`, `request_ms` (wall time of the call, retries included) and, live, `e2e_ms` (from the moment the scene had enough context for a call to the tracker applying the answer). In `compare` mode each request also writes one `decision_comparison` event: the exact state and questions, both providers' answers, errors and latencies, how far they differ (raw, and at the 0.8 acting threshold), `wait_ms` (context-ready to request sent) and a `ground_truth` slot.

**Benchmark.** Agreement is not accuracy, so the benchmark scores each provider against the recordings' step segments where the answer is known:

```bash
DECISION_MODE=compare docker compose run --rm sopvision eval --out runs/eval-compare.json   # also writes runs/eval-compare.<video>.jsonl
docker compose run --rm --entrypoint python sopvision scripts/bench_decisions.py runs/eval-compare.test-*.jsonl
#   --no-replay scores the recorded answers instead of calling both models again
```

It reports per provider: step accuracy, off-SOP false positive/negative rates, anomaly false positives on the clean runs, mean confidence and confidence when wrong, and request latency p50/p95/max (plus end to end for live logs). It also reports how often the two agree.

**Vision model (not used for decisions yet).** `Mapika/decider-2b-vision` is cached in the same volume and smoke-tested on its own (image + text context + bounded options → probabilities):

```bash
docker compose --profile decider stop decider     # free the GPU for it
docker compose --profile decider run --rm decider python scripts/decider_vision_smoke.py
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
