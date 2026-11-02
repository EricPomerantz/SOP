# SOP Vision

Live SOP monitoring: a custom YOLO detector turns camera frames into a JSON scene state ("fan bay 3: installed, hand near PSU bay 1"). **Jev** (TypeSafe AI's System One decision model) reads that state, names the SOP step under way, and can raise off-SOP and anomaly warnings. A step is ticked off once its part is visibly in place: either Jev holds "done" while the part is visible, or the part stays visible for 1 s. On the test footage the 1 s visual check always got there first (see Results). A web dashboard shows the live checklist, Jev's judgment, a timeline and alerts. The first SOP is a server fan / power-supply / cover install.

- Architecture and flow diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Historical design: [docs/superpowers/specs/2026-10-01-sop-vision-design.md](docs/superpowers/specs/2026-10-01-sop-vision-design.md)
- Historical build plan: [docs/superpowers/plans/2026-10-01-sop-vision-phase1.md](docs/superpowers/plans/2026-10-01-sop-vision-phase1.md)
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

Detector (YOLO26m): mAP50 0.994, mAP50-95 0.975 on the validation recording. Jev answers in ~170–195 ms (median).

**What Jev did in these results:**
- **Step confirmation:** all 18 test-run steps were ticked off by visual evidence and none by Jev. For a just-placed part, Jev's "done" confidence sits around the 0.8 threshold, so the 1 s visual check gets there first.
- **Step tracking:** Jev's real job here is naming the step under way, which drives the active step on the dashboard.
- **Off-SOP stretches:** in the test recordings these are an empty bench with nothing changing, so Jev wasn't asked anything during them (0 answers in each of the 4 windows). Criterion 3 passes on "no step progressed" alone.
- **Unverified:** the off-SOP and anomaly warnings work in unit tests with scripted Jev answers, but haven't been tested with real Jev on real footage. That needs a recording with off-SOP work in it.

`sopvision eval` prints these counts with every run (`test-run completions: …`, `off-SOP windows: …`).

## Prerequisites

- Docker Engine with Compose v2.24+ and NVIDIA Container Toolkit on Linux, or Docker Desktop with the WSL2 backend and an NVIDIA driver on Windows
- An NVIDIA GPU for the app and model training; the default unit/frontend tests do not need one
- Bash and curl for `start.sh` / `stop.sh` (run them in WSL on Windows)
- The `server_fan` dataset in `sop-sample-training-data/`, plus the detector weights referenced by each SOP
- A TypeSafe API key for Jev/compare mode, or the optional Decider/Laya service for local decisions

All application, training, labelling, evaluation, and development commands run in Docker. Host Python, uv, Node, npm, and virtual environments are not required. Python dependencies and the dashboard are built inside the images.

## Setup

```bash
cp -n .env.example .env     # preserves an existing .env; configure the decision mode and any required key
docker compose build sopvision
```

**Demo assets.** Git doesn't carry the detectors, recordings or fine-tuned Laya. They live in a private Hugging Face repo at the same paths. On a fresh clone, `./start.sh` creates `.env` from the example, offers to download the assets, and asks for a Hugging Face read token (`HF_TOKEN`). For `jev`/`compare` it also asks for `TYPESAFE_API_KEY`. Both are saved to `.env`. You can also fetch the assets yourself:

```bash
docker compose run --rm demo-assets fetch         # app essentials + lug-bolt test runs (~3.4 GB)
docker compose run --rm demo-assets fetch --all   # + every lug-bolt run (~6.7 GB)
docker compose run --rm demo-assets publish       # from the machine that has them; needs a write token
```

`SOP_DEMO_REPO` names the repo (default `<HF user>/sop-demo`). On Linux with a user ID other than 1000, prefix the command with `HOST_USER=$(id -u):$(id -g)`. The set is listed in `scripts/demo_assets.py`.

`.env` must be named exactly `.env`. `SOP_DATASET_DIR` selects the host dataset folder mounted at `/data`; timeline datasets and recordings use `./datasets`. A Vercel AI Gateway key also works with the two extra lines shown in `.env.example`.

## Run it

```bash
./start.sh                 # select the model, optional checkpoint, and Label Studio
./stop.sh                  # protects active runs and recordings
```

The dashboard is at http://localhost:8000. For non-interactive startup, use e.g. `./start.sh --model laya --checkpoint shipped --no-labelstudio`. The scripts already operate Docker; their model/checkpoint selection does not rewrite `.env`. When using one-off commands below, set `DECISION_MODE`, `COMPARE_SHADOW`, and `LAYA_CHECKPOINT` consistently with the running services, or store the desired defaults in `.env`.

```bash
docker compose run --rm sopvision doctor
docker compose run --rm sopvision eval --out runs/eval.json
docker compose run --rm sopvision compare runs/before.json runs/eval.json
docker compose run --rm --build --entrypoint python tests scripts/docker_smoke.py
```

The smoke check is read-only and expects the app to be running. Video files, timeline recordings, and RTSP cameras are supported. Direct USB webcam input is no longer supported.

Models, SOPs, and scenario videos stay read-only in the running app. Recordings, run reports, and training outputs persist on the host through bind mounts. Do not remove the `hf_cache` or `labelstudio_data` volumes when stopping or rebuilding.

## Tests

```bash
docker compose run --rm --build tests                     # unit tests; no API or GPU required
docker compose run --rm --build web-tests                 # dashboard tests
docker compose run --rm web-tests npm run build           # TypeScript and production build
docker compose run --rm gpu-tests -m gpu                   # real detector weights and GPU
docker compose run --rm gpu-tests -m jev                   # real Jev API; requires the key
docker compose run --rm tests -m decider                   # real Decider; its service must be running
```

Directly naming a service enables its Compose profile. `gpu-tests` uses the same test image as `tests`, with GPU access and the optional `.env` credentials.

For frontend development, start the app first, then run `docker compose up -d --build web-dev` and open http://localhost:5173. Source and configuration changes are mounted read-only; dependencies are installed in the image and the Vite cache stays in the container. HTTP and WebSocket traffic is forwarded to the app container. Rebuild with the same command after changing frontend dependencies. No host `node_modules` directory or dependency volume is needed. Stop the development server with `docker compose stop web-dev`.

## Local Decider (Jev comparison)

[Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) is an open model that serves Jev's own wire format (`POST /v1/systemone`), so the app sends it **byte-identical** requests through the same client. `DECISION_MODE` in `.env` picks who decides:

| `DECISION_MODE` | Decides | Also | Needs |
|---|---|---|---|
| `jev` (default) | Jev | – | `TYPESAFE_API_KEY` |
| `decider` | local Decider | – | the Decider service ready (checked at startup) |
| `laya` | local Laya | - | the Laya service ready; optional `LAYA_CHECKPOINT` |
| `compare` | Jev | `COMPARE_SHADOW=decider` or `laya` gets the same request; both answers are logged | the key; shadow optional (its errors are logged, Jev is unaffected) |

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

The `tools` image includes the training dependency group and writable mounts for generated data, models, SOP calibration, and timeline installation. Use `trainer` for detector training, or override its entrypoint with `--entrypoint python` for other GPU tools. These services share an image and build it explicitly; no host environment is used.

Server-fan labels come from three reference frames annotated in browser-based Label Studio:

```bash
docker compose build tools
docker compose run --rm tools -m training.sample_frames --out data/frames
docker compose --profile labelling up -d labelstudio
docker compose run --rm tools -m training.reference_labels config
docker compose run --rm tools -m training.reference_labels export-tasks --run 4
```

In Label Studio at http://localhost:8080, create a reference project with the printed XML and import `data/reference/labelstudio/tasks.json`. Annotate the three frames, export the reviewed tasks as JSON to `data/reference/annotations.json`, then run:

```bash
docker compose run --rm tools -m training.reference_labels import-annotations --export data/reference/annotations.json
docker compose run --rm --entrypoint python trainer -m training.reference_labels generate
docker compose run --rm tools -m training.review_sheets
docker compose run --rm tools -m training.yolo_dataset
docker compose run --rm trainer train
docker compose run --rm tools -m training.reference_labels slots
docker compose run --rm --entrypoint sopvision tools make-scenarios --run 12
docker compose run --rm --entrypoint sopvision trainer calibrate
```

Existing reference boxes are exported as predictions. `export-tasks --only psu_bay_empty,psu` limits a review to those classes; importing preserves the rest. Only submitted, non-cancelled annotations are imported. Reviewed empty annotations clear the selected classes, while unreviewed tasks leave existing boxes intact. Reference JSON and generated labels keep their existing formats.

The detector-label review workflow also uses `docker compose run --rm tools -m training.labelstudio config|export|import` (choose one subcommand). Other scripts run as `docker compose run --rm tools scripts/<script>.py`; the writable mounts preserve their outputs. For the torque simulator, use `docker compose run --rm tools scripts/simulate_torque.py` against the running app.

A new SOP is a new YAML file in `sops/` plus its detector. See [the lug-bolts workflow](docs/lug-bolts-poc.md) for timeline recording/annotation and [decision-model training](docs/decision-model-training.md) for capture, dataset building, Laya fine-tuning, and checkpoint evaluation. Old native deployment notes under `docs/superpowers/` are historical, not operating instructions.

## Layout

```
sops/                 SOP definitions (steps, slots, order rules, thresholds)
src/sopvision/        sources · perception · scene · decide (Jev) · sop (tracker) · storage · server · evaluation
training/             frame sampling, reference labelling, dataset, training
web/                  React dashboard
tests/                unit tests; tests/jev/ calls the real API
```
