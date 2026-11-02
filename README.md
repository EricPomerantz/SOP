# SOP Vision

SOP Vision watches someone carry out a hands-on procedure through a camera and checks it against the standard
operating procedure (SOP) as it happens. It ticks off each step when its part is visibly in place, follows which step
is under way, and raises an alert straight away when something goes wrong: a step out of order, a part taken back
out, a bolt torqued in the wrong sequence, work that isn't part of the SOP, or a job left unfinished. Video stays on
the machine. Each run is logged and can be downloaded as a report.

New procedures are set up in the app itself. The **SOP builder** walks from a trained detector and a few recorded
runs to a published SOP with its own fine-tuned decision model, without editing code.

## What it does

- **Live monitoring.** Pick a camera (RTSP) or a recording and press Start. The dashboard shows:
  - the video with the work piece and every slot marked (installed, empty, covered by a hand);
  - the checklist, with the step under way and each step's duration;
  - the decision model's current judgment and how fast it answered;
  - a timeline and an alert log.
- **Mistake detection.** Order violations, parts removed after their step was done, an incomplete SOP at the end,
  off-SOP activity and anomalies. For torque work, it also catches a result that didn't reach torque, torque on an
  empty hole, a hole torqued twice, and a result that can't be placed on a hole.
- **Tool results.** A torque wrench (or any system) posts its results to the app, and each result is placed on the
  hole the wrench is on. Until a real wrench is connected, buttons on the dashboard stand in for it.
- **Recording.** The Record page captures new runs, with their torque results, into an SOP's dataset.
- **SOP builder.** Six stages:
  1. Detector: pick a model and give its classes roles.
  2. Recordings: record runs.
  3. Slots: suggest slots from the recordings.
  4. Steps: generate steps from the slots.
  5. Timelines: label when each step happened in the recordings.
  6. Decision model: train one, about an hour on the GPU.

  Then review and publish. See [docs/sop-builder.md](docs/sop-builder.md).
- **Evaluation.** Recordings with labelled step timelines are replayed frame by frame to score the step timing, the
  alerts and the decision model, so changes can be compared.

## How it works

```
camera / recording → YOLO detector → scene state → SOP tracker + decision model → dashboard, alerts, run log
```

1. A **YOLO detector**, trained per SOP, finds the work piece (the *anchor*), the parts, empty slots, tools and
   hands in each frame.
2. The **scene builder** turns detections into a stable scene state. Slots are positions relative to the anchor, so
   they follow it as it moves. The state records which slot holds which part, where the hands and tools are, and
   what just changed.
3. The **SOP tracker** applies the SOP's rules: a step is done when its part has been visibly in place for a moment.
   It also enforces the order (any order, or a fixed sequence) and raises the rule-based alerts.
4. A **decision model** reads a short text description of the scene, never the video. It names the step under way
   and judges off-SOP activity and anomalies. Three models can fill this role:
   - **Jev**, TypeSafe's hosted model.
   - **Laya**, a small local model that the builder fine-tunes per SOP.
   - **Decider**, an optional local model.

   All three get the identical request, so they can be swapped or compared side by side.
5. Everything streams to the React dashboard over a WebSocket and is written to a run log in `runs/`.

An SOP is a YAML file in `sops/`. It names the detector, the slots, the steps and their order, and the decision
thresholds; the builder writes these files for you. Diagrams and details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Included SOPs

| SOP | Procedure | Notes |
|---|---|---|
| `server_fan.yaml` | Install six fans, two power supplies and the cover on a 1U server (9 steps) | The first SOP; comes with two simulated-mistake videos |
| `lug_bolts.yaml` | Put five lug bolts into a wheel hub, then torque them in star sequence 1-3-5-2-4 (10 steps) | Torque results from the wrench, fine-tuned Laya. See [docs/lug-bolts-poc.md](docs/lug-bolts-poc.md) |
| `lug_bolts_new_test.yaml` | Coloured lug bolts into their matching holes, then torque in a star pattern | Built entirely in the SOP builder, reusing the lug-bolt recordings |

Their detectors, recordings and fine-tuned model download on first start (see Setup).

## Dashboard pages

| Page | What it is for |
|---|---|
| Live (`/`) | Run an SOP on a camera or recording and watch it being checked |
| Record (`#/record`) | Record new runs, with torque results, into an SOP's dataset |
| SOPs (`#/sops`) | The SOP builder: create, edit and publish SOPs, train their decision models |
| Latency (`#/latency`) | Measure how far a camera's frames trail reality, with a QR code |

To try it, follow the 5-minute walkthrough in [docs/demo-script.md](docs/demo-script.md): a clean run, a part taken
back out, an unfinished job, and the run report.

## Prerequisites

- Docker Engine with Compose v2.24+ and NVIDIA Container Toolkit on Linux, or Docker Desktop with the WSL2 backend and an NVIDIA driver on Windows
- An NVIDIA GPU for the app and model training; the default unit/frontend tests do not need one
- Bash and curl for `start.sh` / `stop.sh` (run them in WSL on Windows)
- A Hugging Face read token with access to the private demo-assets repo (`EricPo/sop-demo`)
- For decisions: a TypeSafe API key (Jev/compare mode), or the local Laya or Decider service, which needs no key
- On a network that intercepts TLS (e.g. the corporate network): its root CA files in `docker/certs/`, or `CORP_CA_DIR` in `.env` naming their folder. Image builds and Hugging Face downloads need them, and verification stays on

All application, training, labelling, evaluation, and development commands run in Docker. Host Python, uv, Node, npm, and virtual environments are not required. Python dependencies and the dashboard are built inside the images.

## Setup

```bash
git clone https://github.com/EricPomerantz/SOP.git && cd SOP
./start.sh --model laya --checkpoint laya-sop-v2-e12   # or ./start.sh to choose interactively
```

On a fresh clone, `./start.sh`:

1. Creates `.env` from `.env.example`.
2. Finds the demo assets missing and offers to download them, asking for `HF_TOKEN` if it isn't set.
3. Asks for `TYPESAFE_API_KEY` when the chosen model is `jev` or `compare`.
4. Builds the images and starts the app.

Tokens you enter are saved to `.env`, so you're asked only once.

The first start takes a while: it builds the app, worker and model images, and Laya downloads its base models into the `hf_cache` volume. If the app stops during startup, `start.sh` prints the reason (for example a missing key, no GPU, or missing weights) and exits.

**Demo assets.** Git doesn't carry the detectors, recordings or fine-tuned Laya (`models/`, `data/`, `datasets/` and `sop-sample-training-data/` are gitignored). They live in a private Hugging Face repo at the same paths, so downloading into the checkout puts them where the app looks. To manage them directly:

```bash
docker compose run --rm demo-assets fetch         # what the app needs to start + the lug-bolt test runs (~3.4 GB)
docker compose run --rm demo-assets fetch --all   # + every lug-bolt run (~6.7 GB)
docker compose run --rm demo-assets publish       # upload this machine's copies; needs a write token
```

- Fetching again downloads only what is missing or changed.
- Run `publish` again after retraining a detector or the Laya fine-tune, so other machines get the new version.
- `SOP_DEMO_REPO` names the repo (default `<HF user>/sop-demo`).
- On Linux with a user ID other than 1000, prefix the command with `HOST_USER=$(id -u):$(id -g)`. `start.sh` does this for you.
- The set is listed in `scripts/demo_assets.py`.

`.env` must be named exactly `.env`. `SOP_DATASET_DIR` selects the host dataset folder mounted at `/data`; timeline datasets and recordings use `./datasets`. A Vercel AI Gateway key also works with the two extra lines shown in `.env.example`.

## Run it

```bash
./start.sh                 # select the model, optional checkpoint, and Label Studio
./stop.sh                  # protects active runs and recordings
```

The dashboard is at http://localhost:8000. For non-interactive startup, use e.g. `./start.sh --model laya --checkpoint shipped --no-labelstudio`. The scripts already operate Docker. Their model/checkpoint selection applies to that start only and does not rewrite `.env`; only missing tokens are saved there. When using one-off commands below, set `DECISION_MODE`, `COMPARE_SHADOW`, and `LAYA_CHECKPOINT` consistently with the running services, or store the desired defaults in `.env`.

```bash
docker compose run --rm sopvision doctor
docker compose run --rm sopvision eval --out runs/eval.json
docker compose run --rm sopvision compare runs/before.json runs/eval.json
docker compose run --rm --build --entrypoint python tests scripts/docker_smoke.py
```

The smoke check is read-only and expects the app to be running. Video files, timeline recordings, and RTSP cameras are supported. Direct USB webcam input is no longer supported.

Models, SOPs, and scenario videos stay read-only in the running app. Recordings, run reports, and training outputs persist on the host through bind mounts. Do not remove the `hf_cache` or `labelstudio_data` volumes when stopping or rebuilding.

## Results: server-fan SOP

The first SOP's acceptance results, with Jev deciding. Evaluated on the two held-out recordings (Install_12, Install_13) and two simulated-mistake videos, in lockstep (`sopvision eval`):

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
sops/                 SOP definitions (detector, slots, steps, order rules, thresholds); sops/drafts/ holds builder drafts
src/sopvision/        sources · perception (YOLO) · scene · sop (tracker) · decide (Jev/Laya/Decider) · builder · jobs
                      · server (FastAPI) · storage · evaluation
web/                  React dashboard and SOP builder; web/e2e/ holds the browser tests
training/             frame sampling, reference labelling, detector dataset and training
scripts/              decision-model capture, training sets and benchmarks, torque simulator, demo assets
docker/               Laya and Decider service images
docs/                 architecture, SOP builder, lug-bolt POC, decision-model training, demo script
tests/                unit tests; tests/jev/ calls the real API
benchmarks/           frozen decision-model benchmark sets
```

Generated and downloaded data, not in git: `models/` (detectors), `datasets/` (timeline recordings),
`sop-sample-training-data/` (server-fan recordings), `data/` (scenarios, training sets, fine-tuned models, jobs),
`runs/` (run logs and reports).
