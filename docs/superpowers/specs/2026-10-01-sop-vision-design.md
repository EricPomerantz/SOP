# SOP Vision — Live SOP Monitoring with Object Detection + Jev

**Date:** 2026-10-01
**Status:** Design approved in conversation; awaiting written-spec review
**Author:** Eric Pomerantz

## 1. Summary

A configurable system that watches a workstation through a camera, detects the parts and hands involved in a Standard Operating Procedure (SOP), and tracks — live — which step is being performed, which steps are complete, and whether the procedure is being followed.

Perception is a custom-trained YOLO detector. Decisions ("which step is the operator doing?", "is this step done?", "is something off?") are made by **Jev**, TypeSafe AI's System One decision model, from a structured JSON description of the scene. A web dashboard shows the annotated video, a live checklist, Jev's current judgment, a timeline, and an event log.

The first SOP is the **server fan install** dataset already in this folder. The system is built so that a new SOP (for other equipment) is a new YAML file plus a newly trained detector — no new code.

It runs two equivalent ways: natively with `uv` (the primary demo path) and in a GPU-enabled Docker container (the path that later retargets to an on-prem Linux server).

## 2. Goals, non-goals, success criteria

### Goals
- **Phase 1 (this spec's implementation plan):** a quick, good-looking demo for management, running on this PC, replaying the server_fan recordings as a live feed.
- **Headroom for production:** every Phase 1 shortcut sits behind a stable interface so Phase 2 can swap implementations (file → live camera, JSONL → database, single → multi-station) without rewrites.
- **Generic:** SOPs are defined in YAML; Jev questions are generated from the config.
- **Two run paths in Phase 1:** native `uv` (primary) and Docker with NVIDIA GPU via WSL2 (second, equivalent; §13.2). Phase 2 retargets the same image to a Linux on-prem GPU server.

### Non-goals (Phase 1)
- Authentication, multi-station, more than one concurrent run.
- Deployment to machines other than this PC (the on-prem server is Phase 2).
- USB webcam inside Docker (native only; §13.2).
- Real footage of order violations or skipped steps (none exists; covered by synthetic scenario tests — see §12).
- Operator-facing alarms (sound, andon light), MES integration.

### Phase 1 acceptance criteria
Measured by the eval harness on the held-out test runs **Install_12** and **Install_13**:
1. All 9 SOP steps are marked done, in order, on both runs (18/18).
2. Step completion time error (tracker "done" time vs. ground-truth end of that step's clip): **median ≤ 2 s, max ≤ 4 s** across the 18 steps.
3. All 4 off-SOP segments (2 per run: before step 1 and after the cover) are handled cleanly: no SOP step progresses during them and every Jev answer there is idle or off-SOP. *(Revised 2026-10-02 with the user: in this dataset those segments are an empty bench or hands withdrawn, which the system correctly calls idle. Tolerances: the step that just ended may complete inside the segment — that's scored by criterion 2 — and Jev answers in the segment's first second are skipped.)*
4. Zero violation-level alerts (`order_violation`, `part_removed`, `sop_incomplete`) on these clean runs. Warning-level alerts (`off_sop` outside the off-SOP segments, `anomaly`) are reported but not gated.
5. Simulated scenarios raise their expected alert before the scenario video ends (§10): *incomplete run* → `sop_incomplete`; *fan removed* → `part_removed` for fan_3.
6. Dashboard runs smoothly at ≥ 20 fps video with Jev decisions displayed live.
7. **Docker parity:** `sopvision eval` run in the container gives the same results as the native run on the same test and scenario videos, checked by `sopvision compare`:
   - both runs pass criteria 1–5;
   - identical step-completion order and identical set of alert types per video;
   - each step's completion time within ±0.5 s of the native run;
   - slot states agree on ≥ 99 % of frames.

   The tolerances absorb run-to-run numeric variation in Jev and the GPU; eval's lockstep clock (§12) removes differences due to machine speed.

## 3. Context

### 3.1 Jev (TypeSafe AI)
Facts gathered from TypeSafe's launch post, Cloudflare's model page, and a third-party SDK guide (released 2026-09-15, early access; the user has an API key):

- **Input:** a `state` (text, JSON object, or text array) plus typed `questions`. **No images, audio, or video** — hence the detector → JSON → Jev design.
- **Question types:** `Choice` (one of up to 255 labelled options; returns choice, probabilities, confidence), `Score` (2–10 level scale), `Noul` (yes/no as a probability 0–1).
- **Latency:** 70–500 ms end to end. **Rate limits:** 1,200 requests/min, 250k tokens/s. **Context:** 32k tokens for state + longest question.
- **Price:** $0.042 per million input tokens; output free.
- **SDK (Python):** `pip install typesafe-sdk`; `TypeSafeClient().system_one(state=..., questions=...)`; reads `TYPESAFE_API_KEY`.

The SDK details come from a third-party guide for a two-week-old product; **Phase 0 verifies them** against the official docs and a live call.

### 3.2 The server_fan dataset
Location: `sop-sample-training-data/server_fan/` (5.5 GB extracted; the 5.8 GB `.tar.gz` sits beside it).

- **SOP:** 10 actions — fans 1–6 (connect connector, press in place), PSUs 1–2 (place in compartment, press until latch clicks), cover (place on top, press black latch), and (10) "action not belonging to the SOP".
- **Recordings:** 12 runs (`raw/Install_{1,3..13}.MP4`; there is no Install_2), 1920×1080 @ 30 fps, 62–134 s each, several different operators.
- **Splits:** `train/` = Install_1, 3–11; `test/` = Install_12, 13. Each run is cut into 11 clips.
- **Clip naming (verified on all 12 runs):** `{action:02d}_Install_{run}_{occurrence}_{position}.mp4`. Every run has the same sequence by position: off-SOP, fans 1–6, PSU 1–2, cover, off-SOP.
- **Clips are contiguous:** clip durations sum exactly to the raw video for 10 runs, including both test runs, so ground-truth step boundaries = cumulative clip durations. Install_3 and Install_5 are each ~2.2 s short and are not used for timing evaluation.
- **Labels are temporal only** (which step each clip is). No bounding boxes. The `augmented/` folder holds LLaVA-style video QA data (`bcq`, `mcq`, `gqas`, `golden_gqa`) — not used by this design.
- **Scene:** fixed overhead camera, top-down view of a 1U server: 6 fan bays in two rows of 3 in the middle, 2 PSU bays on the left, cover placed last. The chassis position and rotation vary between runs (Install_1 and Install_9 are visibly rotated), so slot positions must be computed relative to the detected chassis, not fixed pixel regions.
- **All runs are correct runs** — no order violations, skips, or removals.

### 3.3 Machine
Windows 11, i7-13700K, RTX 4080 (16 GB, NVIDIA driver 591.86 with WSL GPU support), Python 3.14 (system) and 3.11, Node 25, git 2.52. Docker Desktop 29.1.3 with the WSL2 backend and Compose v5 is installed (the engine was stopped at the time of checking). No webcam currently attached. Ultralytics is licensed by the company (enterprise licence), so AGPL constraints do not apply.

## 4. Architecture

```
 VideoSource ──frames──▶ Detector ──detections──▶ SceneBuilder ──SceneState + SceneEvents──▶ JevDecider ──Decisions──▶ SopTracker
 (webcam/RTSP/file)      (YOLO + ByteTrack)       (facts: slot states,                     (typed questions,        (checklist, order rules,
                                                    hands, held parts)                       async, rate-limited)     timing, alerts)
                                │                         │                                                                │
                                └── annotated frames ─────┴──────────────▶ Dashboard server (FastAPI) ◀────────────────────┘
                                                                           MJPEG video · WebSocket status · REST
```

Three kinds of work, kept separate:
- **Facts (deterministic code):** `SceneBuilder` turns detections into a stable `SceneState`.
- **Judgments (Jev):** `JevDecider` asks config-generated questions about the `SceneState`.
- **Bookkeeping (deterministic code):** `SopTracker` applies judgments with thresholds and evidence checks, enforces order, times steps, raises alerts.

### 4.1 Components and interfaces

| Component | Responsibility | Input → Output | Phase 2 swap |
|---|---|---|---|
| `VideoSource` | Produce timestamped frames | config → `Frame(t, image)` | add sources; no consumer change |
| `Detector` | Detect objects per frame | `Frame` → `list[Detection(cls, box, conf, track_id)]` | retrained / different weights |
| `SceneBuilder` | Map detections to slots, smooth, emit changes | detections → `SceneState`, `list[SceneEvent]` | OBB / homography anchoring if rigs rotate more |
| `JevDecider` | Build state window + questions, call Jev | `SceneState` + tracker status → `Decision` | — |
| `SopTracker` | Step status machine, order rules, alerts | `Decision` + `SceneState` → `StepUpdate`, `Alert` | — |
| `RunLog` | Persist every event | events → `runs/<run_id>.jsonl` | SQLite / Postgres, same event schema |
| `Server` | Video stream, status stream, control API | all of the above → HTTP/WS | WebRTC, auth, multi-station |

All interfaces are Python `Protocol`s or pydantic models in `src/sopvision/`; no component imports another's internals.

### 4.2 Concurrency
- **Pipeline thread:** capture → detect → `SceneBuilder` → draw overlays, at frame rate. In live mode it always processes the **newest** frame (drops stale ones; no queue build-up). Eval uses a lockstep mode instead (§12).
- **Clock:** every timer (smoothing, debounce, heartbeat, hold times) reads an injected clock — wall time when live, video time in eval.
- **asyncio loop (FastAPI):** receives `SceneState`/events from the pipeline thread via `loop.call_soon_threadsafe` into an `asyncio.Queue`; runs `JevDecider` calls, `SopTracker`, `RunLog`, and WebSocket broadcast. Video never waits on Jev.

## 5. SOP configuration

One YAML file per SOP, validated by pydantic at load time. Example (trimmed):

```yaml
sop: server_fan_install
version: 1
description: Install six fans, two power supplies and the cover on a 1U server.

detector:
  weights: models/server_fan/best.pt
  classes: [chassis, fan, fan_bay_empty, psu, psu_bay_empty, cover, hand]
  autolabel_prompts:               # used only by training/autolabel
    chassis: "1U server chassis seen from above"
    fan: "black server fan module"
    fan_bay_empty: "empty fan slot"
    psu: "power supply unit"
    psu_bay_empty: "empty power supply bay"
    cover: "metal server top cover"
    hand: "hand"

layout:
  anchor: chassis                  # slot coords are normalised (0–1) within the chassis box
  match_radius: 0.08               # max normalised distance for a detection to match a slot
  slots:
    fan_bay_1: {at: [0.47, 0.78], filled_by: fan, empty_as: fan_bay_empty}
    # fan_bay_2 … fan_bay_6, psu_bay_1, psu_bay_2 (values set by the calibrate tool)
    cover:     {at: [0.50, 0.50], filled_by: cover, match_radius: 0.25}

steps:
  - id: fan_1
    text: Install the first fan by connecting the connector, then pressing the fan in place.
    evidence: {fan_bay_1: installed}
  # fan_2 … fan_6, psu_1, psu_2
  - id: cover
    text: Place the cover on top of the server and press the black latch to lock it.
    evidence: {cover: installed}
    hands_off: true                # the latch isn't visible: evidence counts only once hands leave the cover

order: strict                      # strict | any | groups
# groups: [[fan_1, fan_2, fan_3, fan_4, fan_5, fan_6], [psu_1, psu_2], [cover]]   # when order: groups

off_sop: Any action that is not one of the steps above.

decisions:
  accept_confidence: 0.8           # Jev confidence/probability needed to act
  hold_seconds: 1.0                # how long a judgment must persist
  evidence_only_after: 1.0         # mark done on evidence alone if Jev hasn't confirmed by then (tuned from 3.0 in eval)
  hands_off_seconds: 0.5           # hands_off steps: part in place and untouched this long
```

**Validation rules (fail fast at startup):** every `filled_by`/`empty_as` is a detector class; every `evidence` key is a declared slot; every value is `installed` or `empty`; step ids are unique; `groups` (if used) cover every step exactly once; slot coordinates are in [0, 1].

## 6. Perception and training

### 6.1 Training workflow (`training/`)
1. **Sample frames** — from the 9 training runs (Install_1, 3–10): 4 frames per clip at 10/35/65/90 % of its duration, plus frames at ±0.5 s around each clip boundary (≈ 550 frames). **Install_11 = validation. Install_12/13 are never sampled.**
2. **Auto-label** — a zero-shot model labels each frame using `autolabel_prompts`. Phase 0 compares **SAM 3** and **Grounding DINO** on ~20 frames and keeps the better one. Output: YOLO-format boxes.
3. **Count check** — from the clip's position, the expected number of installed fans, empty fan bays, PSUs, empty PSU bays and cover presence is known (the step in progress may be either value; hand occlusion may hide items). Frames whose labels disagree are flagged and placed **first** in the review queue.
4. **Review** — import frames + pre-annotations into **Label Studio** (local); fix flagged frames, skim the rest (~1 h). Export YOLO format.
5. **Train** — latest Ultralytics YOLO detection model (size `s` or `m`; pick `m` if the full pipeline still sustains ≥ 30 fps), `imgsz=1280`, on the RTX 4080. Output `models/server_fan/best.pt` + Ultralytics metrics report (mAP on Install_11 frames).

**Fallback:** if Phase 0 shows auto-labels are poor (< 50 % of sampled frames pass the count check), hand-label ~200 frames instead; the rest of the workflow is unchanged.

### 6.2 Runtime perception
- YOLO + ByteTrack (Ultralytics `track`), up to 30 fps.
- **Chassis anchor:** highest-confidence `chassis` box, smoothed with an exponential moving average (the chassis rarely moves). Not seen for 3 s → `chassis_lost` event + alert; slot mapping pauses.
- **Slot matching:** each `filled_by`/`empty_as` detection is converted to chassis-normalised coordinates and matched to the nearest slot centre within `match_radius` (one-to-one, Hungarian assignment).
- **Slot state** ∈ `installed | empty | occluded | unknown`:
  - `installed` / `empty` are **observed**: a `filled_by` / `empty_as` detection matches the slot, and the reading has agreed for 0.5 s. Slots without `empty_as` (e.g. `cover`) read `empty` when no `filled_by` detection matches and no hand overlaps.
  - `occluded`: nothing observed and a hand box overlaps the slot. Keeps `last`.
  - `unknown`: nothing observed and no hand overlaps (e.g. fan bays hidden under the installed cover, or the chassis not visible). Keeps `last`.
  - Evidence checks use the observed state, or `last` when `occluded`/`unknown` — a hand or the cover never makes an installed part read as missing.
- **Hands:** each `hand` box → `near: <slot>` (nearest slot centre within 2× `match_radius`), else `near: chassis` or `outside`.
- **Held parts:** a `fan`/`psu`/`cover` detection that matches no slot and overlaps or touches a hand box → `held: [<class>]`.
- **Calibrate tool:** `sopvision calibrate --sop sops/server_fan.yaml --video <file>` shows a frame with the detected chassis; the user clicks each slot centre in order; normalised coordinates are written back to the YAML.

### 6.3 SceneState contract
```json
{
  "version": 812,
  "t": 41.2,
  "chassis": {"visible": true, "conf": 0.99},
  "slots": {
    "fan_bay_1": {"state": "installed", "conf": 0.97},
    "fan_bay_3": {"state": "occluded", "last": "empty"}
  },
  "hands": [{"near": "fan_bay_3"}],
  "held": ["fan"]
}
```
`SceneEvent` types: `slot_changed(slot, from, to)`, `hand_entered(zone)`, `hand_left(zone)`, `held_changed(items)`, `chassis_lost`, `chassis_found`. `version` increments on every emitted `SceneState`.

## 7. Jev decisions (`decide/`)

### 7.1 When to call
- On any `SceneEvent`, debounced 250 ms (≤ 4 calls/s).
- Heartbeat every 2 s while any hand is visible.
- No calls while nothing changes and no hands are visible.

At ~1,000 tokens per call this is far below rate limits and costs well under $0.001 per station-hour.

### 7.2 What is sent
`state` = SOP description, step list with current tracker status, the current `SceneState`, and the last 10 `SceneEvent`s with timestamps (rolling window — never full history). **Only this JSON leaves the machine; video frames never do.**

### 7.3 Questions (generated from config)
| Key | Type | Text | Options |
|---|---|---|---|
| `current_step` | `Choice` | "Which step is the operator performing right now?" | every step id → its `text`; `idle` → "No one is working on the server"; `off_sop` → config `off_sop` text |
| `active_step_done` | `Noul` | "Has this step been completed: <active step text>?" | — |
| `anomaly` | `Noul` | "Is anything in the scene inconsistent with the SOP so far?" | — |

**Active step** = the step most recently identified by `current_step` with confidence ≥ `accept_confidence`; before any such identification, the first pending step.

### 7.4 Client behaviour
- Thin wrapper around the SDK (the only module that imports `typesafe_sdk`).
- Timeout 1.5 s; retry with exponential backoff on 429/5xx (max 2 retries).
- Each request carries the `SceneState.version`; responses for a version older than the last applied decision are discarded.
- Output `Decision{version, current_step{choice, confidence, probabilities}, active_step_done, anomaly, latency_ms}`.

## 8. SOP tracking rules (`sop/`)

Step status ∈ `pending | active | done`. Alert levels: `info | warning | violation`.

| Rule | Condition | Result |
|---|---|---|
| Activate | `current_step = S` with confidence ≥ accept for `hold_seconds` | S → `active` |
| Complete | S active, `active_step_done` ≥ accept for `hold_seconds`, **and** S's `evidence` (if any) holds | S → `done`, record duration |
| Evidence-only complete | S's evidence holds for `evidence_only_after` s (1.0) without Jev confirmation | S → `done`, flagged `source: evidence` in the log |
| Hands-off steps | For a step marked `hands_off` (its last action, e.g. a latch, isn't visible), evidence only counts while no hand touches the part; evidence-only completion after `hands_off_seconds` (0.5) | applies to both completion paths |
| Order violation | S → `done` while a step that must precede S is not `done` (strict: all earlier steps; groups: all steps in earlier groups; any: never) | `order_violation` (violation) |
| Part removed | a `done` step's evidence slot is **observed** contradicting the evidence (e.g. `empty` where `installed` is required) for ≥ 1 s | `part_removed` (violation); S → `pending` |
| Off-SOP activity | `current_step = off_sop` with confidence ≥ accept for `hold_seconds` | `off_sop` (warning) |
| Anomaly | `anomaly` ≥ accept for `hold_seconds` | `anomaly` (warning) with scene snapshot |
| Incomplete | run ends with any step not `done` | `sop_incomplete` (violation) listing missing steps |
| Jev offline | 3 consecutive failed calls | `jev_offline` (warning); evidence-only mode until a call succeeds |

`SopTracker` is pure and deterministic (inputs: decisions, scene states, clock) so every rule is unit-testable.

## 9. Dashboard and API

### 9.1 Backend (FastAPI, single process)
- `GET /video.mjpg` — annotated frames, ~20 fps, 1280×720. Overlays drawn server-side: chassis box, slots (installed = green, empty = grey, occluded = amber), hands, held parts, fps/latency.
- `WS /ws` — typed JSON messages: `scene`, `decision`, `step_update`, `alert`, `run_summary`.
- `GET /sops` · `GET /sources` (sources available in this environment; omits webcam in the container) · `POST /runs` (body: sop id + source) · `POST /runs/{id}/stop` · `GET /runs/{id}/report` (JSON or CSV).
- Serves the built frontend from `web/dist`, so the demo is one command and one URL.

### 9.2 Frontend (React + Vite + TypeScript)
Layout as approved in the brainstorming mockup:
- **Top bar:** SOP name and version, source, run timer, status pill (On track / Deviation / Incomplete).
- **Left:** live video.
- **Right:** checklist — done (✓ + duration), active (highlighted + Jev confidence), pending, violated.
- **Below video:** "Jev now" strip (current step, confidence bar, off-SOP probability, call latency) and a timeline of step segments.
- **Bottom:** event log (timestamp, message, confidence, evidence).
- **Controls:** source picker (test runs, simulated scenarios, webcam), play/pause, restart, speed (1×/2×), download run report.

The frontend holds no SOP logic; it renders WebSocket messages.

## 10. Video sources and demo scenarios

- `FileSource` (plays at real-time speed by default; `--speed` and headless lockstep mode for eval), `WebcamSource` (device index; native only), `RtspSource` (URL). Webcam/RTSP disconnects reconnect with exponential backoff and raise `source_lost`.
- **Scenario videos** are pre-rendered once by `eval/make_scenarios.py` into `data/scenarios/` (gitignored), from Install_12 clips. Each ends with a 3 s freeze of its last frame so the final state can settle before the run ends:
  - **Incomplete run:** positions 1–5 (off-SOP + fans 1–4), then end. Expected: `sop_incomplete` listing fan_5, fan_6, psu_1, psu_2, cover.
  - **Fan removed:** positions 1–4 (off-SOP + fans 1–3), then the fan-3 clip played in reverse, then end. Expected: `part_removed` for fan_3 (plus `sop_incomplete` at the end).
- Wrong order and skipped steps cannot be built from this footage (each clip shows all prior installs) and are covered by scenario tests (§12).

## 11. Error handling

| Failure | Behaviour |
|---|---|
| Jev slow/erroring | Timeout + backoff (§7.4); stale responses discarded; evidence-only mode after 3 consecutive failures; dashboard badge "Jev offline". |
| Source drops | Reconnect with backoff; `source_lost` alert; file end → run ends normally. |
| Chassis not visible | Slot mapping pauses; dashboard shows "Chassis not visible"; alert after 3 s. |
| Invalid config | Startup fails with a message naming the field and the problem. |
| Missing weights / API key | Startup fails with a clear message. |
| Webcam requested in container | `POST /runs` rejects it: "Webcam isn't available in the container — run natively or use an RTSP source." |
| No GPU visible in container | Startup fails with a message pointing at Docker Desktop's WSL2 GPU setup (no silent CPU fallback — it would break the fps target). |
| Secrets | `TYPESAFE_API_KEY` from `.env` (gitignored; `.env.example` committed); never logged. |

## 12. Testing and evaluation

- **Unit tests (pytest, no network, no GPU):**
  - `SceneBuilder` with synthetic detections: slot matching under small rotations, 0.5 s smoothing, occlusion handling, hand zones, held parts.
  - `SopTracker`: every rule in §8, each order mode, thresholds, evidence check, removal, incomplete.
  - Config validation (each rule in §5) and question generation.
  - `JevDecider` with a fake client: debouncing, heartbeat, stale-version discard, retries, offline mode.
- **Contract test (opt-in, `pytest -m jev`):** a few fixed states against the real Jev API to catch SDK/API changes.
- **Scenario tests (opt-in, real Jev):** scripted `SceneState` sequences through `JevDecider` → `SopTracker`: wrong order (fan_2 before fan_1), skipped step (fan_3 never installed), cover before PSUs. Each must raise the expected alert.
- **End-to-end eval (`sopvision eval`):** replays Install_12 and Install_13 headless with the real detector and Jev; writes a JSON report with per-step completion-time error, step accuracy, off-SOP segment detection, alert list, per-frame slot states, and Jev latency stats; checks §2 acceptance criteria 1–5 and exits non-zero if any fail. Also runs the two scenario videos and checks their expected alerts.
  - **Lockstep clock:** eval processes every frame (no drops), all timers use video time, and each Jev answer is applied at its request's video time + 500 ms (Jev's stated worst-case latency) regardless of the real round trip. Results therefore don't depend on machine speed; real latencies are still recorded.
- **Docker parity (`sopvision compare <native.json> <docker.json>`):** checks §2 criterion 7 between a native and a container eval report; exits non-zero on any mismatch.
- **Container smoke test:** the image builds, the container sees the GPU, `serve` starts, and `GET /sops` and `GET /sources` respond.
- **Frontend:** a smoke test that the built dashboard loads and renders a recorded WebSocket message stream.

## 13. Project layout, tooling, and run paths

### 13.1 Layout
```
SOP/                                  git repo root
  .gitignore  .env.example  README.md
  Dockerfile  docker-compose.yml  .dockerignore
  pyproject.toml  uv.lock             Python 3.12, managed by uv
  sops/server_fan.yaml
  src/sopvision/
    config.py                         pydantic models, YAML loading, validation
    sources/                          VideoSource protocol; file, webcam, rtsp
    perception/                       Detector protocol; YOLO + ByteTrack
    scene/                            SceneBuilder, slot matching, smoothing, events
    decide/                           JevDecider, question builder, Jev client wrapper
    sop/                              SopTracker, order rules, alerts
    storage/                          JSONL RunLog, run reports
    server/                           FastAPI app: MJPEG, WebSocket, REST
    pipeline.py                       wires components; pipeline thread
    cli.py                            sopvision serve | calibrate | eval | compare
  training/                           sample_frames, autolabel, labelstudio_export/import, train
  eval/                               replay harness, metrics, make_scenarios
  web/                                React + Vite + TS dashboard
  tests/
  docs/superpowers/specs/
  models/  runs/  data/               gitignored (weights, run logs, frames/labels/scenarios)
  sop-sample-training-data/           gitignored (existing dataset; path set in config)
  sop-sample-training-data.tar.gz     gitignored
```

Python 3.12 rather than the system 3.14 because PyTorch/Ultralytics support for 3.14 is not yet dependable. Frontend uses the installed Node 25. Paths in SOP YAML (e.g. `models/server_fan/best.pt`) are relative to the repo root natively and to `/app` in the container, so the same config works in both. The dataset location comes from `SOP_DATA_ROOT`.

### 13.2 Run paths

**Native (`uv`) — the primary demo path.** Everything runs natively, including the Phase 1-only tools (`calibrate` needs a desktop window; training, auto-labelling and Label Studio stay native):
```
uv run sopvision serve                                   # dashboard on http://localhost:8000
uv run sopvision eval --out runs/eval-native.json
```

**Docker — the second, equivalent path** for `serve` and `eval`:
```
docker compose up --build                                # dashboard on http://localhost:8000
docker compose run --rm sopvision eval --out runs/eval-docker.json
uv run sopvision compare runs/eval-native.json runs/eval-docker.json
```

- **`Dockerfile` (multi-stage):** a Node stage builds `web/dist`; the runtime stage is `python:3.12-slim` + uv running `uv sync --frozen` from the **same `uv.lock`** as native, so package versions are identical. PyTorch's CUDA wheels bundle the CUDA runtime, so the host only needs the NVIDIA driver. Workdir `/app`; entrypoint `sopvision`; default command `serve --host 0.0.0.0`.
- **`.dockerignore`** excludes the dataset, the `.tar.gz`, `models/`, `runs/`, `data/`, `.venv/`, `node_modules/` and `.env` — otherwise the build context would be ~11 GB and the API key would be baked into the image.
- **`docker-compose.yml`:** one service, `sopvision`, with an NVIDIA GPU reservation (`deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`), port `8000:8000`, `env_file: .env` (provides `TYPESAFE_API_KEY`), and environment `SOP_DATA_ROOT=/data/server_fan`, `SOPVISION_IN_CONTAINER=1`.
- **Volumes:**

  | Host | Container | Mode | Why |
  |---|---|---|---|
  | `${SOP_DATASET_DIR:-./sop-sample-training-data}` | `/data` | read-only | test videos |
  | `./models` | `/app/models` | read-only | trained weights |
  | `./runs` | `/app/runs` | read-write | run logs, eval reports |
  | `./sops` | `/app/sops` | read-only | SOP configs |
  | `./data/scenarios` | `/app/data/scenarios` | read-only | scenario videos for the parity check |

- **Sources in the container:** file and RTSP (outbound to LAN cameras over the default bridge network). USB webcam is native-only: Docker Desktop on Windows can't see USB cameras without extra setup, so `GET /sources` omits it when `SOPVISION_IN_CONTAINER=1`.
- **Windows prerequisites:** Docker Desktop with the WSL2 backend and the Windows NVIDIA driver (both present; §3.3). Docker Desktop must be running.
- **Phase 2:** the same image and compose file retarget to a Linux on-prem GPU server with the NVIDIA Container Toolkit; only volume paths and `.env` change.

## 14. Phases

### Phase 0 — Feasibility test (~1 hour)
- Auto-label ~20 frames spanning all steps with SAM 3 and Grounding DINO; compare count-check pass rate and visual quality; pick one (or trigger the manual-label fallback).
- Verify the Jev SDK (package, call shape, response fields, Choice key constraints), measure latency from this machine with a hand-written `SceneState`, and check whether identical inputs give identical outputs.
- Verify Docker GPU access: `docker run --rm --gpus all` with a CUDA base image runs `nvidia-smi` and sees the RTX 4080.

### Phase 1 — Demo (this spec's implementation plan)
1. Repo skeleton, config schema + validation, video sources; `Dockerfile` + `docker-compose.yml` skeleton passing the container smoke test (so Docker problems surface early, not at the end).
2. Training tools; labelled dataset; trained `best.pt`.
3. `SceneBuilder` + calibrate tool.
4. `JevDecider` + `SopTracker`.
5. FastAPI server + React dashboard (built into the image).
6. Eval harness with lockstep clock, scenario videos, scenario tests, `compare`; meet §2 acceptance criteria 1–7 (including Docker parity).
7. Demo polish: README run instructions for both paths, a short demo script, and a recorded demo video as a backup for live showings.

### Phase 2 — Production (after approval; not planned yet)
- Onboarding workflow for new equipment/SOPs; active-learning loop queuing low-confidence frames for labelling.
- Retarget the Phase 1 image to a Linux on-prem GPU server; live cameras on the floor; database storage; multi-station dashboard; authentication; monitoring.
- Oriented boxes or homography anchoring if new rigs allow larger rotations.
- Operator alerts (sound / andon light); MES integration; review TypeSafe on-prem/VPC options to remove the cloud dependency.

## 15. Risks

| Risk | Mitigation |
|---|---|
| Jev is two weeks old in early access; the API may change | Single wrapper module; opt-in contract test; Phase 0 verification |
| Auto-labels poor on domain-specific parts (installed fan vs empty bay) | Phase 0 gate; count check prioritises review; manual fallback |
| Hands occlude slots during every install | `occluded` state keeps last value; 0.5 s smoothing; evidence holds |
| Jev can't tell step boundaries from abstract scene JSON alone | Hand zones + held parts + recent events in state; evidence-only completion; eval measures it |
| Only 2 test runs → noisy metrics | Treat Phase 1 numbers as indicative; more recordings in Phase 2 |
| Order/skip detection only validated synthetically | Scenario tests now; real footage when new equipment is recorded |
| Cloud dependency for decisions | Evidence-only fallback mode; Phase 2 reviews on-prem options |
| Docker GPU passthrough on Windows/WSL2 misconfigured | Phase 0 `nvidia-smi` check; skeleton smoke test in Phase 1 step 1; native path stays primary |
| Native (Windows) and container (Linux) builds of the same packages differ numerically | Same `uv.lock`; lockstep eval; parity tolerances in §2 criterion 7 |
| Reading the dataset through a Windows bind mount is slower in WSL2 | Video bitrate is low (~1 MB/s at 1080p30); acceptable for replay and eval; Phase 2 on Linux has native mounts |

## 16. References
- TypeSafe AI — Introducing System One Models & Jev: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Cloudflare AI — Jev (typesafe) model page: https://developers.cloudflare.com/ai/models/typesafe/jev/
- How to Use Jev: a practical guide (DEV Community): https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
