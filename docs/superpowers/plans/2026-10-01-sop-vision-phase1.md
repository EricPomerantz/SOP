# SOP Vision Phase 1 Implementation Plan

**Goal:** Build the Phase 1 demo: a live SOP monitor that detects server-fan-install parts with a custom YOLO model, turns detections into a JSON scene state, asks Jev which step is happening and whether it's done, tracks the SOP, and shows it all on a web dashboard. It runs natively (`uv`) and in Docker with GPU.

**Architecture:** `VideoSource → Detector → SceneBuilder → JevDecider → SopTracker`, wired by `pipeline.py`. A FastAPI server streams MJPEG video and WebSocket status messages to a React dashboard. Offline `training/` tools build the detector; `sopvision.evaluation` replays test videos in a lockstep clock and checks the acceptance criteria. Facts are deterministic code, judgments come from Jev, bookkeeping is deterministic code.

**Tech Stack:** Python 3.12 (uv), PyTorch CUDA 12.8 wheels, Ultralytics YOLO + ByteTrack, OpenCV, scipy, pydantic v2 + pydantic-settings, ruamel.yaml, typer, FastAPI + uvicorn, `typesafe-sdk` (Jev), pytest + pytest-asyncio; Hugging Face transformers (auto-labelling only); Label Studio (via `uvx`); React + Vite + TypeScript + vitest; Docker Desktop (WSL2) + Compose.

**Spec:** `docs/superpowers/specs/2026-10-01-sop-vision-design.md` — read it before starting any task; section numbers below (§) refer to it.

## Global Constraints

- Python `>=3.12,<3.13` managed by `uv`; one `uv.lock` shared by native and Docker. PyTorch/torchvision come from the `https://download.pytorch.org/whl/cu128` index.
- Ultralytics is company-licensed (enterprise); no licence swap needed.
- Only `src/sopvision/decide/jev_client.py` imports `typesafe_sdk`.
- `TYPESAFE_API_KEY` is read from `.env` (gitignored) as a `SecretStr`, and never logged or written to run logs.
- Video frames never leave the machine; only the JSON state in §7.2 is sent to Jev.
- Every timer (settle, debounce, heartbeat, holds, chassis-lost) reads an injected clock: wall time live, video time in eval.
- Spec values (use exactly): `accept_confidence 0.8`, `hold_seconds 1.0`, `evidence_only_after 3.0`, slot settle `0.5 s`, chassis lost after `3 s`, Jev throttle `0.25 s`, heartbeat `2 s`, Jev timeout `1.5 s`, max `2` retries, last `10` events in state, lockstep Jev latency `0.5 s`, part-removed hold `1.0 s`, YOLO `imgsz 1280`, MJPEG `~20 fps` fit within `1280×720`, port `8000`.
- Data splits: train = Install 1, 3–10; val = Install 11; test = Install 12, 13. Test runs are never sampled for training.
- Paths in SOP YAML are relative to the repo root natively and to `/app` in the container; the dataset root comes from `SOP_DATA_ROOT`.
- Webcam is native-only; the container supports file and RTSP sources.
- In the container, startup fails if no GPU is visible (no CPU fallback).
- UI copy is sentence case, no exclamation marks.
- pytest markers: `gpu`, `jev`, `slow`, `dataset`; default run excludes `gpu` and `jev`.

## Review Focus

1. **Jev response fields:** `Choice` confidence and `Noul` probability are read from the real SDK response shape, not guessed — test in Task 14 against a response recorded in Task 2.
2. **Video without FPS metadata** (common for webcams and some files report `0` fps): timestamps must come from `CAP_PROP_POS_MSEC`, not divide by zero — test in Task 6.
3. **`SOP_DATA_ROOT` unset or wrong:** startup must say which variable to fix, not crash deep in OpenCV — test in Task 4.
4. **Stopping a run while a Jev call is in flight:** no decision, step update or log line may land after `run_summary` — tests in Tasks 15 and 18.
5. **Input that isn't 1920×1080** (webcam 640×480, RTSP 1280×720): overlays and MJPEG must scale correctly — test in Task 18.

---

## File map

```
pyproject.toml  uv.lock  .env.example  README.md
Dockerfile  .dockerignore  docker-compose.yml
sops/server_fan.yaml
spikes/jev_smoke.py                        Phase 0 (throwaway)
docs/superpowers/notes/phase0.md           Phase 0 + training results
scripts/docker_smoke.py
src/sopvision/
  cli.py  settings.py  config.py  dataset.py  doctor.py  calibrate.py  messages.py  pipeline.py
  sources/   base.py file.py webcam.py rtsp.py latest.py registry.py
  perception/detector.py
  scene/     geometry.py types.py builder.py overlay.py
  decide/    types.py questions.py jev_client.py decider.py
  sop/       types.py tracker.py
  storage/   runlog.py report.py
  server/    app.py runs.py frames.py
  evaluation/ scenarios.py metrics.py run_eval.py compare.py
training/    sample_frames.py count_check.py yolo_format.py autolabel.py labelstudio.py train.py
web/         package.json vite.config.ts tsconfig.json index.html src/{main.tsx,App.tsx,types.ts,state.ts,api.ts,styles.css,components/*}
tests/       mirrors src/ and training/; tests/helpers.py; tests/jev/
```

Deviations from spec §13.1, decided here: eval code lives in the package as `sopvision.evaluation` (so it ships in the Docker image; `eval` would shadow a builtin); CLI gains `doctor` and `make-scenarios`; the server gains `GET /runs/current` (a dashboard opened mid-run needs a snapshot) and `POST /runs/{id}/control` (play/pause/speed in §9.2 need an endpoint); `StepSpec` gains an optional `label` and `SopConfig` an `idle` text (keeps the §7.3 idle wording out of code). Phase 0's auto-label comparison runs in Task 8 with the real tooling instead of throwaway code.

**👤 marks a step the user does** (needs their key, clicks, or judgment). The executor stops and asks.

---

### Task 1: Toolchain and repo skeleton

**Files:**
- Create: `pyproject.toml`, `.env.example`, `README.md`, `src/sopvision/__init__.py`, `src/sopvision/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: console script `sopvision` → `sopvision.cli:app` (typer) with commands `serve`, `calibrate`, `eval`, `compare`, `doctor`, `make-scenarios` (stubs printing "Not implemented yet" and exiting 2 until their task lands).

- [ ] **Step 1: Install uv and Python 3.12**

Run: `winget install --id astral-sh.uv -e` then, in a new shell, `uv --version` and `uv python install 3.12`
Expected: uv prints a version; Python 3.12 installs.

- [ ] **Step 2: Write `pyproject.toml`**

Project `sopvision`, `requires-python = ">=3.12,<3.13"`, src layout (`packages = ["src/sopvision"]` via hatchling), script `sopvision = "sopvision.cli:app"`.
Dependencies: `ultralytics`, `torch`, `torchvision`, `numpy`, `scipy`, `pydantic>=2`, `pydantic-settings`, `pyyaml`, `ruamel.yaml`, `typer`, `fastapi`, `uvicorn[standard]`, `httpx`, `typesafe-sdk`.
Groups: `dev = ["pytest", "pytest-asyncio"]`; `training = ["transformers", "accelerate", "pillow"]`.
Torch index:
```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["gpu: needs CUDA", "jev: calls the real Jev API", "slow: long-running", "dataset: needs the server_fan dataset"]
addopts = "-m 'not gpu and not jev'"
pythonpath = ["."]
```
`pythonpath = ["."]` makes top-level `training/` importable in tests.

- [ ] **Step 3: Write the failing test** `tests/test_cli.py`

```python
from typer.testing import CliRunner
from sopvision.cli import app

def test_cli_lists_all_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["serve", "calibrate", "eval", "compare", "doctor", "make-scenarios"]:
        assert cmd in result.output
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv sync` then `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: sopvision.cli`.

- [ ] **Step 5: Implement `src/sopvision/cli.py`** with the six stub commands, plus `.env.example` containing `TYPESAFE_API_KEY=`, `SOP_DATA_ROOT=./sop-sample-training-data/server_fan`, `SOP_DATASET_DIR=./sop-sample-training-data`, `SOPVISION_RTSP_URLS=`, and a one-paragraph `README.md` pointing at the spec.

- [ ] **Step 6: Verify tests and CUDA**

Run: `uv run pytest -v` → PASS. Run: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA GeForce RTX 4080`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example README.md src tests
git commit -m "chore: project skeleton with uv, CUDA torch, CLI stubs"
```

---

### Task 2: Phase 0 — Jev and Docker GPU checks (spike)

**Files:**
- Create: `spikes/jev_smoke.py`, `docs/superpowers/notes/phase0.md`, `tests/decide/fixtures/jev_response.json`

**Interfaces:**
- Produces: `phase0.md` recording the verified SDK import path, call signature, response attribute paths for `Choice` (choice, confidence, probabilities) and `Noul` (probability), any constraint on Choice option keys, latency p50/p90/max over 20 calls, whether identical inputs return identical outputs, and the Docker GPU result. `jev_response.json` is one raw response serialised field-by-field (Task 14's fake SDK replays it).

- [ ] **Step 1: 👤 Put the API key in `.env`** (copy `.env.example` → `.env`, fill `TYPESAFE_API_KEY`). 👤 Start Docker Desktop.

- [ ] **Step 2: Write `spikes/jev_smoke.py`**

Loads `.env`; sends the §6.3 example scene as `state` with the §7.3 questions (Choice with keys `fan_1`…`fan_6`, `psu_1`, `psu_2`, `cover`, `idle`, `off_sop`; two Noul questions); calls 20 times; prints latencies, `repr()` of the first response, and whether all 20 answers are identical; writes the first response's fields to `tests/decide/fixtures/jev_response.json`.

- [ ] **Step 3: Run it**

Run: `uv run python spikes/jev_smoke.py`
Expected: 20 successful calls; latency p90 ≤ 500 ms. If the SDK's names differ from spec §3.1, record the real ones — only Task 14's `jev_client.py` depends on them.

- [ ] **Step 4: Check Docker GPU**

Run: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
Expected: table listing `NVIDIA GeForce RTX 4080`.

- [ ] **Step 5: Write `phase0.md`** with the findings above, then commit.

```bash
git add spikes docs/superpowers/notes tests/decide/fixtures
git commit -m "spike: verify Jev SDK, latency and Docker GPU access"
```

---

### Task 3: Docker skeleton and `doctor`

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `src/sopvision/doctor.py`, `tests/test_doctor.py`
- Modify: `src/sopvision/cli.py` (`doctor` command)

**Interfaces:**
- Produces: `GpuReport(cuda_available: bool, device_name: str | None, torch_version: str)`; `gpu_report() -> GpuReport`; `require_gpu(in_container: bool, report: GpuReport) -> None` raising `GpuUnavailableError`. `serve` and `eval` call `require_gpu` at startup (Tasks 19, 22).

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from sopvision.doctor import GpuReport, require_gpu, GpuUnavailableError

NO_GPU = GpuReport(cuda_available=False, device_name=None, torch_version="x")
GPU = GpuReport(cuda_available=True, device_name="NVIDIA GeForce RTX 4080", torch_version="x")

def test_container_without_gpu_fails_with_setup_hint():
    with pytest.raises(GpuUnavailableError, match="No GPU visible in the container"):
        require_gpu(True, NO_GPU)

def test_native_without_gpu_does_not_raise():
    require_gpu(False, NO_GPU)

def test_container_with_gpu_passes():
    require_gpu(True, GPU)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_doctor.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `doctor.py` and the `doctor` command.** Error message: `"No GPU visible in the container. Docker Desktop needs the WSL2 backend and the Windows NVIDIA driver; check: docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi"`. `doctor` prints `cuda: <bool>`, `gpu: <name>`, `torch: <ver>`, `in_container: <bool>` (from env `SOPVISION_IN_CONTAINER`) and exits 1 when `require_gpu` raises.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Write the Docker files** per spec §13.2:
  - `Dockerfile`: `FROM python:3.12-slim-bookworm`; `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`; `apt-get install -y --no-install-recommends libgl1 libglib2.0-0`; `WORKDIR /app`; copy `pyproject.toml uv.lock`, `uv sync --frozen --no-dev --no-install-project`; copy `src/ sops/`, `uv sync --frozen --no-dev`; `ENV PATH=/app/.venv/bin:$PATH`; `ENTRYPOINT ["sopvision"]`; `CMD ["serve","--host","0.0.0.0"]`. (Task 20 adds the web build stage.)
  - `.dockerignore`: `sop-sample-training-data/`, `*.tar.gz`, `models/`, `runs/`, `data/`, `.venv/`, `**/node_modules/`, `.env`, `.git/`, `web/dist/`.
  - `docker-compose.yml`: service `sopvision`, `build: .`, GPU reservation `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]`, `ports: ["8000:8000"]`, `env_file: .env`, environment `SOP_DATA_ROOT=/data/server_fan`, `SOPVISION_IN_CONTAINER=1`, and the five volumes from §13.2 (dataset via `${SOP_DATASET_DIR:-./sop-sample-training-data}:/data:ro`).

- [ ] **Step 6: Verify the container sees the GPU**

Run: `docker compose build` then `docker compose run --rm sopvision doctor`
Expected: `cuda: True`, `gpu: NVIDIA GeForce RTX 4080`, `in_container: True`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml src/sopvision/doctor.py src/sopvision/cli.py tests/test_doctor.py
git commit -m "feat: Docker image with GPU and doctor command"
```

---

### Task 4: Settings and SOP config

**Files:**
- Create: `src/sopvision/settings.py`, `src/sopvision/config.py`, `sops/server_fan.yaml`, `tests/conftest.py`, `tests/test_settings.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings` (pydantic-settings, `env_file=".env"`): `typesafe_api_key: SecretStr = SecretStr("")` (emptiness is checked at `serve`/`eval` startup, Task 19), `sop_data_root: Path = Path("sop-sample-training-data/server_fan")`, `sopvision_in_container: bool = False`, `sopvision_rtsp_urls: list[str] = []` (comma-separated env), `sops_dir: Path = Path("sops")`, `runs_dir: Path = Path("runs")`, `scenarios_dir: Path = Path("data/scenarios")`; `check_data_root() -> None`.
  - `ConfigError(Exception)`.
  - `SlotSpec(at: tuple[float, float], filled_by: str, empty_as: str | None = None, match_radius: float | None = None)`; `Layout(anchor: str, match_radius: float, slots: dict[str, SlotSpec])`; `StepSpec(id: str, text: str, label: str | None = None, evidence: dict[str, Literal["installed", "empty"]] = {})`; `DetectorSpec(weights: Path, classes: list[str], autolabel_prompts: dict[str, str])`; `DecisionSpec(accept_confidence=0.8, hold_seconds=1.0, evidence_only_after=3.0)`; `SopConfig(sop, version, description, detector, layout, steps, order: Literal["strict","any","groups"], groups: list[list[str]] | None = None, off_sop: str, idle: str, decisions: DecisionSpec)`.
  - `SopConfig.step(id) -> StepSpec`, `.step_label(id) -> str` (label or id), `.slot_radius(slot) -> float`, `.must_precede(step_id) -> set[str]`.
  - `load_sop(path: Path) -> SopConfig`; `list_sops(dir: Path) -> list[SopConfig]`.
  - `tests/conftest.py` fixture `sop` → `load_sop(Path("sops/server_fan.yaml"))`.

- [ ] **Step 1: Write `sops/server_fan.yaml`** from spec §5 with all 9 slots and 9 steps (texts from spec §5 / dataset `actions.json`, labels `Fan 1`…`Fan 6`, `PSU 1`, `PSU 2`, `Cover`), `idle: "No one is working on the server"`, and **placeholder** slot coordinates (overwritten by Task 13): `fan_bay_1..3 = [0.47, 0.62|0.72|0.82]`, `fan_bay_4..6 = [0.47, 0.18|0.28|0.38]`, `psu_bay_1 = [0.12, 0.15]`, `psu_bay_2 = [0.12, 0.85]`, `cover = [0.5, 0.5]` with `match_radius: 0.25`.

- [ ] **Step 2: Write the failing tests** — one per validation rule in §5, plus order and settings:

```python
def test_server_fan_loads(sop):
    assert [s.id for s in sop.steps][:2] == ["fan_1", "fan_2"] and len(sop.steps) == 9
    assert len(sop.layout.slots) == 9 and sop.slot_radius("cover") == 0.25

@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["layout"]["slots"]["fan_bay_1"].update(filled_by="fann"), "slot fan_bay_1: filled_by 'fann' is not a detector class"),
    (lambda d: d["steps"][0].update(evidence={"fan_bay_9": "installed"}), "step fan_1: evidence slot 'fan_bay_9' is not declared"),
    (lambda d: d["steps"][1].update(id="fan_1"), "duplicate step id 'fan_1'"),
    (lambda d: d["layout"]["slots"]["fan_bay_1"].update(at=[1.2, 0.5]), "slot fan_bay_1: coordinates must be within [0, 1]"),
    (lambda d: d.update(order="groups", groups=[["fan_1"]]), "groups must contain every step exactly once"),
])
def test_invalid_config_names_field_and_problem(tmp_path, mutate, message): ...  # load raw YAML, mutate, dump, load_sop → ConfigError(match=message)

def test_evidence_value_must_be_installed_or_empty(tmp_path): ...  # "present" → ConfigError

def test_must_precede_by_order_mode(sop):
    assert sop.must_precede("fan_3") == {"fan_1", "fan_2"}                     # strict
    assert sop.model_copy(update={"order": "any"}).must_precede("cover") == set()
    # groups [[fans],[psus],[cover]] → must_precede("psu_1") == all six fans

def test_api_key_never_in_repr(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-secret")
    assert "sk-secret" not in repr(Settings())

def test_missing_data_root_names_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("SOP_DATA_ROOT", str(tmp_path / "nope"))
    with pytest.raises(ConfigError, match="SOP_DATA_ROOT points to .*nope, which doesn't exist. Set it in .env."):
        Settings().check_data_root()
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_config.py tests/test_settings.py -v` → FAIL.

- [ ] **Step 4: Implement `settings.py` and `config.py`.** Cross-field rules go in a `model_validator(mode="after")` on `SopConfig`; `load_sop` converts pydantic `ValidationError` into `ConfigError` whose message starts with the file name.

- [ ] **Step 5: Run tests** → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat: settings and validated SOP config"` (add the files above).

---

### Task 5: Dataset index and ground truth

**Files:**
- Create: `src/sopvision/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Produces: `ACTION_TO_STEP: dict[int, str]` (`1–6 → fan_1–fan_6`, `7 → psu_1`, `8 → psu_2`, `9 → cover`, `10 → off_sop`); `parse_clip_name(name: str) -> tuple[int, int, int, int]` (action, run, occurrence, position; `ValueError` otherwise); `ClipInfo(path: Path, action: int, run: int, occurrence: int, position: int, duration: float)`; `Segment(step_id: str, start: float, end: float)`; `RunIndex(run: int, split: Literal["train","val","test"], clips: list[ClipInfo], raw_path: Path, raw_duration: float)` with `.segments -> list[Segment]` (cumulative clip durations, by position), `.timing_valid -> bool` (`abs(sum - raw) <= 0.1`), `.actions_by_position -> list[int]`; `video_duration(path: Path) -> float`; `load_index(data_root: Path, probe=video_duration) -> list[RunIndex]` (sorted by run; split: `test/` → test, run 11 → val, else train).

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_clip_name():
    assert parse_clip_name("10_Install_1_2_11.mp4") == (10, 1, 2, 11)
    with pytest.raises(ValueError):
        parse_clip_name("Install_1.MP4")

def test_segments_are_cumulative_and_timing_flag(tmp_path):
    # fake tree: train/Install_3 with clips at positions 1..3 of 5 s, 10 s, 4 s; raw probe 21.2 s
    idx = load_index(fake_root, probe=fake_probe)
    run = idx[0]
    assert [(s.step_id, s.start, s.end) for s in run.segments] == [("off_sop", 0, 5), ("fan_1", 5, 15), ("fan_2", 15, 19)]
    assert run.timing_valid is False

def test_split_assignment(tmp_path): ...  # Install_11 under train/ → "val"; Install_12 under test/ → "test"

@pytest.mark.dataset
def test_real_dataset_index():
    idx = {r.run: r for r in load_index(Settings().sop_data_root)}
    assert sorted(idx) == [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    assert [r for r in idx if not idx[r].timing_valid] == [3, 5]
    assert idx[12].actions_by_position == [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement `dataset.py`** (regex `^(\d+)_Install_(\d+)_(\d+)_(\d+)\.mp4$`; raw path `raw/Install_{run}.MP4`; `video_duration` = frame count / fps via OpenCV).
- [ ] **Step 4: Run tests** — `uv run pytest tests/test_dataset.py -v -m "dataset or not dataset"` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: dataset index with ground-truth segments"`.

---

### Task 6: Video sources

**Files:**
- Create: `src/sopvision/sources/{__init__,base,file,webcam,rtsp,latest,registry}.py`, `tests/sources/conftest.py` (`tiny_video` fixture), `tests/sources/test_file.py`, `test_rtsp.py`, `test_latest.py`, `test_registry.py`

**Interfaces:**
- Produces:
  - `Frame(t: float, index: int, image: np.ndarray)` (BGR).
  - `VideoSource` Protocol: `fps: float`, `frames() -> Iterator[Frame]`, `close() -> None`.
  - `FileSource(path, mode: Literal["realtime","lockstep"] = "realtime", speed: float = 1.0, open_capture=cv2.VideoCapture, sleep=time.sleep, clock=time.monotonic)` with `pause()`, `resume()`, `set_speed(x)`. `t = index / fps`, or `CAP_PROP_POS_MSEC / 1000` when the reported fps ≤ 0. Realtime paces to `t / speed`; lockstep never sleeps.
  - `WebcamSource(device: int, open_capture=cv2.VideoCapture)`; `RtspSource(url, open_capture=cv2.VideoCapture, sleep=time.sleep, on_lost: Callable[[], None] | None = None, backoff=(1, 2, 4, 8, 16))` — on read failure calls `on_lost` once per outage, reopens with the next backoff delay (capped at 16 s, retries forever), timestamps from wall clock since start.
  - `LatestFrameReader(source)`: thread keeps only the newest frame; `get(timeout: float) -> Frame | None`; `dropped: int`; `close()`; `exhausted: bool`.
  - `SourceSpec(id: str, kind: Literal["file","webcam","rtsp"], label: str, path: Path | None = None, device: int | None = None, url: str | None = None)`; `SourceUnavailableError`; `list_sources(settings) -> list[SourceSpec]` (ids `test:Install_12`, `test:Install_13` → raw videos; `scenario:<stem>` for `scenarios_dir/*.mp4`; `webcam:0` only when not in container; `rtsp:<i>` per configured URL); `open_source(spec, settings, mode="realtime") -> VideoSource`.

- [ ] **Step 1: Write the failing tests** (`tiny_video` writes a 10-frame 10 fps 64×48 mp4 whose frame `i` is filled with value `i*20`):

```python
def test_lockstep_timestamps(tiny_video):
    ts = [f.t for f in FileSource(tiny_video, mode="lockstep").frames()]
    assert ts == pytest.approx([i / 10 for i in range(10)])

def test_realtime_paces_by_speed(tiny_video):
    sleeps = []; src = FileSource(tiny_video, speed=2.0, sleep=sleeps.append, clock=lambda: 0.0)
    list(src.frames()); assert sum(sleeps) == pytest.approx(0.45, abs=0.01)

def test_zero_fps_metadata_uses_pos_msec():   # Review Focus 2
    cap = FakeCapture(fps=0.0, pos_msec=[0, 40, 80])   # defined in conftest
    assert [f.t for f in FileSource("x", mode="lockstep", open_capture=lambda _: cap).frames()] == [0, 0.04, 0.08]

def test_rtsp_reconnects_with_backoff():
    caps = [FakeCapture(fail_after=2), FakeCapture(open_ok=False), FakeCapture(frames=3)]
    lost, sleeps = [], []
    src = RtspSource("rtsp://x", open_capture=lambda _: caps.pop(0), sleep=sleeps.append, on_lost=lambda: lost.append(1))
    got = list(itertools.islice(src.frames(), 5))
    assert len(got) == 5 and lost == [1] and sleeps == [1, 2]

def test_latest_reader_drops_old_frames(): ...  # fast fake source, consumer sleeps 50 ms → dropped > 0, indices strictly increasing

def test_webcam_not_listed_or_openable_in_container(settings_in_container):
    assert all(s.kind != "webcam" for s in list_sources(settings_in_container))
    with pytest.raises(SourceUnavailableError, match="Webcam isn't available in the container — run natively or use an RTSP source."):
        open_source(SourceSpec(id="webcam:0", kind="webcam", label="Webcam", device=0), settings_in_container)

def test_lists_scenarios_and_rtsp(tmp_path): ...  # scenarios_dir with fan_removed.mp4 + one RTSP URL → ids include "scenario:fan_removed", "rtsp:0"
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement the sources.**
- [ ] **Step 4: Run tests** — `uv run pytest tests/sources -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: file, webcam and RTSP sources with newest-frame reader"`.

---

### Task 7: Frame sampling and count check

**Files:**
- Create: `training/__init__.py`, `training/sample_frames.py`, `training/count_check.py`, `tests/training/test_sample_frames.py`, `tests/training/test_count_check.py`

**Interfaces:**
- Consumes: `RunIndex`, `load_index` (Task 5).
- Produces:
  - `SampleSpec(run: int, split: str, position: int, action: int, t_in_clip: float, clip_path: Path, tag: str)`; `plan_samples(runs: list[RunIndex]) -> list[SampleSpec]` — per clip at 10/35/65/90 % (tags `p10 p35 p65 p90`) plus, for every boundary between consecutive clips, 0.5 s before (tag `bm`, last clip) and 0.5 s after (tag `bp`, next clip). Runs with `split == "test"` are dropped.
  - `extract(samples, out_dir: Path) -> Path` writes `out_dir/{split}/Install_{run}_{position:02d}_{tag}.jpg` and `out_dir/manifest.jsonl` rows `{image, run, split, position, action, actions_by_position}`. CLI: `uv run python -m training.sample_frames --out data/frames`.
  - `ExpectedCounts(fan, fan_bay_empty, psu, psu_bay_empty, cover, chassis)` — each `tuple[int, int] | None` (inclusive; `None` = unconstrained); `expected_counts(actions_by_position: list[int], position: int) -> ExpectedCounts`; `CheckResult(ok: bool, reasons: list[str])`; `check(counts: dict[str, int], expected: ExpectedCounts, hands: int) -> CheckResult`.
- Rules for `expected_counts` (spec §6.1 step 3): fans done = fan actions (1–6) at earlier positions; if the current clip is a fan step the range is `(k, k+1)`, else `(k, k)`; `fan_bay_empty = (6 - hi, 6 - lo)`; PSUs likewise with actions 7–8 and 2 bays; `cover` = `(0,0)` before the cover step, `(0,1)` during, `(1,1)` after; during or after the cover step, fan/fan_bay_empty/psu/psu_bay_empty are `None`; `chassis = (1,1)`. In `check`, when `hands > 0` only the upper bound applies to fan/fan_bay_empty/psu/psu_bay_empty. Reason format: `"fan: 4 not in 2–3"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_times_and_boundaries():
    run = fake_run(durations=[10.0, 4.0])          # helper builds a RunIndex, split="train"
    got = sorted((s.position, round(s.t_in_clip, 2), s.tag) for s in plan_samples([run]))
    assert got == [(1, 1.0, "p10"), (1, 3.5, "p35"), (1, 6.5, "p65"), (1, 9.0, "p90"), (1, 9.5, "bm"),
                   (2, 0.4, "p10"), (2, 0.5, "bp"), (2, 1.4, "p35"), (2, 2.6, "p65"), (2, 3.6, "p90")]

def test_test_runs_never_sampled():
    assert plan_samples([fake_run(split="test")]) == []

SEQ = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def test_expected_during_third_fan():
    e = expected_counts(SEQ, 4)
    assert (e.fan, e.fan_bay_empty, e.psu, e.cover, e.chassis) == ((2, 3), (3, 4), (0, 0), (0, 0), (1, 1))
def test_cover_hides_parts():
    assert expected_counts(SEQ, 10).fan is None and expected_counts(SEQ, 10).cover == (0, 1)
    assert expected_counts(SEQ, 11).cover == (1, 1)
def test_check_reasons_and_hand_allowance():
    e = expected_counts(SEQ, 4)
    assert check({"fan": 4, "fan_bay_empty": 2, "chassis": 1, "cover": 0, "psu": 0, "psu_bay_empty": 2}, e, hands=0).reasons[0] == "fan: 4 not in 2–3"
    assert check({"fan": 1, "fan_bay_empty": 3, "chassis": 1, "cover": 0, "psu": 0, "psu_bay_empty": 2}, e, hands=1).ok
```

- [ ] **Step 2: Run to verify failure** → FAIL.
- [ ] **Step 3: Implement both modules.**
- [ ] **Step 4: Run tests** → PASS. Then run `uv run python -m training.sample_frames --out data/frames` → ≈ 550 JPEGs (9 train runs + Install_11 val) and `data/frames/manifest.jsonl`.
- [ ] **Step 5: Commit** — `git commit -m "feat: frame sampling and expected-count check"`.

---

### Task 8: Auto-labeller and comparison (Phase 0 gate)

**Files:**
- Create: `training/yolo_format.py`, `training/autolabel.py`, `tests/training/test_yolo_format.py`, `tests/training/test_autolabel.py`

**Interfaces:**
- Consumes: manifest (Task 7), `expected_counts`/`check` (Task 7), `SopConfig.detector` (Task 4).
- Produces:
  - `LabeledBox(cls: str, xyxy: tuple[float, float, float, float], conf: float)`; `to_yolo(box, w, h, classes) -> str` (`"<idx> cx cy bw bh"`, 6 decimals); `from_yolo(line, w, h, classes) -> LabeledBox`.
  - `postprocess(boxes: list[LabeledBox], iou: float = 0.5, single: tuple[str, ...] = ("chassis", "cover")) -> list[LabeledBox]` — class-wise NMS, keep only the top box for `single` classes.
  - `Labeler` Protocol: `name: str`, `label(image_rgb: np.ndarray) -> list[LabeledBox]`; `GroundingDinoLabeler(prompts, model_id="IDEA-Research/grounding-dino-base", box_threshold=0.35, text_threshold=0.25)`; `Sam3Labeler(prompts, model_id="facebook/sam3", threshold=0.5)` (one text prompt per class, boxes from instance masks; follow the model card for the processor calls).
  - CLI `uv run --group training python -m training.autolabel compare --n 20 --out data/autolabel_compare` (20 frames spread across positions; per labeller: overlay JPEGs + count-check pass rate) and `... run --labeler gdino|sam3 --out data/labels_auto` (YOLO `.txt` per frame + `flags.jsonl` rows `{image, ok, reasons}`).

- [ ] **Step 1: 👤 Request access to `facebook/sam3` on Hugging Face and run `uv run --group training hf auth login`.** If access isn't granted, compare Grounding DINO alone and note it in `phase0.md`.

- [ ] **Step 2: Write the failing tests**

```python
def test_yolo_round_trip():
    b = LabeledBox("fan", (100.0, 50.0, 300.0, 250.0), 0.9)
    back = from_yolo(to_yolo(b, 1920, 1080, CLASSES), 1920, 1080, CLASSES)
    assert back.cls == "fan" and back.xyxy == pytest.approx(b.xyxy, abs=1e-2)

def test_postprocess_nms_and_single_chassis():
    boxes = [LabeledBox("chassis", (0, 0, 10, 10), 0.6), LabeledBox("chassis", (1, 1, 11, 11), 0.9),
             LabeledBox("fan", (0, 0, 10, 10), 0.8), LabeledBox("fan", (1, 0, 11, 10), 0.7),   # IoU ≈ 0.82 → one dropped
             LabeledBox("fan", (20, 0, 30, 10), 0.7)]                                          # separate fan kept
    out = postprocess(boxes)
    assert [b.conf for b in out if b.cls == "chassis"] == [0.9]
    assert len([b for b in out if b.cls == "fan"]) == 2

def test_unknown_class_rejected():
    with pytest.raises(ValueError): to_yolo(LabeledBox("dog", (0, 0, 1, 1), 1.0), 10, 10, CLASSES)
```

- [ ] **Step 3: Run to verify failure** → FAIL. **Step 4: Implement.** **Step 5: Run tests** → PASS.

- [ ] **Step 6: Run the comparison**

Run: `uv run --group training python -m training.autolabel compare --n 20 --out data/autolabel_compare`
Expected: a pass rate per labeller, e.g. `gdino: 14/20 pass`, plus overlay images.

- [ ] **Step 7: 👤 Choose the labeller with the user** (view a few overlays together). Record the choice and pass rates in `phase0.md`. **Gate:** if the best pass rate is below 50 %, use the manual fallback (Task 9 Step 5 exports 200 frames without predictions).

- [ ] **Step 8: Label all frames** — `uv run --group training python -m training.autolabel run --labeler <choice> --out data/labels_auto` → one `.txt` per frame and `flags.jsonl`.

- [ ] **Step 9: Commit** — `git commit -m "feat: zero-shot auto-labelling with count-check flags"` (code, tests, `phase0.md`).

---

### Task 9: Label Studio round trip

**Files:**
- Create: `training/labelstudio.py`, `tests/training/test_labelstudio.py`

**Interfaces:**
- Consumes: manifest (Task 7), `data/labels_auto` + `flags.jsonl` (Task 8), `to_yolo`/`from_yolo` (Task 8).
- Produces:
  - `label_config(classes) -> str` — `<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image">` with one `<Label value="..."/>` per class.
  - `yolo_to_ls(line, classes) -> dict` / `ls_to_yolo(result: dict, classes) -> str` (Label Studio `value.x/y/width/height` are top-left percentages).
  - `export_tasks(manifest, labels_dir, flags, out_json, doc_root, model_version, limit: int | None = None, with_predictions: bool = True) -> int` — task `data = {image: "/data/local-files/?d=<path relative to doc_root, forward slashes>", flagged, reasons}`; `predictions = [{model_version, result: [...]}]`; flagged tasks first, then manifest order; `limit` picks evenly spaced frames.
  - `import_export(export_json, manifest, classes, out_dir=Path("data/yolo")) -> ImportStats(annotated: int, prediction_only: int)` — per task, the latest annotation with `was_cancelled == False`, else the prediction; copies images to `data/yolo/images/{train,val}/`, labels to `data/yolo/labels/{train,val}/`, writes `data/yolo/dataset.yaml` (`path`, `train: images/train`, `val: images/val`, `names` in `classes` order).
  - CLI: `python -m training.labelstudio config | export | import`.

- [ ] **Step 1: Write the failing tests**

```python
def test_percent_round_trip():
    line = "1 0.500000 0.400000 0.100000 0.200000"
    assert ls_to_yolo(yolo_to_ls(line, CLASSES), CLASSES) == line

def test_flagged_tasks_first(tmp_path): ...       # 3 frames, middle one flagged → exported order [middle, first, last]
def test_annotation_beats_prediction(tmp_path): ...  # task with annotation + prediction → label from annotation; stats.annotated == 1
def test_cancelled_annotation_ignored(tmp_path): ... # only cancelled annotation → prediction used; stats.prediction_only == 1
def test_dataset_yaml_names_and_split(tmp_path): ... # names == CLASSES; Install_11 frame lands in images/val
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.

- [ ] **Step 5: Export tasks** — `uv run python -m training.labelstudio export --out data/labelstudio/tasks.json` (fallback from Task 8 gate: `--no-predictions --limit 200`).

- [ ] **Step 6: 👤 Review in Label Studio (~1 h).** The executor gives the user these exact steps:
  1. Run `$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"; $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\Users\Eric\Desktop\Projects\SOP\data"; uvx --python 3.12 label-studio start --port 8080`.
  2. Create a project; in Labeling Setup → Code paste the output of `uv run python -m training.labelstudio config`.
  3. Settings → Cloud Storage → Add Source Storage → Local files, path `C:\Users\Eric\Desktop\Projects\SOP\data\frames` (don't sync).
  4. Import `data/labelstudio/tasks.json`.
  5. Fix the flagged frames (they come first), skim the rest, then Export → JSON to `data/labelstudio/export.json`.

- [ ] **Step 7: Import** — `uv run python -m training.labelstudio import --export data/labelstudio/export.json` → prints counts; `data/yolo/dataset.yaml` exists.

- [ ] **Step 8: Commit** — `git commit -m "feat: Label Studio export/import round trip"`.

---

### Task 10: Detector training and `YoloDetector`

**Files:**
- Create: `training/train.py`, `src/sopvision/perception/{__init__,detector}.py`, `tests/perception/test_detector.py`

**Interfaces:**
- Consumes: `Frame` (Task 6), `SopConfig.detector` (Task 4), `data/yolo/dataset.yaml` (Task 9).
- Produces: `Detection(cls: str, xyxy: tuple[float, float, float, float], conf: float, track_id: int | None)`; `Detector` Protocol `detect(frame: Frame) -> list[Detection]`, `reset() -> None`; `YoloDetector(weights: Path, classes: list[str], imgsz: int = 1280, tracker: str = "bytetrack.yaml", model_factory=ultralytics.YOLO)` using `model.track(frame.image, persist=True, imgsz=..., tracker=..., verbose=False)`; raises `ConfigError("Model classes [...] don't match SOP detector.classes [...]")` when `list(model.names.values()) != classes`. Weights at `models/server_fan/best.pt`.

- [ ] **Step 1: Write the failing tests**

```python
def test_class_mismatch_is_rejected():
    fake = FakeYolo(names={0: "chassis", 1: "fan"})
    with pytest.raises(ConfigError, match="don't match SOP detector.classes"):
        YoloDetector(Path("w.pt"), CLASSES, model_factory=lambda _: fake)

def test_results_become_detections():
    fake = FakeYolo(names=dict(enumerate(CLASSES)), boxes=[([10, 20, 30, 40], 1, 0.9, 7), ([0, 0, 5, 5], 6, 0.8, None)])
    dets = YoloDetector(Path("w.pt"), CLASSES, model_factory=lambda _: fake).detect(Frame(0.0, 0, np.zeros((8, 8, 3), np.uint8)))
    assert dets == [Detection("fan", (10, 20, 30, 40), 0.9, 7), Detection("hand", (0, 0, 5, 5), 0.8, None)]

@pytest.mark.gpu
def test_real_weights_find_chassis(): ...  # first frame of raw Install_12 → at least one "chassis"
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement `detector.py`.** **Step 4: Run tests** → PASS.

- [ ] **Step 5: Implement `training/train.py`** — `train` subcommand: `--data`, `--model` (default `yolo26m.pt`; if the installed Ultralytics can't load it, use `yolo11m.pt` and note it), `--imgsz 1280`, `--epochs 100`, `--out models/server_fan`; copies `best.pt` there and writes `metrics.json` (`map50`, `map50_95` on val). `bench` subcommand: `--weights`, `--video`, runs `track` on 300 frames and prints fps.

- [ ] **Step 6: Train and benchmark**

Run: `uv run python -m training.train train --data data/yolo/dataset.yaml` then `uv run python -m training.train bench --weights models/server_fan/best.pt --video <SOP_DATA_ROOT>/raw/Install_12.MP4`
Expected: `models/server_fan/best.pt` exists; bench prints fps. Record mAP and fps in `phase0.md`. (The s-vs-m decision is made in Task 18, once the full pipeline's fps is known.)

- [ ] **Step 7: Run the gpu test** — `uv run pytest -m gpu tests/perception -v` → PASS.

- [ ] **Step 8: Commit** — `git commit -m "feat: YOLO training script and detector"` (not the weights).

---

### Task 11: Scene geometry

**Files:**
- Create: `src/sopvision/scene/{__init__,geometry}.py`, `tests/helpers.py`, `tests/scene/test_geometry.py`

**Interfaces:**
- Consumes: `Detection` (Task 10), `Layout`/`SopConfig` (Task 4).
- Produces:
  - `AnchorState(box: tuple | None, visible: bool, lost_event: bool, found_event: bool)`; `ChassisAnchor(alpha: float = 0.1, lost_after: float = 3.0)` with `update(t: float, box: tuple | None) -> AnchorState` — first box initialises; later boxes EMA-blend; `lost_event` fires once when no box for ≥ 3.0 s; `found_event` fires on the first box after a loss.
  - `to_norm(xy, chassis_box) -> tuple[float, float]`; `box_distance_norm(uv, box_xyxy, chassis_box) -> float` (0 inside; distance in chassis-normalised units, x by chassis width and y by height).
  - `match_slots(dets: list[Detection], sop: SopConfig, chassis_box) -> dict[str, Detection]` — Hungarian (`scipy.optimize.linear_sum_assignment`) on normalised centre distance between class-compatible pairs (`filled_by` or `empty_as`), keeping pairs within the slot's radius.
  - `tests/helpers.py`: `CHASSIS = (100.0, 100.0, 1100.0, 600.0)`; `det_at(cls, u, v, size=40, conf=0.9) -> Detection` placing a box centred at normalised `(u, v)` of `CHASSIS`.

- [ ] **Step 1: Write the failing tests**

```python
def test_single_outlier_moves_anchor_little():
    a = ChassisAnchor(); a.update(0.0, CHASSIS)
    s = a.update(0.033, (400.0, 100.0, 1400.0, 600.0))               # one-frame 30 % jump
    assert abs(s.box[0] - 100.0) <= 0.031 * 1000

def test_lost_fires_once_after_three_seconds():
    a = ChassisAnchor(); a.update(0.0, CHASSIS)
    assert not a.update(2.9, None).lost_event
    assert a.update(3.0, None).lost_event and not a.update(3.5, None).lost_event
    assert a.update(4.0, CHASSIS).found_event

def test_matching_survives_three_degree_rotation(sop):
    dets = [det_at("fan", *rotate(sop.layout.slots[f"fan_bay_{i}"].at, deg=3)) for i in range(1, 7)]
    assert {k: v for k, v in match_slots(dets, sop, CHASSIS).items()} == {f"fan_bay_{i}": d for i, d in zip(range(1, 7), dets)}

def test_out_of_radius_and_competition(sop): ...   # det 0.2 away → unmatched; two fans near fan_bay_1 → only nearer matched
def test_class_must_be_compatible(sop): ...        # psu at a fan bay centre → not matched
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: chassis anchor and slot matching"`.

---

### Task 12: `SceneBuilder`

**Files:**
- Create: `src/sopvision/scene/types.py`, `src/sopvision/scene/builder.py`, `tests/scene/test_builder.py`

**Interfaces:**
- Consumes: `ChassisAnchor`, `match_slots`, `box_distance_norm`, `to_norm` (Task 11).
- Produces:
  - `SlotView(state: Literal["installed","empty","occluded","unknown"], conf: float | None = None, last: Literal["installed","empty"] | None = None)` with `effective() -> Literal["installed","empty"] | None` (state if observed, else `last`).
  - `HandView(near: str)`.
  - `SceneState(version: int, t: float, chassis_visible: bool, chassis_conf: float | None, slots: dict[str, SlotView], hands: list[HandView], held: list[str], chassis_box: tuple | None)` with `to_jev_dict() -> dict` in exactly the §6.3 shape (no `chassis_box`).
  - `SceneEvent(t: float, kind: Literal["slot_changed","hand_entered","hand_left","held_changed","chassis_lost","chassis_found"], slot: str | None = None, from_state: str | None = None, to_state: str | None = None, zone: str | None = None, items: list[str] | None = None)` with `describe() -> str` (e.g. `"31.2s fan_bay_1 empty→installed"`, `"33.0s hand entered fan_bay_2"`).
  - `SceneBuilder(sop: SopConfig, settle_seconds: float = 0.5)` with `update(t: float, detections: list[Detection]) -> tuple[SceneState, list[SceneEvent]]` (version +1 per call) and `last_detections`.
- Rules (spec §6.2): observed reading = matched class; a new observed reading must persist ≥ 0.5 s before the slot changes (then `slot_changed`); no match + a hand box within the slot's radius (`box_distance_norm ≤ r`) → `occluded` keeping `last`; no match + no hand → `unknown` keeping `last`, except slots without `empty_as` read observed `empty`; hand zone = among slots with `box_distance_norm ≤ 2r`, the one whose centre is nearest the hand box centre, else `"chassis"` if the hand box intersects the chassis box, else `"outside"`; held = sorted unique classes of unmatched `fan`/`psu`/`cover` detections within normalised box distance 0.02 of a hand box; chassis not visible → every slot becomes `unknown` keeping `last`, matching skipped.

- [ ] **Step 1: Write the failing tests** (built with `det_at` and slot centres from `sop`):

```python
def test_reading_must_settle_half_a_second(sop):
    b = SceneBuilder(sop); fan1 = det_at("fan", *sop.layout.slots["fan_bay_1"].at)
    b.update(0.0, [chassis(), empty_at("fan_bay_1")]); b.update(0.6, [chassis(), empty_at("fan_bay_1")])
    s, ev = b.update(1.0, [chassis(), fan1]); assert s.slots["fan_bay_1"].state == "empty" and ev == []
    s, ev = b.update(1.5, [chassis(), fan1]); assert s.slots["fan_bay_1"].state == "installed"
    assert ev[0].describe() == "1.5s fan_bay_1 empty→installed"

def test_hand_occludes_and_keeps_last(sop): ...           # hand over fan_bay_1, no fan/empty detection → occluded, last="empty", effective()=="empty"
def test_hidden_without_hand_is_unknown(sop): ...         # fan_bay_1 installed, then cover on, no fan detections → unknown, last="installed"
def test_slot_without_empty_as_reads_empty(sop): ...       # no cover detection, no hand, 0.5 s → cover "empty"
def test_hand_zone_events(sop): ...                       # hand near psu_bay_1 → hands==[HandView("psu_bay_1")], event hand_entered psu_bay_1; hand gone → hand_left
def test_held_part(sop): ...                              # fan at (0.8, 0.9) touching a hand → held == ["fan"], held_changed event
def test_chassis_lost_makes_slots_unknown(sop): ...        # no chassis 3 s → chassis_lost event once; slots unknown with last kept
def test_version_increments_and_jev_dict_shape(sop): ...   # versions 1,2,3; to_jev_dict() keys == {"version","t","chassis","slots","hands","held"}
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: SceneBuilder with settling, occlusion, hands and held parts"`.

---

### Task 13: Calibrate tool

**Files:**
- Create: `src/sopvision/calibrate.py`, `tests/test_calibrate.py`
- Modify: `src/sopvision/cli.py` (`calibrate` command), `sops/server_fan.yaml` (real coordinates)

**Interfaces:**
- Consumes: `RunIndex` (Task 5), `YoloDetector` (Task 10), `ChassisAnchor`/`to_norm` (Task 11).
- Produces: `CalTarget(slot: str, step_id: str, video: Path, t: float)`; `calibration_targets(sop, run: RunIndex) -> list[CalTarget]` — for each slot, the step whose `evidence` names it; `t` = that step's segment end − 0.2 s in the raw video (the part was just installed); `write_layout(path: Path, coords: dict[str, tuple[float, float]]) -> None` (ruamel round-trip, keeps comments and other keys, rounds to 3 decimals). CLI: `sopvision calibrate --sop sops/server_fan.yaml --run 4` opens an OpenCV window per target with the detected chassis drawn and the prompt `Click the centre of fan_bay_1 (installed in step fan_1)`; Esc aborts without writing. `--check --run 9` draws the configured slot centres on a mid-run frame of a rotated run and saves `data/calibrate_check.jpg`.

- [ ] **Step 1: Write the failing tests**

```python
def test_targets_use_end_of_installing_step(sop):
    run = fake_run_from_seq([10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], durations=[5] * 11)
    t = {c.slot: c for c in calibration_targets(sop, run)}
    assert t["fan_bay_1"].step_id == "fan_1" and t["fan_bay_1"].t == pytest.approx(9.8)
    assert t["cover"].step_id == "cover" and t["cover"].t == pytest.approx(49.8)

def test_write_layout_preserves_comments(tmp_path):
    p = tmp_path / "s.yaml"; shutil.copy("sops/server_fan.yaml", p)
    write_layout(p, {"fan_bay_1": (0.4567, 0.7891)})
    text = p.read_text(); assert "# slot coords are normalised" in text and "[0.457, 0.789]" in text
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: 👤 Calibrate** — user runs `uv run sopvision calibrate --sop sops/server_fan.yaml --run 4` and clicks 9 centres.
- [ ] **Step 6: Check on a rotated run** — `uv run sopvision calibrate --sop sops/server_fan.yaml --check --run 9`; the executor views `data/calibrate_check.jpg` with the user and confirms every centre lands on its bay.
- [ ] **Step 7: Commit** — `git commit -m "feat: calibrate tool and calibrated server_fan layout"`.

---

### Task 14: Jev client and question builder

**Files:**
- Create: `src/sopvision/decide/{__init__,types,questions,jev_client}.py`, `tests/decide/test_questions.py`, `tests/decide/test_jev_client.py`

**Interfaces:**
- Consumes: `SopConfig` (Task 4), `SceneState`/`SceneEvent` (Task 12), `phase0.md` + `jev_response.json` (Task 2).
- Produces:
  - `QuestionSet(current_step_options: dict[str, str], active_step_id: str, active_step_question: str)`; `build_questions(sop, active_step_id) -> QuestionSet`.
  - `build_state(sop, statuses: dict[str, str], scene: SceneState, recent: list[SceneEvent]) -> dict` → `{"sop": sop.description, "steps": [{"id", "status"}], "scene": scene.to_jev_dict(), "recent_events": [e.describe() for e in recent[-10:]]}`.
  - `JevAnswers(current_step: str, current_confidence: float, current_probabilities: dict[str, float], active_step_done: float, anomaly: float, latency_ms: float)`.
  - `Decision(version: int, t_request: float, active_step_id: str, answers: JevAnswers)` (in `decide/types.py`).
  - `JevUnavailableError`; `JevClient(api_key: str, sdk=None, timeout: float = 1.5, max_retries: int = 2, sleep=asyncio.sleep)` with `async ask(state: dict, qs: QuestionSet) -> JevAnswers`. Retries on HTTP 429/5xx and timeouts with delays 0.25 s then 0.5 s; then `JevUnavailableError`. A blocking SDK call runs via `asyncio.to_thread` inside `asyncio.wait_for(timeout)`.
- Exact question copy (spec §7.3): `current_step` → `"Which step is the operator performing right now?"`, options = each step id → its `text`, `"idle"` → `sop.idle`, `"off_sop"` → `sop.off_sop`; `active_step_done` → `"Has this step been completed: " + text.rstrip(".") + "?"`; `anomaly` → `"Is anything in the scene inconsistent with the SOP so far?"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_options_and_copy(sop):
    qs = build_questions(sop, "fan_1")
    assert list(qs.current_step_options)[-2:] == ["idle", "off_sop"] and len(qs.current_step_options) == 11
    assert qs.current_step_options["idle"] == "No one is working on the server"
    assert qs.active_step_question == "Has this step been completed: Install the first fan by connecting the connector, then pressing the fan in place?"

def test_state_keeps_last_ten_events(sop, scene):
    st = build_state(sop, {"fan_1": "done"}, scene, [event(i) for i in range(15)])
    assert len(st["recent_events"]) == 10 and st["recent_events"][0].startswith("5.0s")
    assert "image" not in json.dumps(st)

async def test_recorded_response_maps_fields():           # Review Focus 1
    sdk = ReplaySdk("tests/decide/fixtures/jev_response.json")
    a = await JevClient("k", sdk=sdk).ask({}, qs)
    assert a.current_step == sdk.recorded_choice and a.current_confidence == sdk.recorded_confidence
    assert a.active_step_done == sdk.recorded_noul_done and 0.0 <= a.anomaly <= 1.0

async def test_retries_429_then_succeeds():
    sdk = FlakySdk(errors=[429, 429]); sleeps = []
    async def rec(s): sleeps.append(s)
    await JevClient("k", sdk=sdk, sleep=rec).ask({}, qs)
    assert sdk.calls == 3 and sleeps == [0.25, 0.5]

async def test_gives_up_after_retries():
    with pytest.raises(JevUnavailableError): await JevClient("k", sdk=FlakySdk(errors=[500, 500, 500])).ask({}, qs)
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement** (SDK names from `phase0.md`). **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: Jev client wrapper and config-generated questions"`.

---

### Task 15: `JevDecider`

**Files:**
- Create: `src/sopvision/decide/decider.py`, `tests/decide/test_decider.py`

**Interfaces:**
- Consumes: `JevClient`, `build_questions`, `build_state`, `Decision` (Task 14); `SceneState`, `SceneEvent` (Task 12).
- Produces: `JevDecider(sop, client, mode: Literal["live","lockstep"], on_decision: Callable[[Decision], None], on_status: Callable[[Literal["online","offline"], float], None], throttle: float = 0.25, heartbeat: float = 2.0, lockstep_latency: float = 0.5)` with:
  - `observe(scene: SceneState, events: list[SceneEvent], statuses: dict[str, str], active_step_id: str) -> None` — stores the latest inputs and appends events to the unsent list.
  - `async poll(now: float) -> None` — makes a call when (unsent events and `now - last_call ≥ throttle`) or (hands visible and `now - last_call ≥ heartbeat`), with `last_call` initially −∞; live mode starts the call as a task (at most one in flight; skip while one is running); lockstep mode awaits the call and queues the decision for `t_request + lockstep_latency`, then delivers every queued decision with due time ≤ `now`.
  - `async aclose() -> None` — cancels the in-flight call; nothing is delivered after it returns.
  - A decision whose `version` is lower than the last delivered one is dropped. 3 consecutive `JevUnavailableError`s → `on_status("offline", now)` once; the next success → `on_status("online", now)`.

- [ ] **Step 1: Write the failing tests** (`FakeClient` returns canned answers, records call times, optional delay):

```python
async def test_throttles_event_bursts(sop):
    d, client = make(sop, "lockstep")
    for i in range(21):
        t = i * 0.05; d.observe(scene(t, hands=0), [ev(t)], {}, "fan_1"); await d.poll(t)
    assert client.call_times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

async def test_heartbeat_while_hands_visible(sop): ...   # no events, hands=1, poll every 0.1 s to 6.0 → calls at 0, 2, 4, 6
async def test_no_calls_when_idle(sop): ...              # no events, no hands, 6 s → 0 calls
async def test_lockstep_delivers_after_latency(sop): ... # call at 1.0 → not delivered at poll(1.49), delivered at poll(1.5) with t_request 1.0
async def test_offline_then_online(sop): ...             # 3 failures → statuses [("offline", t)]; next success → ("online", t) appended
async def test_stale_version_dropped(sop): ...           # deliver v7 then v5 → only v7 reaches on_decision
async def test_nothing_after_close(sop):                 # Review Focus 4
    d, client = make(sop, "live", delay=0.2); d.observe(scene(0, hands=1), [], {}, "fan_1"); await d.poll(0)
    await d.aclose(); await asyncio.sleep(0.3); assert d.delivered == []
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: JevDecider with throttle, heartbeat, lockstep and offline handling"`.

---

### Task 16: `SopTracker`

**Files:**
- Create: `src/sopvision/sop/{__init__,types,tracker}.py`, `tests/sop/test_tracker.py`

**Interfaces:**
- Consumes: `SopConfig` (Task 4), `SceneState` (Task 12), `Decision` (Task 14).
- Produces:
  - `StepState(id: str, status: Literal["pending","active","done"], started_t: float | None, done_t: float | None, duration: float | None, source: Literal["jev","evidence"] | None)`.
  - `StepUpdate(t: float, step_id: str, status: str, source: str | None)`.
  - `Alert(t: float, level: Literal["info","warning","violation"], kind: Literal["order_violation","part_removed","off_sop","anomaly","sop_incomplete","jev_offline","chassis_lost","source_lost"], message: str, step_id: str | None = None, data: dict = {})`.
  - `SopTracker(sop)`: `on_decision(d: Decision, scene: SceneState) -> list[StepUpdate | Alert]`; `on_scene(scene: SceneState) -> list[StepUpdate | Alert]`; `set_jev_online(online: bool, t: float) -> list[Alert]`; `end_run(t: float) -> list[Alert]`; `active_step_id() -> str` (latest activated, else first pending); `statuses() -> dict[str, str]`; `steps() -> list[StepState]`.
- Rule precision (spec §8): a condition "holds" when it is true on consecutive decisions and `latest.t_request − first_true.t_request ≥ hold_seconds`; a false decision resets the streak. Activating a step returns a previously active, not-done step to `pending`. Jev completion needs `active_step_done ≥ 0.8` held on decisions for that same `active_step_id`, and every evidence slot's `effective()` equal to the required value at that moment. Evidence-only completion: a not-done step whose evidence `effective()` values have all held continuously for `evidence_only_after` (3.0 s) of scene time → done with `source="evidence"` (also used while Jev is offline). Part removed: a done step's evidence slot **observed** (`installed`/`empty`, not occluded/unknown) contradicting for ≥ 1.0 s. Off-SOP and anomaly alerts fire once per streak.
- Exact messages: `"{step} completed before {missing joined by ', '}"`; `"{step}: part removed after the step was completed"`; `"Off-SOP activity"`; `"Something looks inconsistent with the SOP"` (data `{"scene": scene.to_jev_dict()}`); `"SOP incomplete: missing {ids joined by ', '}"`; `"Jev offline — using evidence only"`.

- [ ] **Step 1: Write the failing tests** — one per rule, using `decision(t, current, conf, done, anomaly, active)` and `scene_with(t, {"fan_bay_1": "installed", ...})` helpers:

```python
def test_activate_needs_hold(sop):
    tr = SopTracker(sop)
    tr.on_decision(decision(0.0, "fan_1", 0.9), s0); assert tr.statuses()["fan_1"] == "pending"
    tr.on_decision(decision(1.0, "fan_1", 0.9), s0); assert tr.statuses()["fan_1"] == "active"

def test_complete_requires_jev_and_evidence(sop): ...    # done ≥0.8 held but fan_bay_1 empty → not done; installed → done, source "jev", duration = done_t − started_t
def test_evidence_only_after_three_seconds(sop): ...     # fan_bay_1 installed from t=10; on_scene at 12.9 → pending; 13.0 → done, source "evidence"
def test_order_violation_strict(sop): ...                # fan_2 completes while fan_1 pending → Alert(kind="order_violation", message="fan_2 completed before fan_1")
def test_order_modes_any_and_groups(sop): ...            # any → no alert; groups → psu_1 before fan_6 alerts, fan_3 before fan_2 doesn't
def test_part_removed_needs_observed_empty(sop): ...     # done fan_3; occluded 5 s → nothing; observed empty 1.0 s → part_removed, fan_3 back to pending
def test_off_sop_and_anomaly_once_per_streak(sop): ...
def test_incomplete_lists_missing(sop):
    assert SopTracker(sop).end_run(60.0)[0].message == "SOP incomplete: missing fan_1, fan_2, fan_3, fan_4, fan_5, fan_6, psu_1, psu_2, cover"
def test_jev_offline_alert(sop): ...                     # set_jev_online(False, 5.0) → warning "Jev offline — using evidence only"
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement** (pure, no I/O, no clock reads). **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: SopTracker rules, order modes and alerts"`.

---

### Task 17: Run log and reports

**Files:**
- Create: `src/sopvision/storage/{__init__,runlog,report}.py`, `tests/storage/test_runlog.py`, `tests/storage/test_report.py`

**Interfaces:**
- Consumes: `Alert`, `StepUpdate` (Task 16).
- Produces: `RunEvent(run_id: str, t: float, type: Literal["run_started","scene_event","decision","step_update","alert","run_summary"], payload: dict)`; `RunLog(path: Path)` with `write(event)` (append one JSON line, flush), `close()`, `RunLogClosedError` on write after close; `read_events(path) -> list[RunEvent]`; `RunReport(run_id, sop, source, total_seconds, steps: list[dict], deviations: list[dict])`; `build_report(events) -> RunReport`; `report_csv(report) -> str` with header `row_type,id,t,status,duration_s,message` and rows `step` (one per step), `alert` (one per warning/violation), `total`.

- [ ] **Step 1: Write the failing tests** — round trip; write-after-close raises; report from synthetic events has `steps[0] == {"id": "fan_1", "status": "done", "duration": 6.4, "completed_by": "jev"}` and one deviation; CSV first line equals the header exactly and the last row starts with `total,`.
- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: JSONL run log and run reports"`.

---

### Task 18: Pipeline, overlays and frame encoding

**Files:**
- Create: `src/sopvision/messages.py`, `src/sopvision/scene/overlay.py`, `src/sopvision/server/{__init__,frames}.py`, `src/sopvision/pipeline.py`, `tests/scene/test_overlay.py`, `tests/server/test_frames.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything in Tasks 6, 10, 12, 14–17.
- Produces:
  - `messages.py` (pydantic, `type` literal discriminator): `SceneMsg(t, scene: dict)`, `DecisionMsg(t, current_step, confidence, active_step_id, active_step_done, anomaly, latency_ms)`, `StepUpdateMsg(t, step_id, status, source, duration)`, `AlertMsg(t, level, kind, message, step_id)`, `RunSummaryMsg(t, report: dict)`; `to_ws(msg) -> dict`. Scene messages are throttled to 10 per second.
  - `OverlayStats(fps: float, det_ms: float, jev_ms: float | None)`; `draw_overlay(image, scene, detections, sop, stats) -> np.ndarray` — chassis box dashed white; slots as circles at their centres: installed green `(0, 200, 0)`, empty grey `(160, 160, 160)`, occluded/unknown amber `(0, 191, 255)` (BGR); hands blue boxes; labels and a fps/latency footer scaled to image height.
  - `encode_jpeg(image, max_size=(1280, 720), quality=80) -> bytes` (fit inside, preserve aspect, never upscale); `FrameBuffer` with `put(jpeg: bytes)` and `async next(after_seq: int) -> tuple[int, bytes]`.
  - `RunContext(run_id: str, sop: SopConfig, source_spec: SourceSpec, runlog: RunLog, broadcast: Callable[[dict], None], frames: FrameBuffer)`.
  - `LiveRun(ctx, source: VideoSource, detector: Detector, client: JevClient)` with `async start()`, `async stop()` (idempotent: closes decider first, then `end_run`, writes `run_summary`, closes the log), `pause()`, `resume()`, `set_speed(x)`, `done: asyncio.Event`. The pipeline thread reads frames via `LatestFrameReader`, runs detect → `SceneBuilder` → overlay → `FrameBuffer.put`, and hands `(scene, events)` to the event loop with `loop.call_soon_threadsafe`; the loop task runs tracker, decider (`mode="live"`, polled every 50 ms), run log and broadcast. `chassis_lost` events become `Alert(kind="chassis_lost", level="warning", message="Chassis not visible")`; the source's `on_lost` becomes `Alert(kind="source_lost", level="warning", message="Video source lost — reconnecting")`.
  - `FrameRecord(t: float, slots: dict[str, str])`; `LockstepResult(frames: list[FrameRecord], steps: list[StepState], alerts: list[Alert], latencies_ms: list[float])`; `async run_lockstep(sop, source: FileSource, detector, client, runlog: RunLog | None = None) -> LockstepResult` — every frame processed in order; tracker and decider (`mode="lockstep"`) use the frame time as the clock; ends with `end_run(last_t)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_installed_slot_drawn_green(sop):
    img = draw_overlay(np.zeros((1080, 1920, 3), np.uint8), scene_with_installed("fan_bay_1"), [], sop, OverlayStats(30, 6, 140))
    u, v = sop.layout.slots["fan_bay_1"].at; x, y = slot_pixel(u, v)
    assert tuple(img[y, x]) == (0, 200, 0)

@pytest.mark.parametrize("w,h,ew,eh", [(1920, 1080, 1280, 720), (640, 480, 640, 480), (1280, 960, 960, 720)])
def test_encode_fits_without_upscaling(w, h, ew, eh):     # Review Focus 5
    out = cv2.imdecode(np.frombuffer(encode_jpeg(np.zeros((h, w, 3), np.uint8)), np.uint8), 1)
    assert out.shape[:2] == (eh, ew)

async def test_lockstep_completes_scripted_run(sop, tmp_path):
    # ScriptedDetector fills fan bays 1..6, psu bays, then cover at known times; OracleClient answers from the scene
    res = await run_lockstep(sop, scripted_source(seconds=40), ScriptedDetector(sop), OracleClient(sop), RunLog(tmp_path / "r.jsonl"))
    assert [s.status for s in res.steps] == ["done"] * 9 and not [a for a in res.alerts if a.level == "violation"]
    assert len(res.frames) == 40 * 30

async def test_live_drops_frames_under_slow_detector(sop): ...  # 30 fps source, detector sleeps 50 ms → reader.dropped > 0, processed t strictly increasing
async def test_stop_during_inflight_call_writes_nothing_after_summary(sop, tmp_path):   # Review Focus 4
    run = live_run(sop, tmp_path, client=SlowClient(delay=1.0)); await run.start(); await asyncio.sleep(0.3); await run.stop()
    types = [e.type for e in read_events(tmp_path / "run.jsonl")]
    assert types[-1] == "run_summary" and types.count("run_summary") == 1
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.

- [ ] **Step 5: Measure real fps and choose model size** — write `spikes/bench_pipeline.py`: runs `LiveRun` on raw Install_12 for 30 s with a no-op broadcast and the real Jev client, then prints processed frames per second. Run it and record the result in `phase0.md`. If below 30 fps with the `m` model, retrain with `--model yolo26s.pt` (or `yolo11s.pt`, matching Task 10's family) and re-measure.

- [ ] **Step 6: Commit** — `git commit -m "feat: live and lockstep pipeline with overlays and MJPEG frames"`.

---

### Task 19: Server API, WebSocket and MJPEG

**Files:**
- Create: `src/sopvision/server/app.py`, `src/sopvision/server/runs.py`, `tests/server/test_app.py`, `scripts/docker_smoke.py`
- Modify: `src/sopvision/cli.py` (`serve --host --port 8000` and, in Task 22, `eval` both call `startup_checks` first and exit 1 printing its message)

**Interfaces:**
- Consumes: `LiveRun`, `RunContext`, `FrameBuffer` (Task 18); `list_sources`/`open_source` (Task 6); `list_sops` (Task 4); `build_report`/`report_csv` (Task 17).
- Produces: `startup_checks(settings, gpu: GpuReport) -> list[SopConfig]` in `server/runs.py` — in order: `require_gpu(settings.sopvision_in_container, gpu)`; `settings.check_data_root()`; `list_sops(settings.sops_dir)` (a `ConfigError` propagates with its field message); every SOP's `detector.weights` exists, else `ConfigError("Weights not found: models/server_fan/best.pt. Train the detector (training/train.py) or fix detector.weights.")`; the API key is non-empty, else `ConfigError("TYPESAFE_API_KEY is not set. Add it to .env.")`. Returns the loaded SOPs.
- Produces: `RunManager(settings, run_factory)` (one active run; `start(sop_id, source_id) -> RunInfo`, `stop(run_id)`, `current() -> RunSnapshot | None`, `control(run_id, action, speed)`); `create_app(settings, manager) -> FastAPI` with:
  - `GET /sops` → `[{id, version, description, steps: [{id, label, text}]}]`
  - `GET /sources` → `[SourceSpec]`
  - `POST /runs` body `{sop: str, source_id: str}` → `201 RunInfo{run_id, sop, source}`; `409` `{"detail": "A run is already in progress. Stop it first."}`; `400` with the `SourceUnavailableError` message.
  - `POST /runs/{id}/stop` → `200`; `POST /runs/{id}/control` body `{action: "pause"|"resume", speed?: 1|2}` → `200` (file sources only, else `400 "Only recorded videos can be paused."`).
  - `GET /runs/current` → `RunSnapshot{run_id, sop, source, steps: [StepState], alerts: [Alert], last_decision}` or `204`.
  - `GET /runs/{id}/report?format=json|csv` (CSV `text/csv`).
  - `GET /video.mjpg` (`multipart/x-mixed-replace; boundary=frame`, ~20 fps from `FrameBuffer`).
  - `WS /ws` → every message from `messages.py`.
  - Static files from `web/dist` at `/` when the folder exists.

- [ ] **Step 1: Write the failing tests** (TestClient; `FakeRunFactory` produces runs that emit a scripted message list):

```python
def test_sources_hide_webcam_in_container(client_in_container):
    assert all(s["kind"] != "webcam" for s in client_in_container.get("/sources").json())

def test_webcam_rejected_in_container(client_in_container):
    r = client_in_container.post("/runs", json={"sop": "server_fan_install", "source_id": "webcam:0"})
    assert r.status_code == 400 and r.json()["detail"] == "Webcam isn't available in the container — run natively or use an RTSP source."

def test_one_run_at_a_time(client): ...           # second POST → 409 with exact detail
def test_ws_streams_step_updates(client): ...     # start fake run; ws receives {"type": "step_update", "step_id": "fan_1", ...}
def test_current_snapshot_and_report_csv(client): ...  # /runs/current 200 with steps; after stop, report?format=csv → text/csv, first line is the CSV header
def test_pause_rejected_for_rtsp(client): ...

@pytest.mark.parametrize("break_it, message", [
    (lambda s, root: s.__setattr__("typesafe_api_key", SecretStr("")), "TYPESAFE_API_KEY is not set. Add it to .env."),
    (lambda s, root: (root / "models/server_fan/best.pt").unlink(), "Weights not found: models/server_fan/best.pt"),
])
def test_startup_checks_name_the_fix(tmp_repo, break_it, message): ...   # tmp_repo has sops/, fake weights, data root; break one thing → ConfigError(match=message)
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.

- [ ] **Step 5: Native end-to-end check** — `uv run sopvision serve`; `curl -s -X POST localhost:8000/runs -H "content-type: application/json" -d '{"sop":"server_fan_install","source_id":"test:Install_12"}'`; open `http://localhost:8000/video.mjpg` in the browser pane and confirm annotated video plays. Stop the run.

- [ ] **Step 6: Docker smoke test** — implement `scripts/docker_smoke.py` (`docker compose up -d --build`, poll `GET /sops` and `GET /sources` for up to 120 s, assert no webcam source, `docker compose down`). Run: `uv run python scripts/docker_smoke.py` → `docker smoke: OK`.

- [ ] **Step 7: Commit** — `git commit -m "feat: FastAPI server with runs, WebSocket, MJPEG and Docker smoke test"`.

---

### Task 20: Dashboard

**Files:**
- Create: `web/package.json`, `web/vite.config.ts` (dev proxy `/sops /sources /runs /video.mjpg /ws` → `localhost:8000`), `web/tsconfig.json`, `web/index.html`, `web/src/{main.tsx,App.tsx,types.ts,state.ts,api.ts,styles.css}`, `web/src/components/{TopBar,VideoPanel,Checklist,JevStrip,Timeline,EventLog,Controls}.tsx`, `web/src/state.test.ts`, `web/src/App.test.tsx`, `web/src/fixtures/run_stream.json`
- Modify: `Dockerfile` (add `FROM node:24-slim AS web` stage: `npm ci`, `npm run build`; copy `web/dist` into `/app/web/dist`)

**Interfaces:**
- Consumes: the HTTP/WS API (Task 19); message shapes in `messages.py` (Task 18) mirrored in `web/src/types.ts`.
- Produces: `reduce(state: DashboardState, msg: WsMessage) -> DashboardState` in `state.ts`; `DashboardState = {steps: StepView[], alerts: AlertView[], decision?: DecisionView, status: "On track" | "Deviation" | "Incomplete", elapsed: number, timeline: Segment[], jevOffline: boolean, chassisVisible: boolean}`. Status rule: any violation alert → `"Deviation"`; a `run_summary` with any step not done → `"Incomplete"`; otherwise `"On track"`. A `jev_offline` alert sets `jevOffline`; the next `decision` message clears it (the "Jev now" strip shows a "Jev offline" badge meanwhile). `chassisVisible` follows the latest `scene` message; when false the video panel shows a "Chassis not visible" banner.
- Layout and copy follow the approved mockup (§9.2): top bar (SOP name and version, source, timer, status pill), video left (`<img src="/video.mjpg">`), checklist right (done ✓ with duration, active highlighted with Jev confidence, pending), "Jev now" strip (current step label, confidence bar, off-SOP probability, latency), timeline, event log, controls (source picker from `/sources`, play/pause, restart = stop then start the same source, 1×/2×, download run report). On load it calls `GET /runs/current` to pick up a run already in progress. Colours as CSS custom properties with a dark-mode set; sentence case throughout.

- [ ] **Step 1: Scaffold** — `npm create vite@latest web -- --template react-ts`, then `npm i -D vitest @testing-library/react jsdom` in `web/`.

- [ ] **Step 2: Write the failing tests**

```ts
test("step_update marks a step done with its duration", () => {
  const s = reduce(initial(STEPS), { type: "step_update", t: 9.1, step_id: "fan_1", status: "done", source: "jev", duration: 9.1 });
  expect(s.steps[0]).toMatchObject({ id: "fan_1", status: "done", duration: 9.1 });
});
test("a violation flips status to Deviation", () => { /* alert level "violation" → status "Deviation" */ });
test("run_summary with missing steps → Incomplete", () => { /* ... */ });
test("jev_offline alert sets the badge until the next decision", () => { /* alert → jevOffline true; decision → false */ });
test("scene without chassis shows the banner flag", () => { /* scene.chassis.visible false → chassisVisible false */ });
test("App renders a recorded stream", async () => {
  render(<App socket={fakeSocket(runStream)} api={fakeApi(SOPS)} />);
  expect(await screen.findByText("Fan 1")).toBeInTheDocument();
  expect(screen.getAllByRole("listitem")).toHaveLength(9);
});
```

- [ ] **Step 3: Run to verify failure** — `cd web && npx vitest run` → FAIL.
- [ ] **Step 4: Implement the reducer, API/socket wrappers (injectable for tests) and components.**
- [ ] **Step 5: Run tests and build** — `npx vitest run` → PASS; `npm run build` → `web/dist/index.html` exists.
- [ ] **Step 6: Visual and frame-rate check** (spec §2 criterion 6) — `uv run sopvision serve`, open `http://localhost:8000` in the browser pane, start `test:Install_12`, screenshot and compare against the mockup with the user. Then write `spikes/mjpeg_fps.py` (streams `GET /video.mjpg` with httpx for 10 s, counts `--frame` boundaries, prints frames per second) and run `uv run python spikes/mjpeg_fps.py` while the run plays — expected ≥ 20.
- [ ] **Step 7: Rebuild the image** — `uv run python scripts/docker_smoke.py` → OK, and `http://localhost:8000` serves the dashboard from the container.
- [ ] **Step 8: Commit** — `git commit -m "feat: React dashboard served by FastAPI and the Docker image"`.

---

### Task 21: Scenario videos

**Files:**
- Create: `src/sopvision/evaluation/{__init__,scenarios}.py`, `tests/evaluation/test_scenarios.py`
- Modify: `src/sopvision/cli.py` (`make-scenarios`)

**Interfaces:**
- Consumes: `RunIndex` (Task 5).
- Produces: `ScenarioSpec(name: str, positions: list[int], reverse_last: bool, expected_alerts: list[dict])`; `SCENARIOS = [ScenarioSpec("incomplete_run", [1, 2, 3, 4, 5], False, [{"kind": "sop_incomplete", "missing": ["fan_5", "fan_6", "psu_1", "psu_2", "cover"]}]), ScenarioSpec("fan_removed", [1, 2, 3, 4, 4], True, [{"kind": "part_removed", "step_id": "fan_3"}, {"kind": "sop_incomplete"}])]` (`reverse_last` plays the last listed clip backwards); `render_scenario(run: RunIndex, spec, out_dir, freeze_seconds=3.0) -> Path` writes `<name>.mp4` (source resolution and fps, mp4v) and `<name>.json` (`{"run": 12, "expected_alerts": [...]}`). CLI: `sopvision make-scenarios --run 12 --out data/scenarios`. Reverse decoding buffers the clip as JPEG bytes and decodes in reverse.

- [ ] **Step 1: Write the failing tests** (tiny synthetic clips whose frames encode their index as a grey level):

```python
def test_reverse_and_freeze(tmp_path):
    run = tiny_run(tmp_path, clips=[5, 5, 5, 5], fps=10)                  # positions 1..4, 5 frames each
    path = render_scenario(run, ScenarioSpec("x", [1, 2, 2], True, []), tmp_path, freeze_seconds=0.3)
    levels = read_levels(path)                                            # mean grey per frame → original index
    assert levels[:10] == list(range(10)) and levels[10:15] == [9, 8, 7, 6, 5] and levels[15:] == [5, 5, 5]
    assert json.loads(path.with_suffix(".json").read_text())["expected_alerts"] == []
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.
- [ ] **Step 5: Render the real scenarios** — `uv run sopvision make-scenarios --run 12 --out data/scenarios` → two `.mp4` + two `.json`; watch `fan_removed.mp4` in the browser pane to confirm the fan comes out.
- [ ] **Step 6: Commit** — `git commit -m "feat: simulated-mistake scenario videos"`.

---

### Task 22: Eval harness, metrics, parity and acceptance

**Files:**
- Create: `src/sopvision/evaluation/{metrics,run_eval,compare}.py`, `tests/evaluation/test_metrics.py`, `tests/evaluation/test_compare.py`
- Modify: `src/sopvision/cli.py` (`eval --out`, `compare A B`)

**Interfaces:**
- Consumes: `run_lockstep`, `LockstepResult` (Task 18); `load_index`, `Segment` (Task 5); scenario `.json` (Task 21); `gpu_report` (Task 3).
- Produces:
  - `step_errors(segments: list[Segment], steps: list[StepState]) -> dict[str, float | None]` (done_t − segment end; `None` if not done); `offsop_flagged(segments, alerts, grace: float = 1.5) -> list[bool]` (an `off_sop` alert with `start ≤ t ≤ end + grace`); `check_acceptance(report: EvalReport) -> dict[str, bool]` for criteria `c1`…`c5` of §2 (c1: 18/18 done in order; c2: median |error| ≤ 2.0 and max ≤ 4.0; c3: all 4 off-SOP segments flagged; c4: no `order_violation`/`part_removed`/`sop_incomplete` on test videos; c5: each scenario's expected alert kinds present, with `step_id`/`missing` matching).
  - `EvalVideo(id, kind: Literal["test","scenario"], step_order: list[str], steps: dict[str, dict], alerts: list[dict], frames: list[FrameRecord], offsop: list[bool], latency_ms: dict[str, float])`; `EvalReport(env: dict, videos: list[EvalVideo], acceptance: dict[str, bool], passed: bool)` (pydantic, JSON).
  - `run_eval(settings, out: Path) -> EvalReport` — Install_12 and Install_13 raw videos plus every `scenarios_dir/*.mp4`; the CLI runs `startup_checks` (Task 19) first and exits 1 when `passed` is false.
  - `compare_reports(a: EvalReport, b: EvalReport) -> list[str]` — §2 criterion 7: both passed; per video identical `step_order` and identical set of alert kinds; each step's `done_t` within 0.5 s; slot-state agreement over all (frame, slot) pairs ≥ 0.99. Returns mismatch messages (empty = parity). CLI exits 1 on any.

- [ ] **Step 1: Write the failing tests**

```python
def test_step_errors_and_c2():
    errs = step_errors(SEGS, steps_done_at({"fan_1": 15.5, "fan_2": 21.0}))
    assert errs["fan_1"] == pytest.approx(0.5) and errs["fan_3"] is None

def test_offsop_requires_alert_inside_segment(): ...   # alert at end+1.4 → True, end+1.6 → False
def test_acceptance_fails_on_missing_step(): ...       # report with 17/18 → c1 False, passed False
def test_parity_identical_reports_pass(): assert compare_reports(R, R) == []
def test_parity_step_time_tolerance():
    assert any("fan_2" in m for m in compare_reports(R, shift_step(R, "fan_2", 0.6)))
def test_parity_slot_agreement_threshold():
    assert any("slot agreement 98.9%" in m for m in compare_reports(R, corrupt_slots(R, fraction=0.011)))
def test_parity_alert_kinds_must_match(): ...
```

- [ ] **Step 2: Run to verify failure** → FAIL. **Step 3: Implement.** **Step 4: Run tests** → PASS.

- [ ] **Step 5: Native eval**

Run: `uv run sopvision eval --out runs/eval-native.json`
Expected: acceptance `c1`–`c5` all true and exit 0. If a criterion fails, tune only `sops/server_fan.yaml` `decisions` values or `settle_seconds`, re-run, and record each attempt and its numbers in `phase0.md`; if tuning can't meet the bar, stop and report the numbers to the user rather than relaxing the criteria.

- [ ] **Step 6: Docker eval and parity**

Run: `docker compose run --rm sopvision eval --out runs/eval-docker.json` then `uv run sopvision compare runs/eval-native.json runs/eval-docker.json`
Expected: both exit 0; compare prints `parity: OK`.

- [ ] **Step 7: Commit** — `git commit -m "feat: lockstep eval, acceptance checks and Docker parity compare"` (include the results summary in `phase0.md`; not the run JSONs).

---

### Task 23: Jev contract and scenario tests

**Files:**
- Create: `tests/jev/test_contract.py`, `tests/jev/test_scenarios.py`

**Interfaces:**
- Consumes: `JevClient`, `build_questions`, `build_state` (Task 14); `JevDecider` (Task 15); `SopTracker` (Task 16); `SceneState` helpers from `tests/helpers.py`.

- [ ] **Step 1: Write the tests** (all `@pytest.mark.jev`, real API):
  - Contract: the §6.3 scene → `ask` returns a `current_step` that is one of the 11 option keys, `0 ≤ confidence ≤ 1`, `0 ≤ active_step_done ≤ 1`, `0 ≤ anomaly ≤ 1`.
  - Scenarios: scripted `SceneState` sequences fed through `JevDecider(mode="lockstep")` → `SopTracker`:
    - `test_wrong_order`: fan_bay_2 installed before fan_bay_1 → `order_violation` with message `"fan_2 completed before fan_1"`.
    - `test_skipped_step`: fan_bay_3 never installed, fan_bay_4 installed → `order_violation` naming `fan_3`; `end_run` → `sop_incomplete` including `fan_3`.
    - `test_cover_before_psus`: cover installed with both PSU bays empty → `order_violation` naming `psu_1, psu_2`.
- [ ] **Step 2: Run** — `uv run pytest -m jev -v` → PASS (needs `.env`).
- [ ] **Step 3: Commit** — `git commit -m "test: Jev contract and synthetic violation scenarios"`.

---

### Task 24: Demo polish

**Files:**
- Modify: `README.md`
- Create: `docs/demo-script.md`

- [ ] **Step 1: Write `README.md`** — what it is (two sentences), prerequisites (uv, Python 3.12, Node, Docker Desktop with WSL2 + NVIDIA driver, `.env`), both run paths (commands from spec §13.2), how to run tests (`uv run pytest`, `-m jev`, `-m gpu`, `cd web && npx vitest run`), how to retrain and recalibrate, and the acceptance results from `phase0.md`.
- [ ] **Step 2: Write `docs/demo-script.md`** — a 5-minute walkthrough: start on `test:Install_12` at 1× (steps tick off, "Jev now" follows the operator, off-SOP flagged at the start), switch to `scenario:fan_removed` (part-removed alert), then `scenario:incomplete_run` (incomplete alert), download the run report; talking points: video stays local, cost < $0.001 per station-hour, config-only SOPs, Docker path to on-prem.
- [ ] **Step 3: Full verification** — `uv run pytest -v` → all pass; `uv run pytest -m "gpu or jev" -v` → all pass; `cd web && npx vitest run` → pass; `uv run sopvision eval --out runs/eval-native.json` → passed; `uv run python scripts/docker_smoke.py` → OK.
- [ ] **Step 4: 👤 Record the backup demo video** — the user follows `docs/demo-script.md` and records the screen (Win+Alt+R or OBS) to keep alongside the slides.
- [ ] **Step 5: Commit** — `git commit -m "docs: README run instructions and demo script"`.
