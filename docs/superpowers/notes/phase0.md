# Phase 0 findings

Running record of the spec's Phase 0 checks plus results that later tasks depend on.

## Jev SDK (Task 2, 2026-10-02)

- **Package:** `typesafe-sdk` 0.7.2 (`import typesafe_sdk`). Clients: `TypeSafeClient` and `AsyncTypeSafeClient(api_key=None, model=None, retry=RetryPolicy, timeout=...)`. Env vars read by the SDK: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`), `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`).
- **Call:** `client.system_one(state=<JSON>, questions={key: Choice(...) | Noul(...) | Score(...)})` → `SystemOneResponse(model, usage, answers)`.
  - `Choice(instructions=..., criteria={option_key: description})` → `ChoiceAnswer(type="choice", choice, confidence, probabilities)`.
  - `Noul(instructions=..., criteria={"true": ..., "false": ...} | None)` → `NoulAnswer(type="noul", noul)` — the probability is the `noul` field.
- **Errors:** `TypeSafeRateLimitError` / `TypeSafeInternalServerError` (subclasses of `TypeSafeAPIError`, with `.status`), `TypeSafeAPITimeoutError` / `TypeSafeAPIConnectionError`, `TypeSafeAuthenticationError` (401), `TypeSafePermissionDeniedError` (403). The SDK has its own `RetryPolicy` (default 2 retries); our wrapper sets `max_retries=0` and applies the spec's own retry rules.
- **Choice keys:** underscore keys (`fan_1`, `off_sop`, `idle`) work; 11 options accepted.
- **Live check:** model `jev-1.13.0`; ~1,070 input tokens per call with the §6.3 scene + §7.3 questions.
  - Latency over 20 calls from this PC: **p50 174 ms, p90 196 ms, max 375 ms** (spec target p90 ≤ 500 ms ✓).
  - Answers for the §6.3 scene: `current_step = fan_3` (0.98–0.99), `active_step_done` 0.17–0.21, `anomaly` 0.33–0.42.
  - **Not deterministic:** 17 distinct answer sets in 20 identical calls, but the variation is small — the chosen step never changed and probabilities moved by ≤ 0.04. Docker-vs-native parity therefore needs the tolerances in spec §2 criterion 7.
- Recorded response for tests: `tests/decide/fixtures/jev_response.json`.
- **Access note:** a Vercel AI Gateway key also works (`TYPESAFE_BASE_URL=https://ai-gateway.vercel.sh/typesafe`, `TYPESAFE_DEFAULT_MODEL=typesafe-ai/jev`) but needs paid Vercel credits; the project uses a direct TypeSafe key.

## Docker GPU (Task 2–3)

- `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` sees the RTX 4080 (driver 591.86, CUDA 13.1).
- `docker compose run --rm sopvision doctor`: `cuda: True`, `gpu: NVIDIA GeForce RTX 4080`, `torch: 2.11.0+cu128`, `in_container: True`.

## Training data (Task 7)

- 640 frames sampled (576 train from Install 1, 3–10; 64 val from Install 11); no test-run frames.

## Auto-labelling gate (Task 8) — failed; replaced by reference-frame labelling

Zero-shot candidates on 20 frames spread over all SOP positions (count check: expected parts per position):

| Labeller | Pass | What it finds |
|---|---|---|
| Grounding DINO base (per-class prompts from the SOP config) | 2/20 | hands; every other prompt boxes the whole server |
| YOLOE 11-L (Ultralytics, text prompts; extra candidate) | 9/20, inflated | hands, occasional whole-server box; passes came from a count-check loophole (since fixed) |
| SAM 3 (config prompts, threshold 0.5) | 0/20 | hands only |

SAM 3 short-prompt probe (threshold 0.3): no fans or power supplies under any phrasing; "empty slot" fires on 11–36 slots per frame (PCIe, memory…); "metal cover" finds the fixed right-hand lid; "server" finds the chassis weakly (0.3–0.6); hands 0.9+.

**Decision (user-approved):** reference-frame labelling. The user drew boxes once on three frames of Install_4 (start: chassis, 6 empty fan bays, 2 empty PSU bays; before cover: 6 fans, 2 PSUs; end: cover) — `data/reference/Install_4.json`. For every sampled frame:
- SIFT + RANSAC homography to the reference (800–980 inliers typical, including the rotated Install_1/9);
- bay identity from the step timeline: fan_1 = bottom bay of the lower group, filling upward, then the upper group bottom-up; PSU 1 = top-left, PSU 2 = bottom-left;
- bay state from the timeline; the in-progress bay by appearance (z-normalised crop diff against the empty/installed references, restricted to the sub-region where those differ — a PSU only changes at its handle end);
- bays matching neither reference (arms, cover) are left unlabelled; hands from SAM 3.

Known gap: parts held in a hand (fan/PSU not yet in a bay) are not labelled, so the first detector won't report `held` parts.

## Detector training (Task 10)

- YOLO26m (Ultralytics 8.4.171), imgsz 1280, batch 4 (batch 8 filled the 16 GB GPU and stalled), 2 workers; ~27–45 s/epoch on the RTX 4080.
- Stopped at epoch 51 of 100 after a plateau (mAP50 ≈ 0.99 since epoch 17). Best epoch 48 on the Install_11 validation frames: **mAP50 0.994, mAP50-95 0.975**. Optimizer stripped: `models/server_fan/best.pt` is 44 MB (gitignored).
- `sopvision calibrate --run 9` with the trained detector: every slot centre lands on its bay on the rotated recording.

## Live pipeline throughput (Task 18)

- `spikes/bench_pipeline.py` (LiveRun on raw Install_12, real Jev, 30 s after a 3 s warm-up): **29.8 fps processed** against the 30 fps source (93 frames dropped, mostly during CUDA warm-up); 32 Jev decisions, median 186 ms; fan_1 and fan_2 completed live. YOLO26m is fast enough — no need for the `s` size.
