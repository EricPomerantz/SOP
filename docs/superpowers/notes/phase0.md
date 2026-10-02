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
