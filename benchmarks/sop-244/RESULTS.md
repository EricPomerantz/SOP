# sop-244: frozen Jev vs Decider decision benchmark

244 decision requests recorded with `DECISION_MODE=compare` on the held-out recordings Install_12 and Install_13
(`sopvision eval`, lockstep), 192 with a known true step. `requests.jsonl` holds each request exactly as sent
(byte-identical, checked by `scripts/freeze_benchmark.py`), the tracker's step completion times at that moment, and the
ground truth. **Never train on it; runs 12 and 13 are excluded from all training data.**

Run an experiment (both providers get the identical variant request):

```bash
docker compose run --rm --entrypoint python sopvision scripts/bench_decisions.py --frozen benchmarks/sop-244 --variant <name>
```

Metrics are on the 192 labelled requests. *Acts* = confidence ≥ 0.8 (the tracker's threshold). *Echo* = answer equals the
step the tracker already expects next (a trivial "echo the tracker" baseline scores 97% here, so a high echo rate is not
proof of understanding; deviation cases are needed to tell). Jev varies about ±1.5 points between identical runs.

Decider-4B v2.1 (`eb5fbdfc`), `DECIDER_SHARED_MIN_TOKENS=256`, RTX 5000 Ada Laptop. Latency is per request with Jev
called concurrently.

| Experiment | Provider | Correct | Acts | Correct when acting | Echo | Off-SOP FP | p50 / p95 / max ms |
|---|---|---|---|---|---|---|---|
| 00 baseline (as recorded) | Decider | 49.5% | 39.1% | 68.0% | 43% | 12.0% | 297 / 450 / 8427¹ |
| | Jev | 72.4% | 45.8% | 83.0% | 63% | 5.2% | 159 / 219 / 288 |
| 01 A1: + `progress` (expected step, last completed + seconds ago) | Decider | 55.7% (+6.2) | 44.3% | 72.9% | 52% | 18.2% | 427 / 512 / 587² |
| | Jev | 91.7% (+19.3) | 66.7% | 97.7% | 87% | 0.5% | 160 / 217 / 452 |
| 02 A6: `steps` (statuses) moved to the end of the state | Decider | 78.6% (+29.1) | 45.8% | 92.0% | 70% | 1.6% | 293 / 443 / 11701¹ |
| | Jev | 83.3% (+10.9) | 60.9% | 94.0% | 75% | 1.0% | 160 / 222 / 633 |
| 03 A7: options sorted alphabetically | Decider | 47.4% (−2.1) | 36.5% | 67.1% | 44% | 17.7% | 354 / 504 / 607 |
| | Jev | 69.3% (−3.1) | 47.9% | 81.5% | 59% | 4.7% | 163 / 217 / 261 |
| 04 A6 + A1: `steps` then `progress` at the end | Decider | 83.9% (+34.4) | 56.8% | 96.3% | 82% | 7.8% | 315 / 466 / 624 |
| | Jev | 92.7% (+20.3) | 77.6% | 98.0% | 91% | 0.0% | 163 / 226 / 302 |
| x accidental: state keys and options sorted alphabetically³ | Decider | 78.1% | 45.8% | 95.5% | 72% | 8.3% | 290 / 297 / 554 |
| | Jev | 83.9% | 65.6% | 95.2% | 74% | 0.0% | 157 / 212 / 346 |

¹ the first requests after a long idle period (8-16 s; no kernels were compiled, so not compilation). ² thermal, not tokens:
see Latency below. ³ not a planned experiment: a bug in the first freeze sorted the JSON
keys, which put `steps` (statuses) last in the state and reordered the options. Kept because it shows prompt layout
matters a great deal; to be split into single-change experiments.

## Deviation probes (`benchmarks/probes-d1`, 140 synthetic probes; evaluation only)

The frozen set is clean, in-order work, where echoing the tracker's expected step is right 97% of the time. The probes
edit frozen states so a known deviation is happening (`scripts/make_probes.py`): P1 working on fan_{k+2} while fan_k is
expected (answer fan_{k+2}), P3 an idle bench (idle/off_sop), P2 a part installed out of order and P4 a done part removed
(anomaly should be ≥ 0.8).

| Variant | Provider | P1 out-of-order work | P3 idle bench | P2+P4 anomaly caught (mean p) | Echo |
|---|---|---|---|---|---|
| baseline | Decider | 92.5% | 42.5% | 0% (0.32) | 37% |
| | Jev | 100% | 77.5% | 32% (0.68) | 19% |
| A1 | Decider | 42.5% | 62.5% | 0% (0.27) | 56% |
| | Jev | 75.0% | 5.0% | 38% (0.75) | 67% |
| A6 | Decider | 70.0% | 15.0% | 0% (0.34) | 56% |
| | Jev | 100% | 37.5% | 2% (0.68) | 41% |
| A6 + A1 | Decider | 7.5% | 15.0% | 0% (0.30) | 89% |
| | Jev | 62.5% | 2.5% | 2% (0.68) | 81% |

The variants that raise in-order accuracy trade away deviation detection: the gains are largely the model learning to
repeat the tracker. Decider never reaches 0.8 on the anomaly question in any variant.

## Latency

Decider only (no Jev), the 244 requests run back to back: baseline p50 295 ms over the first 50 requests, 412 ms once the
GPU is hot; A6 run warm is the same as baseline warm (415 ms). The laptop GPU reports software thermal slowdown nearly the
whole time under load (SM clock ~1.1 GHz, 64 °C). Steady-state Decider latency on this machine is ~410 ms; compare
variants at the same thermal state, interleaved.

## Laya zero-shot (`convaiinnovations/laya`, laya 0.3.28, reviewed checkpoint SHAs; baseline request layout)

Laya alone on the GPU (`docker compose --profile laya`), `--providers laya`. Its tokenizer sees our state as ~505 tokens:
the shipped English budget (512 total / 192 for options) drops ~35% of it, so L1 raises the budgets. The English
checkpoint also ships an invalid temperature for choices with 11+ options (clamped to 0.5 by the runtime), and our step
question has exactly 11, so its confidences are uncalibrated.

| Config | Correct (244 set) | Acts | Correct when acting | P1 / P3 probes | Anomaly caught | p50 / p95 ms |
|---|---|---|---|---|---|---|
| L0 English, shipped (state truncated) | 7.8% | 100% | 7.8% | 0% / 0% | 0% | 48 / 64 |
| L1 English, max_len 2048 / head 512 | 7.8% | 14.6% | 10.7% | 0% / 0% | 0% | 78 / 86 |
| L2 typed-decisions (1024, fits) | 9.4% | 0% | — | 0% / 0% | 0% | 83 / 91 |

Chance on the 11-option question is 9.1%. In every configuration Laya answers `fan_1`, the first option, for 231-244 of
the 244 requests: zero-shot it does not read the state. This matches its model card ("a fast base to specialise, not a
zero-shot decision engine"); the open question is what it reaches after fine-tuning on our data.

## Laya fine-tuned on our data (`laya-sop-v1`)

`laya-train` from the English checkpoint on `data/training/sop-v1/train.jsonl` (2,059 rows, 4,633 training questions +
400 held back for calibration; `docs/decision-model-training.md`): default recipe (rlcd loss, 4 epochs, micro-batch 8 x
accumulation 8, seed 0), budgets 2048 / 512, ~28 min on the RTX 5000 Ada Laptop. Mean loss 0.90 -> 0.62. Served with
`LAYA_CHECKPOINT=/data/models/laya-sop-v1` (`docker/laya_serve.py`).

| Set | Step correct | Acts | Correct when acting | Other | p50 / p95 ms |
|---|---|---|---|---|---|
| validation (Install_11, in_order rows) | 27.0% | | | done 93% (mostly "no"), anomaly-free rows all "no" | 85 / 177 |
| frozen 244 | 34.9% | 18.2% | 100% (35/35) | off-SOP FP 27.6% | 92 / 242 |
| probes | P1 37.5%, P3 72.5% | 7.5% | 0% | anomaly caught at 0.8: 0% | 84 / 149 |

The step question is underfit, not overfit (validation is no better than test): the model only ever answers fan_4,
idle, cover, fan_5 or psu_1, never fan_1/2/3/6 or psu_2; the training set's own held-out calibration items score 31%.
The anomaly question did learn: AUROC 0.997 on probe anomalies vs clean requests, 54/60 caught with 2/284 false alarms
at 0.7, but the fitted temperature (1.89) caps it near 0.72, so nothing crosses 0.8. Those anomalies come from the same
generator as the training deviations, so this is in-distribution; real anomalies are untested.

## Laya fine-tuned, 12 epochs (`laya-sop-v1-e12`)

Identical to `laya-sop-v1` except `--epochs 12` (~84 min). Mean loss per epoch hovered at 0.6-0.8 through epoch 7, then
fell to 0.16 by epoch 12, so v1 (4 epochs) was badly undertrained.

| Set | Step correct | Acts | Correct when acting | Other | p50 / p95 ms |
|---|---|---|---|---|---|
| validation (Install_11, all rows) | 95.9% | 94.7% | 99.1% | done 97.6%, anomaly 100% | 95 / 128 |
| **frozen 244** | **86.5%** | 89.1% | 87.7% | off-SOP FP 0.5%, echo 81% | 97 / 131 |
| probes (same generator as training) | P1 100%, P3 100% | 100% | 100% | anomaly caught 100% (mean p 0.9999) | 81 / 120 |

For comparison on the frozen 244: Jev 72.4% (83.0% when acting), Decider-4B 49.5% (68.0%).

Frozen-set errors (26): 22 are "one step ahead", concentrated on fan_1 (7/15 correct) and fan_2 (5/15); every other
step is ≥ 79%. 21 of the 26 are confident (≥ 0.8), so the tracker would act on them. Only 2 of the 26 equal the
tracker's expected step.

**The probes are no longer an independent test for this model**: they come from the same generator as its training
deviations. A diagnostic set of deviations that generator never produced (scratch, 30 each):

| Unseen deviation | Correct |
|---|---|
| out-of-order work at fan_{k+3} (training: k+2) | 100% |
| out-of-order install at fan_{k+3}, different confidences/events | 100% |
| idle bench, 14 s quiet, no closing "hand left chassis" event (training: 20-30 s, with it) | 10% |
| a done PSU removed (training: fans only) | 0% (anomaly p ≈ 0.0) |

Out-of-order work generalised; idle and part-removal were learned as the generator's surface patterns. Deviation
coverage needs more varied (ideally real) examples, and a deviation test set built independently of the training generator.
