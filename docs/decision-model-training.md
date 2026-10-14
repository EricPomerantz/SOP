# Training data for a local decision model

The local decision models (Decider, Laya) read the same `/v1/systemone` requests Jev does. Zero-shot they fall well short
of Jev on our SOP (`benchmarks/sop-244/RESULTS.md`), so the plan is to fine-tune one on decisions from our own recordings.
The commands below are for server fan; for another SOP add `--sop <file>` to the capture, build, freeze, probe and
benchmark scripts (its recordings come from the SOP file's `dataset` section, see `sopvision/recordings.py`).

```bash
# 1. Replay the train/validation recordings through the real detector + tracker and log every decision request (GPU, ~10 min)
docker compose run --rm --entrypoint python sopvision scripts/capture_requests.py --out data/training/captures
# 2. Label, augment and write Laya-format rows {state, questions, expected} (CPU, seconds)
docker compose run --rm --entrypoint python sopvision scripts/build_training_set.py data/training/captures --out data/training/sop-v1
# 3. Check Laya can read every row
docker compose --profile laya run --rm -v "$PWD/data/training":/work:ro --entrypoint laya-train laya \
    --data /work/sop-v1/train.jsonl --max-len 2048 --head-max-len 512 --dry-run
# 4. Fine-tune (stop the laya/decider services first to free the GPU). Mount at /work, not /data: the laya service
#    mounts data/training/models read-only at /data/models, which would shadow the output directory.
docker compose --profile laya run --rm -v "$PWD/data/training":/work --entrypoint laya-train laya \
    --data /work/sop-v1/train.jsonl --out /work/models/<name> --base convaiinnovations/laya \
    --max-len 2048 --head-max-len 512 --epochs 4 --device cuda --seed 0
# 5. Serve it and evaluate
LAYA_CHECKPOINT=/data/models/<name> docker compose --profile laya up -d --force-recreate laya
docker compose run --rm --entrypoint python sopvision scripts/eval_training_rows.py data/training/sop-v1/val.jsonl --provider laya
docker compose run --rm --entrypoint python sopvision scripts/bench_decisions.py --frozen benchmarks/sop-244 --providers laya
```

**Splits.** Train: the train recordings with valid step timing (Install_1, 4, 6, 7, 8, 9, 10; 3 and 5 have broken timing,
so their labels would be wrong). Validation: Install_11. Test: Install_12 and 13 only, as the frozen benchmark
(`benchmarks/sop-244`) and the deviation probes (`benchmarks/probes-d1`). The capture script refuses the test recordings,
and the builder drops any row whose state is byte-identical to a frozen state (3 rows in sop-v1: the empty bench every
run opens with, and one early state; listed in the manifest).

**Capture.** Lockstep replay with a stub model that never acts: steps complete on visual evidence, so the states are what
a live run produces, without any model's "active step" feeding back into them.

**Labels** (from each recording's step timeline): `current_step` = the step under way (off-SOP stretches: `idle`;
unlabelled within 1 s of a step boundary); `active_step_done` = whether the asked-about step's segment has ended
(unlabelled within 1 s of its end); `anomaly` = false (clean installs).

**Rows** (`data/training/sop-v1`, seed 0; counts in `manifest.json`):

| kind | train | val | what |
|---|---|---|---|
| in_order | 951 | 144 | the captured request |
| noise | 389 | 63 | YOLO-style noise on a copy (slot occluded/unknown, hand dropped/duplicated, event missing, confidence jitter), same labels |
| deviation (4 kinds) | 719 | 111 | the D1 probe edits (`scripts/make_probes.py`) applied to training states: out-of-order work, out-of-order install, idle bench, done part removed |

Deviations are ~35% of the set on purpose: on clean runs "answer the step the tracker expects" is right 97% of the time,
and a model trained only on clean runs learns exactly that shortcut.

**Known limits.**
- YOLO was trained on frames from the training recordings, so detections there are cleaner than on the test recordings;
  the noise rows partly compensate.
- `off_sop` never appears as a label (the recordings' off-SOP stretches are an idle bench), and anomalies are synthetic.
  Real recordings of off-SOP and out-of-order work would be the most valuable addition.
- `active_step_done` is ~7% true: the tracker moves on once a step is visibly done, so the asked-about step is usually
  still under way.
