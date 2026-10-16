# Building an SOP in the dashboard

Open **SOPs** in the dashboard's top bar (`#/sops`), create a draft, and work through the tabs. Each tab's tick
comes from the files on disk, so the builder can be left and resumed at any point. Start the app with `./start.sh`
(it starts the `worker` too) and, for stage 6, with Laya: `./start.sh --model laya`.

| Stage | What you do | What it writes |
|---|---|---|
| 1 Detector | Upload a trained Ultralytics detection model (.pt), or pick one under `models/`; give each class a role: anchor (the work piece the slots sit on), part, empty-slot marker, tool, ignore. A class called `hand` is the hands. | `models/<id>/detector-<time>.pt`, the draft's `detector` and `layout.anchor/tools` |
| 2 Recordings | Record runs on the Record page (the SOP preselected): train and val runs teach the decision model, test runs measure it. Record correct runs and deviations. | `datasets/<id>/runs/<run>/{video.mp4, torque.csv}`, `splits.yaml` |
| 3 Slots | "Suggest slots" clusters where parts and empty markers sit on the anchor across the recordings (a moving anchor is fine; a rotating one is warned about). Drag the rings, check them on other runs' frames, set what fills each slot and its radius. | the draft's `layout.slots` (job results in `data/builder/<id>/`) |
| 4 Steps | "Generate from slots", then edit: labels, instructions, evidence, order (strict, any, groups with any-order or sequence groups). | the draft's `steps`, `order`, `groups` |
| 5 Timelines | "Draft timelines" replays every run through the tracker; open each run, correct the step changes on the video (keys 1-9 for steps, `i` idle, `o` off-SOP) and save. | `runs/<run>/steps.draft.csv`, `preview.webm`, `steps.csv` |
| 6 Decision model | "Train a new checkpoint": capture → benchmark (test runs, frozen) → training set (off-SOP labelled apart from idle) → laya-train → benchmark and replay → report. Compare with the active checkpoint, then Activate. | `data/training/models/<id>-v<n>` and `.eval/report.json`; Activate writes `registry.json` |
| Publish | Validates the draft and moves it into `sops/`; the SOP appears in the dashboard's SOP list. Editing a published SOP works on a draft copy; publishing it again bumps the version and keeps the old file in `sops/.history/`. | `sops/<file>.yaml` |

Notes

- Saving through the builder rewrites the YAML: hand-written comments in an SOP file don't survive an edit there.
- Only import detector files you trust: loading a `.pt` file can run code.
- The scene recognises hands only from a detector class literally named `hand`.
- Slots are fractions of the anchor's axis-aligned box, so they follow the anchor moving and scaling, not turning.
- Jobs run one at a time on the GPU; training (about an hour) shares it with a live run. `./stop.sh` refuses
  while a job runs; a job cut off by a stop is marked failed when the worker starts again.
- Each SOP's checkpoint is chosen per request (`ckpt-<folder>`); an SOP without one gets the shared checkpoint
  (`LAYA_CHECKPOINT`). Rebuild the Laya image after updating `docker/laya_serve.py`
  (`docker compose --profile laya build laya`).
