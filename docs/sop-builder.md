# Building an SOP in the dashboard

Open **SOPs** in the dashboard's top bar (`#/sops`). Every SOP is a card. **New SOP** starts a draft, and the builder
then walks through six stages in a row at the top of the page. A stage's dot turns green when it has everything it
needs. Each stage page says what it is for and, in a yellow box, what is still missing. Edits save by themselves:
the top bar says "Saving…", then "All changes saved". The builder can be left and resumed at any point.

Start the app with `./start.sh`, which starts the `worker` too. The decision-model stage also needs Laya:
`./start.sh --model laya`.

## The stages

| Stage | What you do | What it writes |
|---|---|---|
| Detector | Pick a trained model from the list (found under `models/`) or upload a `.pt`. Give each class a role: one **anchor** (the work piece the slots sit on), the **parts**, the **empty-slot marker**, the **tool**. A class named `hand` is the hands. | the draft's `detector` and `layout.anchor/tools` |
| Recordings | **Record a run** opens the Record page for this SOP and comes back here. Train runs teach the model, val runs tune it, test runs measure it and are never trained on. Record correct runs and mistakes. | `datasets/<folder>/runs/<run>/`, `splits.yaml` |
| Slots | **Suggest slots** finds where parts sit on the anchor across the recordings. **Use the N suggested slots**, then rename them as the procedure numbers them (`hole_1`, `hole_2`, …). Drag a ring to move a slot and check it on the other runs' frames. **Allow every part in every slot** when any part may go anywhere. | the draft's `layout.slots` |
| Steps | **Generate from slots** writes a place step per slot (any order) and, with a tool, a torque step per slot (a sequence). Edit the labels and instructions. Set the sequence with the **Position** boxes. **Advanced** holds the step ids, the off-SOP and idle wording, and when a step counts as done. | the draft's `steps`, `order`, `groups`, `decisions` |
| Timelines | **Draft** a run (or all of them) with the tracker, then **Review** it. On the run's page, play the video and press a step's key the moment it starts, then **Save**. | `runs/<run>/steps.draft.csv`, `preview.webm`, `steps.csv` |
| Decision model | **Train a new model** (about an hour on the GPU). Its results on the test runs read as "named the right step", "missed off-SOP actions", "false alarms". **Switch on** the one to use. | `data/training/models/<id>-v<n>`, `registry.json` |

**Review and publish** (top right) shows what goes live and lists anything still missing. Publishing puts the SOP in
the dashboard; recordings, timelines and the model can still be worked on afterwards. Editing a published SOP works
on a draft copy; publishing it again bumps the version and keeps the old file in `sops/.history/`. **Settings**
(the gear) changes the name, the description and the recordings folder.

## Reusing recordings

A new version of a procedure that was already recorded can reuse those recordings. In **New SOP**, pick the SOP
under **Reuse recordings from**. The two SOPs then share one folder: a run recorded or a timeline saved in either
changes both, and the Recordings and Timelines stages say so. Steps generated from numbered slots get the same ids
as the original's when its slots were numbered the same way (`hole_3` gives `place_3` and `torque_3`), so the
existing timelines keep matching.

## Rebuilding the lug-bolt SOP

The lug-bolt SOP (`sops/lug_bolts.yaml`) was written by hand. It can be rebuilt through the dashboard alone:

1. **New SOP**: name *Wheel lug bolts*, its description, **Reuse recordings from** *Wheel lug bolts*.
2. **Detector**: pick *lug_bolts / best.pt* and press **Use this model**. Set `hub` to Anchor, `hole_empty` to
   Empty-slot marker, the four `lug_bolt_*` to Part, and `torque_wrench` to Tool.
3. **Recordings**: the 25 shared runs show up (16 train, 2 val, 7 test).
4. **Slots**: **Suggest slots**, wait for it to finish (a few minutes), then **Use the 5 suggested slots**. Rename
   them clockwise from the top: `hole_1` at the top, `hole_2` at the right, `hole_3` bottom right, `hole_4` bottom
   left, `hole_5` at the left. Press **Allow every part in every slot**. Under **Advanced**, set the default radius
   to 0.15, anchor follow to 0.7 and removal wait to 8 s. The camera shakes while torquing, and the detector
   sometimes misreads a seated bolt as an empty hole.
5. **Steps**: **Generate from slots** gives `place_1` to `place_5` and `torque_1` to `torque_5`. In Group 2, set
   the positions to the star sequence: `torque_1`, `torque_3`, `torque_5`, `torque_2`, `torque_4`. Reword the
   labels ("Bolt 1", "Torque 1") and instructions if wanted. Under **Advanced**, set "Idle means" to
   *No one is working on the wheel*.
6. **Timelines**: all 25 runs already have saved timelines. Don't save over them here: they are the original SOP's
   labels too.
7. **Review and publish**.

The automated version of these steps runs as an end-to-end test (below). Its slot suggestions land within 0.03 of
the hand-measured hole positions, in fractions of the hub's box.

## The end-to-end test

```
docker compose build sopvision e2e
./start.sh                                        # the app and the worker
docker compose --profile e2e run --rm e2e         # everything
docker compose --profile e2e run --rm e2e smoke.spec.ts   # every page loads on the theme (seconds)
```

`lug_bolts_clone.spec.ts` follows the steps above in a browser as `lug_bolts_ui`. It suggests slots, drafts one
run's timeline, publishes, then compares the published file with `sops/lug_bolts.yaml`. The comparison covers the
slots (within `E2E_SLOT_TOL`, default 0.03), steps, order, groups, decisions and the recordings folder. Afterwards
it removes the clone's files; `E2E_KEEP=1` keeps them. The app keeps the published clone in its SOP list until its
next restart. It takes about ten minutes and needs the GPU. It refuses to start while a run, a recording or a job is
going. Screenshots of each stage and the report land in `web/e2e/test-results/`. Training the decision model is
left out: it takes an hour. Do it once by hand from the Decision model stage.

## Notes

- Saving through the builder rewrites the YAML: hand-written comments in an SOP file don't survive an edit there.
- Only import detector files you trust: loading a `.pt` file can run code.
- The scene recognises hands only from a detector class literally named `hand`.
- Slots are fractions of the anchor's axis-aligned box, so they follow the anchor moving and scaling, not turning.
- Jobs run one at a time on the GPU. Training shares it with a live run. `./stop.sh` refuses while a job runs; a job
  cut off by a stop is marked failed when the worker starts again.
- Each SOP's decision model is chosen per request (`ckpt-<folder>`). An SOP without one gets the shared model
  (`LAYA_CHECKPOINT`). Rebuild the Laya image after updating `docker/laya_serve.py`
  (`docker compose --profile laya build laya`).
- The pages use the deviceWISE suite's theme (`web/src/theme.ts`, `web/src/theme/`, copied unchanged). Update
  them from `devicewise-intelligence-suite/frontend` rather than editing them here.
