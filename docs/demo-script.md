# SOP Vision — 5-minute demo

**Before you start:** `./start.sh --model jev --no-labelstudio`, open http://localhost:8000, make sure no run is in progress.

## 1. A clean run (2 min)

1. Pick **Install_12 (test run)** and press **Start**. This recording was never used for training.
2. While the operator works, point out:
   - **Video:** the dashed chassis box and the dot on every bay — green installed, grey empty, amber when a hand covers it.
   - **Checklist:** a step turns blue when Jev names it as the one under way (with Jev's confidence), and ticks off with its duration once the part is visibly in place.
   - **Jev now:** which step Jev thinks is happening, how sure it is, and how fast it answered (~150–250 ms).
3. Use **2×** to move faster if needed. At the end, every step is ticked and the status stays **On track**.

## 2. A part taken back out (1 min)

1. Pick **Simulated: fan removed** and **Start**.
2. Three fans go in, then the third comes back out (the clip plays backwards).
3. The log shows **"fan_3: part removed after the step was completed"** and the status turns **Deviation**; Fan 3 returns to pending.

## 3. An unfinished job (30 s)

1. Pick **Simulated: incomplete run** and **Start**.
2. When the video ends: **"SOP incomplete: missing fan_5, fan_6, psu_1, psu_2, cover"**, status **Incomplete**.

## 4. The record (30 s)

Press **Download run report** — a CSV with every step's completion time, duration and how it was confirmed (Jev or visual evidence), plus every alert.

## Talking points

- **Video never leaves the machine.** Only a small JSON description of the scene goes to Jev (~1,000 tokens a call).
- **Cost:** Jev input is $0.042 per million tokens and output is free — well under $0.001 per station-hour.
- **Who does what:** Jev reads the scene description, names the step under way, and watches for off-SOP activity and anomalies. A step is ticked off once its part is visibly in place, so the checklist keeps working even if Jev is unreachable.
- **Don't overclaim:** on this footage every step was ticked off by visual evidence, not by Jev. The off-SOP stretches are an empty bench, so you'll see no off-SOP warning. Present Jev as "knows which step is happening", not "confirms each step" or "flags off-SOP work". Those still need footage that shows them.
- **New equipment = new config.** An SOP is a YAML file; the detector is retrained from three reference frames boxed once (~10 minutes of someone's time).
- **Wrong order and skipped steps** are caught too (see `docker compose run --rm gpu-tests -m jev`); this footage only contains correct runs, so they're shown with scripted scenes.
- **Production path:** the same Docker image runs on an on-prem Linux GPU server; RTSP cameras plug in through `.env`.
