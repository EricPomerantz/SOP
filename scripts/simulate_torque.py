"""Send torque results to the running app, standing in for the torque wrench system until it is connected.

The app works out *which* hole a result belongs to from where the wrench is in the video, so this only says *that*
a tightening finished (and whether it reached torque).

    python scripts/simulate_torque.py                     interactive: Enter = OK, "f" + Enter = not reached
    python scripts/simulate_torque.py --log torque.csv    replay a recorded log in time with a video started now
    python scripts/simulate_torque.py --every 5 --count 5 one OK result every 5 s

    --url http://localhost:8000 (default)    --torque 110 (Nm reported with each result)

The run is the one currently running in the app (`GET /runs/current`). From inside Docker:
    docker compose run --rm --entrypoint python sopvision scripts/simulate_torque.py --url http://sopvision:8000 ...
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def current_run(url: str) -> str:
    run = httpx.get(f"{url}/runs/current", timeout=5).json()
    if not run:
        sys.exit("no run in progress: start one on the dashboard first")
    return run["run_id"]


def send(url: str, run_id: str, ok: bool, torque_nm: float | None) -> None:
    r = httpx.post(f"{url}/runs/{run_id}/torque", json={"ok": ok, "torque_nm": torque_nm}, timeout=5)
    if r.status_code != 200:
        sys.exit(f"{r.status_code}: {r.json().get('detail')}")
    print(f"{'OK' if ok else 'NOT REACHED'} {torque_nm or ''} Nm -> run time {r.json()['t']:.2f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--log", type=Path, help="torque.csv to replay (time_s, ok, torque_nm)")
    ap.add_argument("--every", type=float, help="seconds between OK results")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--torque", type=float, default=110.0)
    args = ap.parse_args()
    run_id = current_run(args.url)
    print(f"run {run_id}", flush=True)
    if args.log:
        from sopvision.sop.torque import read_torque_log

        start = time.monotonic()
        for r in read_torque_log(args.log):
            time.sleep(max(0.0, r.t - (time.monotonic() - start)))
            send(args.url, run_id, r.ok, r.torque_nm)
    elif args.every:
        for _ in range(args.count):
            time.sleep(args.every)
            send(args.url, run_id, True, args.torque)
    else:
        print("Enter = torque OK, f + Enter = not reached, q + Enter = quit", flush=True)
        for line in sys.stdin:
            if line.strip().lower() == "q":
                break
            send(args.url, run_id, line.strip().lower() != "f", args.torque)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
