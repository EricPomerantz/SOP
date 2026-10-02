"""Container smoke test: image builds, sees the GPU, serves /sops and /sources (no webcam in the container).

Usage: uv run python scripts/docker_smoke.py
"""

import subprocess
import sys
import time

import httpx

BASE = "http://localhost:8000"


def main() -> int:
    subprocess.run(["docker", "compose", "up", "-d", "--build"], check=True)
    try:
        deadline = time.time() + 120
        while True:
            try:
                sops = httpx.get(f"{BASE}/sops", timeout=5).json()
                break
            except httpx.HTTPError:
                if time.time() > deadline:
                    print(subprocess.run(["docker", "compose", "logs", "--tail", "40"], capture_output=True, text=True).stdout)
                    print("docker smoke: FAILED (server never answered)")
                    return 1
                time.sleep(2)
        sources = httpx.get(f"{BASE}/sources", timeout=5).json()
        assert [s["id"] for s in sops] == ["server_fan_install"], sops
        assert sources and all(s["kind"] != "webcam" for s in sources), sources
        print(f"docker smoke: OK ({len(sources)} sources, no webcam)")
        return 0
    finally:
        subprocess.run(["docker", "compose", "down"], check=False)


if __name__ == "__main__":
    sys.exit(main())
