"""Read-only API and dashboard smoke test for an already-started Docker deployment.

docker compose run --rm --entrypoint python tests scripts/docker_smoke.py
"""

import argparse
import sys

import httpx


def check_server(client: httpx.Client) -> tuple[int, int]:
    response = client.get("/sops")
    response.raise_for_status()
    sops = response.json()
    response = client.get("/sources")
    response.raise_for_status()
    sources = response.json()
    if not sops or not all(sop["id"] and sop["steps"] for sop in sops):
        raise ValueError("No valid SOPs available")
    if not sources or not all(source["kind"] in {"file", "rtsp"} for source in sources):
        raise ValueError("Expected file or RTSP sources")
    for path in ("/", "/config", "/latency", "/recording", "/runs/current"):
        client.get(path).raise_for_status()
    return len(sops), len(sources)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://sopvision:8000")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            sops, sources = check_server(client)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        print(f"docker smoke: FAILED ({error})", file=sys.stderr)
        return 1
    print(f"docker smoke: OK ({sops} SOPs, {sources} sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
