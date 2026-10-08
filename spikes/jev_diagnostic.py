"""Diagnostic: check Jev connectivity and identify the issue."""

import asyncio
import argparse
import os
import socket
import ssl
import traceback
from urllib.error import HTTPError
from urllib.request import urlopen

def load_env_file() -> None:
    if not os.path.isfile(".env"):
        return
    with open(".env", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def test_dns() -> None:
    """Test DNS resolution for api.typesafe.ai."""
    print("\n=== DNS Resolution Test ===")
    try:
        ip = socket.gethostbyname("api.typesafe.ai")
        print(f"✓ DNS resolved: api.typesafe.ai → {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS resolution failed: {e}")


def test_https_connectivity(verify_ssl: bool) -> None:
    """Test basic HTTPS connectivity to the API."""
    print("\n=== HTTPS Connectivity Test ===")
    context = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
    try:
        with urlopen("https://api.typesafe.ai/health", timeout=5, context=context) as response:
            print(f"✓ HTTPS connection successful: {response.status} {response.reason}")
    except HTTPError as e:
        print(f"✓ TLS connection succeeded; health endpoint returned HTTP {e.code}")
    except Exception as e:
        print(f"❌ HTTPS connection failed: {type(e).__name__}: {e}")


async def test_jev_api(verify_ssl: bool) -> None:
    """Test the actual Jev API call."""
    print("\n=== Jev API Call Test ===")
    load_env_file()
    api_key = os.getenv("TYPESAFE_API_KEY", "")

    if not api_key:
        print("❌ TYPESAFE_API_KEY is not set in .env")
        return

    try:
        from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy
        print("✓ typesafe_sdk imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import typesafe_sdk: {e}")
        return

    base_url = os.getenv("TYPESAFE_BASE_URL")
    model = os.getenv("TYPESAFE_DEFAULT_MODEL")
    print(f"✓ API key loaded: {'*' * max(0, len(api_key) - 4)}{api_key[-4:]}")
    print(f"  Base URL: {base_url or 'https://api.typesafe.ai (default)'}")
    print(f"  Model: {model or 'jev-latest (default)'}")

    client = None
    try:
        import httpx2
        client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            retry=RetryPolicy(max_retries=0),
            timeout=5.0,
            http_client=httpx2.AsyncClient(verify=verify_ssl),
        )
        print("\nAttempting to call Jev API...")
        response = await client.system_one(
            state={"sop": "Connectivity test", "steps": []},
            questions={
                "current_step": Choice(
                    instructions="Which activity is happening?",
                    criteria={"idle": "No activity", "off_sop": "Other activity"},
                ),
                "active_step_done": Noul(instructions="Is the task complete?"),
                "anomaly": Noul(instructions="Is there an anomaly?"),
            },
        )
        answer = response.answers["current_step"]
        print("✓ Jev call succeeded")
        print(f"  Response: choice={answer.choice}, confidence={answer.confidence}")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if client is not None:
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check Jev API connectivity")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS certificate verification for this diagnostic only",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Jev Connectivity Diagnostic")
    print("=" * 60)
    if args.insecure:
        print("WARNING: TLS certificate verification is disabled; do not use this for normal app traffic.")

    test_dns()
    test_https_connectivity(verify_ssl=not args.insecure)
    await test_jev_api(verify_ssl=not args.insecure)

    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

