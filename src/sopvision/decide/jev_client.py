"""The only module that talks to the TypeSafe SDK (spec §7.4)."""

import asyncio
import time

from sopvision.decide.questions import ANOMALY_QUESTION, CURRENT_STEP_QUESTION, QuestionSet
from sopvision.decide.types import JevAnswers

RETRY_DELAYS = (0.25, 0.5)


class JevUnavailableError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):  # includes the SDK's timeout/connection errors
        return True
    status = getattr(exc, "status", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


class JevClient:
    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None, sdk=None,
                 timeout: float = 1.5, max_retries: int = 2, sleep=asyncio.sleep):
        if sdk is None:
            from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

            # The SDK's own retries are off: the spec's retry rules apply here, inside our timeout.
            sdk = AsyncTypeSafeClient(api_key=api_key, base_url=base_url, model=model,
                                      retry=RetryPolicy(max_retries=0), timeout=timeout)
        self.sdk, self.timeout, self.max_retries, self._sleep = sdk, timeout, max_retries, sleep

    async def ask(self, state: dict, qs: QuestionSet) -> JevAnswers:
        from typesafe_sdk import Choice, Noul

        questions = {
            "current_step": Choice(instructions=CURRENT_STEP_QUESTION, criteria=qs.current_step_options),
            "active_step_done": Noul(instructions=qs.active_step_question),
            "anomaly": Noul(instructions=ANOMALY_QUESTION),
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                await self._sleep(RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)])
            start = time.perf_counter()
            try:
                resp = await asyncio.wait_for(self.sdk.system_one(state=state, questions=questions), self.timeout)
            except Exception as e:  # noqa: BLE001 — classified below
                if isinstance(e, asyncio.TimeoutError):
                    e = TimeoutError(f"Jev call timed out after {self.timeout}s")
                last_error = e
                if not _retryable(e):
                    raise JevUnavailableError(f"Jev request failed: {e}") from e
                continue
            current = resp.answers["current_step"]
            return JevAnswers(current.choice, float(current.confidence), dict(current.probabilities),
                              float(resp.answers["active_step_done"].noul), float(resp.answers["anomaly"].noul),
                              (time.perf_counter() - start) * 1000)
        raise JevUnavailableError(f"Jev unavailable after {self.max_retries + 1} attempts: {last_error}")

    async def aclose(self) -> None:
        close = getattr(self.sdk, "close", None) or getattr(self.sdk, "aclose", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result
