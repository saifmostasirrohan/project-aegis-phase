import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit is open and calls are temporarily blocked."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False

        elapsed = time.monotonic() - self.opened_at
        if elapsed >= self.recovery_timeout:
            self.reset()
            return False

        return True

    def reset(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_success(self) -> None:
        self.reset()

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.monotonic()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.is_open:
            remaining = self.recovery_timeout - (time.monotonic() - (self.opened_at or 0))
            raise CircuitBreakerOpenError(
                f"Circuit is open. Try again in {max(0, remaining):.1f} seconds."
            )

        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise

        self.record_success()
        return result

    def protect(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def flaky_arxiv_api(query: str = "medical imaging") -> str:
    """Pretend to call ArXiv, but fail 70% of the time."""
    if random.random() < 0.70:
        raise Exception("Simulated ArXiv API failure.")

    return f"ArXiv results for '{query}': simulated successful response."


arxiv_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
safe_arxiv_api = arxiv_circuit_breaker.protect(flaky_arxiv_api)


if __name__ == "__main__":
    print("Testing flaky ArXiv API with Tenacity retry + circuit breaker.\n")

    for attempt in range(1, 11):
        print(f"Call {attempt}: ", end="")

        try:
            response = safe_arxiv_api("transformers medical imaging")
            print(response)
        except CircuitBreakerOpenError as exc:
            print(f"Blocked by circuit breaker: {exc}")
        except Exception as exc:
            print(f"Failed after retries: {exc}")

        time.sleep(1)
