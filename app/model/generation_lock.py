from dataclasses import dataclass
from threading import BoundedSemaphore

from app.schemas.errors import ErrorCode


@dataclass
class GenerationLockLease:
    _semaphore: BoundedSemaphore
    acquired: bool
    error_code: ErrorCode | None = None

    def release(self) -> None:
        if not self.acquired:
            return
        self._semaphore.release()
        self.acquired = False

    def __enter__(self) -> "GenerationLockLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class GenerationLock:
    def __init__(self, max_concurrent_generations: int = 1) -> None:
        if max_concurrent_generations < 1:
            raise ValueError("max_concurrent_generations must be >= 1")
        self.max_concurrent_generations = max_concurrent_generations
        self._semaphore = BoundedSemaphore(max_concurrent_generations)

    def acquire(self, *, blocking: bool = True, timeout: float | None = None) -> GenerationLockLease:
        if not blocking and timeout is not None:
            raise ValueError("timeout cannot be set for non-blocking acquire")

        if timeout is None:
            acquired = self._semaphore.acquire(blocking=blocking)
        else:
            acquired = self._semaphore.acquire(blocking=blocking, timeout=timeout)

        if acquired:
            return GenerationLockLease(self._semaphore, acquired=True)
        return GenerationLockLease(
            self._semaphore,
            acquired=False,
            error_code=ErrorCode.PARSER_BUSY,
        )
