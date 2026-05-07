from app.model.generation_lock import GenerationLock
from app.schemas.errors import ErrorCode


def test_generation_lock_rejects_second_nonblocking_acquire_as_parser_busy():
    lock = GenerationLock(max_concurrent_generations=1)

    first = lock.acquire(blocking=False)
    second = lock.acquire(blocking=False)

    try:
        assert first.acquired is True
        assert second.acquired is False
        assert second.error_code == ErrorCode.PARSER_BUSY
    finally:
        first.release()
        second.release()

    third = lock.acquire(blocking=False)
    try:
        assert third.acquired is True
    finally:
        third.release()
