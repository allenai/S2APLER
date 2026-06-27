import platform
import threading
import time

import pytest

from s2apler.mp import UniversalPool


def _double(value):
    return value * 2


def _slow_zero_raise_one(value):
    if value == 0:
        time.sleep(0.2)
        return value
    raise ValueError("later item failed")


def _fail_first_sleep_rest(value):
    if value == 0:
        raise ValueError("fail fast")
    time.sleep(0.6)
    return value


def test_universal_pool_threaded_imap_preserves_order():
    with UniversalPool(processes=3, use_threads=True) as pool:
        output = list(pool.imap(_double, range(10), chunksize=2))

    assert output == [value * 2 for value in range(10)]


def test_universal_pool_default_prefetch_uses_all_thread_workers():
    active = 0
    max_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def _wait_for_all_workers(value):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2.0)
        except threading.BrokenBarrierError:
            pass
        finally:
            with lock:
                active -= 1
        return value

    with UniversalPool(processes=8, use_threads=True) as pool:
        output = list(pool.imap(_wait_for_all_workers, range(8), chunksize=1))

    assert output == list(range(8))
    assert max_active == 8


def test_universal_pool_threaded_imap_preserves_order_before_later_exception():
    with UniversalPool(processes=2, use_threads=True) as pool:
        iterator = pool.imap(_slow_zero_raise_one, [0, 1], chunksize=1)
        assert next(iterator) == 0
        with pytest.raises(ValueError, match="later item failed"):
            next(iterator)


def test_universal_pool_threaded_abort_does_not_wait_for_running_work():
    start = time.perf_counter()
    with pytest.raises(ValueError, match="fail fast"):
        with UniversalPool(processes=4, use_threads=True) as pool:
            list(pool.imap(_fail_first_sleep_rest, range(4), chunksize=1))

    assert time.perf_counter() - start < 0.5


@pytest.mark.skipif(
    platform.system() in {"Windows", "Darwin"},
    reason="Process path is Linux/fork-oriented",
)
def test_universal_pool_process_imap_preserves_order():
    with UniversalPool(processes=2, use_threads=False) as pool:
        output = list(pool.imap(_double, range(8), chunksize=2))

    assert output == [value * 2 for value in range(8)]


def test_universal_pool_rejects_invalid_chunksize():
    with UniversalPool(processes=1, use_threads=True) as pool:
        with pytest.raises(ValueError, match="chunksize"):
            list(pool.imap(_double, range(3), chunksize=0))
