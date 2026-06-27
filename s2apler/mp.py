"""Cross-platform ordered worker pool used by S2APLER hot paths."""

import multiprocessing as mp
import os
import platform
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


def _run_chunk(func: Callable[[Any], Any], idx_items: List[Tuple[int, Any]]) -> List[Tuple[int, bool, Any]]:
    """Run `func` over one submitted chunk while preserving source indexes."""
    out = []
    for idx, item in idx_items:
        try:
            out.append((idx, True, func(item)))
        except Exception as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"UniversalPool.imap item {idx} raised")  # type: ignore[attr-defined]
            out.append((idx, False, exc))
            break
    return out


class UniversalPool:
    """Small ordered `imap` pool with S2AND's platform policy.

    Linux uses forked processes for CPU-bound work. Windows and macOS use
    threads to avoid spawn-time dataset serialization.
    """

    def __init__(self, processes: Optional[int] = None, use_threads: Optional[bool] = None):
        if use_threads is None:
            use_threads = platform.system() in ("Windows", "Darwin")
        if processes is None:
            detected_cpu_count = os.cpu_count()
            self.processes = int(detected_cpu_count) if detected_cpu_count is not None else 1
        else:
            if int(processes) <= 0:
                raise ValueError(f"processes must be a positive integer when provided, got {processes!r}")
            self.processes = int(processes)

        self._pool: Union[ProcessPoolExecutor, ThreadPoolExecutor]
        if use_threads:
            self._pool = ThreadPoolExecutor(max_workers=self.processes)
        else:
            if platform.system() not in ("Windows", "Darwin"):
                ctx = mp.get_context("fork")
            else:
                ctx = mp.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.processes, mp_context=ctx)
        self._aborted = False

    def imap(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int = 1,
        max_prefetch: Optional[int] = None,
    ) -> Iterator[Any]:
        """Stream ordered results like `multiprocessing.Pool.imap`."""
        if int(chunksize) <= 0:
            raise ValueError(f"chunksize must be >= 1, got {chunksize!r}")
        prefetch = max(4, self.processes * 2) if max_prefetch is None else int(max_prefetch)
        if prefetch <= 0:
            raise ValueError(f"max_prefetch must be >= 1, got {max_prefetch!r}")
        return self._streaming_imap(func, iterable, int(chunksize), prefetch)

    def _streaming_imap(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int,
        max_prefetch: int,
    ) -> Iterator[Any]:
        it = enumerate(iterable)
        next_yield = 0
        buffer: Dict[int, Any] = {}
        errors: Dict[int, BaseException] = {}
        pending: Set[Future] = set()

        def submit_chunk() -> bool:
            chunk = list(islice(it, chunksize))
            if not chunk:
                return False
            pending.add(self._pool.submit(_run_chunk, func, chunk))
            return True

        for _ in range(max_prefetch):
            if not submit_chunk():
                break

        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.remove(fut)
                    for idx, ok, value in fut.result():
                        if ok:
                            buffer[idx] = value
                        else:
                            errors[idx] = value
                    submit_chunk()

                while True:
                    if next_yield in errors:
                        exc = errors.pop(next_yield)
                        self._annotate_abort(exc, pending)
                        raise exc
                    if next_yield not in buffer:
                        break
                    yield buffer.pop(next_yield)
                    next_yield += 1
        except BaseException as exc:
            if not self._aborted:
                self._annotate_abort(exc, pending)
            raise

    def _annotate_abort(self, exc: BaseException, pending: Set[Future]) -> None:
        cancelled_count = self._cancel_pending_futures(pending)
        if hasattr(exc, "add_note"):
            exc.add_note(  # type: ignore[attr-defined]
                "UniversalPool._streaming_imap aborted after an exception; "
                f"best-effort cancelled {cancelled_count} outstanding futures."
            )

    def _cancel_pending_futures(self, pending: Set[Future]) -> int:
        self._aborted = True
        cancelled_count = 0
        for fut in tuple(pending):
            if fut.cancel():
                cancelled_count += 1
            pending.discard(fut)
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._pool.shutdown(wait=False)
        return cancelled_count

    def __enter__(self) -> "UniversalPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._pool.shutdown(wait=not self._aborted)


def get_pool(processes: Optional[int] = None, threads: Optional[bool] = None) -> UniversalPool:
    """Return a `UniversalPool` using the default platform policy."""
    return UniversalPool(processes, use_threads=threads)
