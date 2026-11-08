"""Session checks shared by the test suite. Copied from s2and's `tests/conftest.py`."""

import os

# Set these before a library starts its thread pool. Each xdist worker otherwise
# starts one OpenMP thread for each core, and the workers compete for the cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
# A known count stops the joblib physical-core probe, which costs about 2 s in each worker.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import pytest  # noqa: E402

from tests.rust_build_freshness import duplicate_package_metadata_message, stale_rust_extension_message  # noqa: E402


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop before collection when package metadata or the native extension is stale."""
    message = duplicate_package_metadata_message() or stale_rust_extension_message()
    if message is not None:
        pytest.exit(message, returncode=pytest.ExitCode.USAGE_ERROR)
