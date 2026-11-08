"""Load S2APLER's native extension. Copied from s2and's `runtime.py`."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any

from s2apler import __version__

_CARGO_PRERELEASE = re.compile(r"([0-9]+\.[0-9]+\.[0-9]+)-(alpha|beta|rc)\.([0-9]+)")
_PEP440_PRERELEASE_TAGS = {"alpha": "a", "beta": "b", "rc": "rc"}


def _python_version_from_cargo(version: object) -> object:
    """Map the extension's Cargo pre-release spelling to PEP 440, e.g. 1.0.0-beta.1 -> 1.0.0b1."""
    # The extension reports its Cargo version, which spells pre-releases differently from its wheel.
    match = _CARGO_PRERELEASE.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        return version
    base, tag, number = match.groups()
    return f"{base}{_PEP440_PRERELEASE_TAGS[tag]}{number}"


def load_s2apler_rust_extension(*, import_module: Callable[[str], Any] | None = None) -> Any:
    """Import the native extension and require its version to match the Python package.

    Args:
        import_module: replaces `importlib.import_module`, for tests.

    Returns:
        The `s2apler._s2apler_rust` module.

    Raises:
        RuntimeError: when the extension is missing or was built for another version.
    """
    importer = import_module or importlib.import_module
    try:
        module = importer("s2apler._s2apler_rust")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("s2apler._s2apler_rust"):
            raise
        raise RuntimeError(f"s2apler=={__version__} native extension is not importable") from exc

    found_version = getattr(module, "__version__", None)
    if _python_version_from_cargo(found_version) != __version__:
        raise RuntimeError(
            "The installed extension version does not match the Python package: "
            f"required={__version__!r} found={found_version!r}"
        )
    return module
