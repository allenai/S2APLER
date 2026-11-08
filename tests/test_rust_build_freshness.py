"""Source digest used to reject stale native extensions before collection."""

import importlib.machinery
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from tests.rust_build_freshness import (
    CLEAN_COMMAND,
    REBUILD_COMMAND,
    duplicate_package_metadata_message,
    rust_source_digest,
    stale_rust_extension_message,
)

_TREE = {
    "Cargo.toml": b"[package]\n",
    "Cargo.lock": b"# lock\n",
    "build.rs": b"fn main() {}\n",
    "src/lib.rs": b"pub fn f() {}\n",
    "src/nested/mod.rs": b"pub fn g() {}\n",
}


def _write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _link_directory(link: Path, target: Path) -> None:
    """Exercise Windows junctions without requiring symlink privileges."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def _unsupported_source_path(root: Path) -> Path:
    name = "\udcff.rs" if os.name == "nt" else os.fsdecode(b"\xff.rs")
    return root / "src" / name


def _install_native_stub(monkeypatch: pytest.MonkeyPatch, native_path: Path, build_info: dict[str, str]) -> None:
    native = ModuleType("s2apler._s2apler_rust")
    native.__file__ = str(native_path)
    monkeypatch.setattr(native, "get_build_info", lambda: build_info, raising=False)
    monkeypatch.setitem(sys.modules, "s2apler._s2apler_rust", native)


def test_digest_is_independent_of_file_creation_order(tmp_path: Path) -> None:
    forward = _write_tree(tmp_path / "forward", _TREE)
    reverse = _write_tree(tmp_path / "reverse", dict(reversed(list(_TREE.items()))))

    assert rust_source_digest(forward) == rust_source_digest(reverse)
    assert rust_source_digest(forward).startswith("v1:")


@pytest.mark.parametrize(
    "change",
    [
        "modify_source",
        "modify_manifest",
        "modify_lockfile",
        "modify_build_script",
        "add_source",
        "delete_source",
        "rename_source",
    ],
)
def test_digest_changes_for_every_build_input_change(tmp_path: Path, change: str) -> None:
    root = _write_tree(tmp_path, _TREE)
    older_ns = 1_600_000_000_000_000_000
    newer_ns = older_ns + 100_000_000_000
    if change in {"modify_source", "add_source", "delete_source"}:
        # Older additions/deletions and preserved mtimes must still change the
        # digest even when a timestamp-based build cache might miss the edit.
        for path in (root / "src").rglob("*.rs"):
            os.utime(path, ns=(older_ns, older_ns))
        os.utime(root / "src/lib.rs", ns=(newer_ns, newer_ns))
    baseline = rust_source_digest(root)

    if change == "modify_source":
        (root / "src/lib.rs").write_bytes(b"pub fn f() { }\n")
        os.utime(root / "src/lib.rs", ns=(newer_ns, newer_ns))
    elif change == "modify_manifest":
        (root / "Cargo.toml").write_bytes(b"[package]\nname = 'x'\n")
    elif change == "modify_lockfile":
        (root / "Cargo.lock").write_bytes(b"# updated lock\n")
    elif change == "modify_build_script":
        (root / "build.rs").write_bytes(b"fn main() { }\n")
    elif change == "add_source":
        (root / "src/new.rs").write_bytes(b"")
        os.utime(root / "src/new.rs", ns=(older_ns, older_ns))
    elif change == "delete_source":
        (root / "src/nested/mod.rs").unlink()
    else:
        (root / "src/nested/mod.rs").rename(root / "src/nested/other.rs")

    assert rust_source_digest(root) != baseline, change


def test_digest_ignores_metadata_and_build_outputs(tmp_path: Path) -> None:
    root = _write_tree(tmp_path, _TREE)
    baseline = rust_source_digest(root)

    _write_tree(
        root,
        {
            "target/release/lib.dll": b"binary",
            "s2apler_rust/__init__.py": b"",
            "README.md": b"documentation",
            "src/lib.rs.orig": b"editor backup",
            "src/.swp": b"editor state",
        },
    )

    assert rust_source_digest(root) == baseline


def test_digest_rejects_linked_source_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_tree(tmp_path / "crate", _TREE)
    _install_native_stub(monkeypatch, tmp_path / "native.pyd", {"source_digest": rust_source_digest(source)})
    external = tmp_path / "external"
    external.mkdir()
    (external / "module.rs").write_bytes(b"pub fn linked() {}\n")
    _link_directory(source / "src/linked", external)

    with pytest.raises(ValueError, match="linked native source directories are unsupported") as error:
        rust_source_digest(source)
    message = stale_rust_extension_message(source)
    assert message == (
        f"Cannot verify s2apler_rust source freshness: {error.value}. "
        "Correct the native source inputs before rebuilding."
    )
    assert "INTERNALERROR" not in message


def test_digest_rejects_unsupported_source_names_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_tree(tmp_path, _TREE)
    _install_native_stub(monkeypatch, tmp_path / "native.pyd", {"source_digest": rust_source_digest(source)})
    _unsupported_source_path(source).write_bytes(b"// unused module\n")

    with pytest.raises(ValueError, match="native source paths must be valid UTF-8") as error:
        rust_source_digest(source)
    message = stale_rust_extension_message(source)
    assert message == (
        f"Cannot verify s2apler_rust source freshness: {error.value}. "
        "Correct the native source inputs before rebuilding."
    )
    assert "INTERNALERROR" not in message


def _uv_cache_project(root: Path) -> tuple[Path, Path]:
    """Use the native package's real cache keys with an offline wheel builder."""
    source = _write_tree(root / "package", _TREE)
    native_manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
    keys = tomllib.loads(native_manifest.read_text(encoding="utf-8"))["tool"]["uv"]["cache-keys"]
    cache_keys = ", ".join("{ file = " + json.dumps(key["file"].removeprefix("s2apler_rust/")) + " }" for key in keys)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "s2apler"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        f"[tool.uv]\ncache-keys = [{cache_keys}]\n",
        encoding="utf-8",
    )
    (source / "backend.py").write_text(
        "from pathlib import Path\n"
        "from zipfile import ZipFile\n"
        "\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    counter = Path(__file__).resolve().parent / 'build_count.txt'\n"
        "    count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "    counter.write_text(str(count))\n"
        "    name = 's2apler-0.0.0-py3-none-any.whl'\n"
        "    dist = 's2apler-0.0.0.dist-info'\n"
        "    with ZipFile(Path(wheel_directory) / name, 'w') as wheel:\n"
        "        wheel.writestr('freshness_probe.py', f'BUILD_COUNT = {count}\\n')\n"
        "        wheel.writestr(dist + '/METADATA', "
        "'Metadata-Version: 2.1\\nName: s2apler\\nVersion: 0.0.0\\n')\n"
        "        wheel.writestr(dist + '/WHEEL', "
        "'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
        "        wheel.writestr(dist + '/RECORD', '')\n"
        "    return name\n",
        encoding="utf-8",
    )
    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "freshness-probe-host"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["s2apler"]\n[project.optional-dependencies]\ndev = []\n'
        '[tool.uv.sources]\ns2apler = { path = "../package" }\n',
        encoding="utf-8",
    )
    return source, project


def _sync_uv_cache_project(project: Path, *, recovery: bool = False) -> int:
    """Run uv against only the fixture's environment and local cache."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to exercise local package cache invalidation")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    arguments = shlex.split(REBUILD_COMMAND)[1:] if recovery else ["sync", "--extra", "dev"]
    result = subprocess.run(
        [
            uv,
            "--cache-dir",
            str(project.parent / "cache"),
            *arguments,
            "--project",
            str(project),
            "--python",
            sys.executable,
            "--offline",
            "--no-python-downloads",
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return int((project.parent / "package/build_count.txt").read_text(encoding="utf-8"))


def test_uv_cache_and_digest_track_only_native_inputs(tmp_path: Path) -> None:
    """Check every cache edit independently while reusing one offline fixture."""
    source, project = _uv_cache_project(tmp_path)
    baseline = rust_source_digest(source)
    assert _sync_uv_cache_project(project) == 1, "initial build"
    assert _sync_uv_cache_project(project) == 1, "unchanged fixture must stay cached"

    for relative in ("src/lib.rs.orig", "src/.swp", "target/release/generated.rs", "README.md"):
        ignored = source / relative
        ignored.parent.mkdir(parents=True, exist_ok=True)
        for operation, content in (
            ("add", b"irrelevant file\n"),
            ("modify", b"changed irrelevant file\n"),
            ("delete", None),
        ):
            if content is None:
                ignored.unlink()
            else:
                ignored.write_bytes(content)
            scenario = f"{relative}: {operation}"
            assert rust_source_digest(source) == baseline, scenario
            assert _sync_uv_cache_project(project) == 1, scenario

    for expected_builds, relative in enumerate(("src/lib.rs", "src/nested/mod.rs", "Cargo.toml", "build.rs"), start=2):
        previous_digest = rust_source_digest(source)
        path = source / relative
        path.write_bytes(path.read_bytes() + b"\n// changed\n")
        assert rust_source_digest(source) != previous_digest, f"{relative}: edited input must change digest"
        assert _sync_uv_cache_project(project) == expected_builds, f"{relative}: edit must rebuild once"

    # The suggested recovery must force a build even when the cache keys match.
    assert _sync_uv_cache_project(project) == 5, "unchanged fixture must stay cached"
    assert _sync_uv_cache_project(project, recovery=True) == 6, "recovery must rebuild a cached package"


def test_uv_cache_tracks_packaged_licenses(tmp_path: Path) -> None:
    """Rebuild cached distributions after each packaged license or notice changes."""
    source, project = _uv_cache_project(tmp_path)
    manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
    license_files = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["license-files"]
    paths = [source / relative.removeprefix("s2apler_rust/") for relative in license_files]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Original notice\n", encoding="utf-8")

    assert _sync_uv_cache_project(project) == 1, "initial build"
    for expected_builds, path in enumerate(paths, start=2):
        path.write_text("Updated notice\n", encoding="utf-8")
        assert _sync_uv_cache_project(project) == expected_builds, f"{path.name}: edit must rebuild once"
        assert _sync_uv_cache_project(project) == expected_builds, "unchanged fixture must stay cached"


def _session_project(root: Path) -> tuple[Path, Path]:
    """Copy the real session hook into a small independent pytest project.

    Like the repository, the project root holds an ``s2apler_rust/`` source
    directory without ``__init__.py``; the package imports from ``installed/``.
    """
    source_root = _write_tree(root / "s2apler_rust", _TREE)
    tests = root / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    for name in ("conftest.py", "rust_build_freshness.py"):
        shutil.copyfile(Path(__file__).with_name(name), tests / name)
    (tests / "test_collection.py").write_text(
        "from pathlib import Path\nPath('collected').touch()\ndef test_collected():\n    pass\n",
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    package = root / "installed" / "s2apler"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ._s2apler_rust import *\n", encoding="utf-8")
    return source_root, package / "_s2apler_rust.py"


def _run_session(root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "installed")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini", "tests"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_session_rejects_source_edit_before_collection(tmp_path: Path) -> None:
    """Exercise the real hook and import path, without compiling a fake crate."""
    source_root, native = _session_project(tmp_path)
    digest = rust_source_digest(source_root)
    native.write_text(f"def get_build_info():\n    return {{'source_digest': {digest!r}}}\n", encoding="utf-8")

    fresh = _run_session(tmp_path)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert (tmp_path / "collected").exists()
    (tmp_path / "collected").unlink()

    (source_root / "src/lib.rs").write_bytes(b"pub fn changed() {}\n")

    stale = _run_session(tmp_path)
    output = stale.stdout + stale.stderr
    assert stale.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert str(native) in output
    assert REBUILD_COMMAND in output
    assert CLEAN_COMMAND in output
    assert "different sources" in output


def test_missing_source_digest_reports_loaded_path_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_tree(tmp_path / "crate", _TREE)
    native = tmp_path / "installed" / "_s2apler_rust.pyd"
    _install_native_stub(monkeypatch, native, {"crate_version": "1.0.0"})

    message = stale_rust_extension_message(source)

    assert message is not None
    assert "different sources" in message
    assert f"embedded=None, checkout={rust_source_digest(source)!r}" in message
    assert str(native) in message
    assert REBUILD_COMMAND in message
    assert CLEAN_COMMAND in message


def test_missing_native_module_reports_rebuild_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "s2apler._s2apler_rust", None)

    message = stale_rust_extension_message(tmp_path)

    assert message is not None
    assert "Cannot import the s2apler_rust extension" in message
    assert REBUILD_COMMAND in message


def test_uninstalled_extension_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without an install, the checkout's source directory imports as a namespace package."""
    source_root = _write_tree(tmp_path / "s2apler_rust", _TREE)
    spec = importlib.machinery.PathFinder.find_spec("s2apler_rust", [str(tmp_path)])
    assert spec is not None and spec.loader is None
    monkeypatch.setitem(sys.modules, "s2apler._s2apler_rust", importlib.util.module_from_spec(spec))

    message = stale_rust_extension_message(source_root)

    assert message is not None
    assert "not the compiled extension" in message
    assert str(source_root) in message
    assert REBUILD_COMMAND in message


def test_real_build_script_tracks_directories_and_current_package_root(tmp_path: Path) -> None:
    """Exercise Cargo invalidation and the Rust/Python digest agreement."""
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("Cargo is required to exercise native build-script invalidation")
    source = _write_tree(
        tmp_path / "crate",
        {
            "Cargo.toml": (
                b'[package]\nname = "s2apler_freshness_probe"\nversion = "0.0.0"\nedition = "2021"\n'
                b'[build-dependencies]\nsha2 = "0.10"\n'
            ),
            "src/main.rs": b'fn main() { println!("{}", env!("S2APLER_RUST_SOURCE_DIGEST")); }\n',
        },
    )
    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo / "s2apler_rust/build.rs", source / "build.rs")
    # Reuse the real lockfile so `--offline` resolves sha2 from the local Cargo cache.
    shutil.copyfile(repo / "s2apler_rust/Cargo.lock", source / "Cargo.lock")
    target = tmp_path / "target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYO3_PYTHON"] = sys.executable
    command = [cargo, "run", "--release", "--offline", "--quiet", "--manifest-path", str(source / "Cargo.toml")]

    def assert_current_digest() -> str:
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True, check=False, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        digest = rust_source_digest(source)
        assert result.stdout.strip() == digest
        return digest

    baseline = assert_current_digest()
    added = source / "src/new.rs"
    added.write_bytes(b"// Newly added, unreferenced build input\n")
    assert assert_current_digest() != baseline
    added.unlink()
    assert assert_current_digest() == baseline

    # A compiled build script may be reused with a shared Cargo target. Its
    # digest must use the current manifest directory supplied at execution.
    alternate = tmp_path / "alternate"
    shutil.copytree(source, alternate)
    (alternate / "src/new.rs").write_bytes(b"// Only in the current package\n")
    build_script_name = "build-script-build.exe" if os.name == "nt" else "build-script-build"
    build_script = next((target / "release/build").glob(f"s2apler_freshness_probe-*/{build_script_name}"))
    env["CARGO_MANIFEST_DIR"] = str(alternate)
    env["OUT_DIR"] = str(target / "release/build/direct/out")
    result = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"cargo:rustc-env=S2APLER_RUST_SOURCE_DIGEST={rust_source_digest(alternate)}" in result.stdout

    invalid_source = _unsupported_source_path(alternate)
    invalid_source.write_bytes(b"// unsupported path\n")
    invalid = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert invalid.returncode != 0
    assert "native source paths must be valid UTF-8" in invalid.stderr
    invalid_source.unlink()

    external = tmp_path / "external"
    external.mkdir()
    _link_directory(alternate / "src/linked", external)
    linked = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert linked.returncode != 0
    assert "linked native source directories are unsupported" in linked.stderr


def _write_metadata(root: Path, directory: str, version: str) -> Path:
    metadata = root / directory
    metadata.mkdir(parents=True)
    name = "PKG-INFO" if directory.endswith(".egg-info") else "METADATA"
    (metadata / name).write_text(f"Metadata-Version: 2.1\nName: s2apler\nVersion: {version}\n", encoding="utf-8")
    return root


def test_duplicate_metadata_names_stale_egg_info(tmp_path: Path) -> None:
    site_packages = _write_metadata(tmp_path / "site-packages", "s2apler-1.0.0b6.dist-info", "1.0.0b6")
    checkout = _write_metadata(tmp_path / "checkout", "s2apler.egg-info", "1.0.0b4")

    assert duplicate_package_metadata_message([str(site_packages)]) is None
    # The editable install can list the checkout twice, as "." and as an absolute path.
    assert duplicate_package_metadata_message([str(site_packages), str(site_packages / ".")]) is None
    message = duplicate_package_metadata_message([str(checkout), str(site_packages)])
    assert message is not None
    assert "1.0.0b4 in" in message and "1.0.0b6 in" in message and "s2apler.egg-info" in message
