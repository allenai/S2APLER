"""Importing s2apler must not write to disk or require a data directory."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
from hyperopt import Trials

from s2apler.model import PairwiseModeler

PACKAGE = Path(__file__).resolve().parents[1] / "s2apler"


def installed_layout(root: Path, path_config: str | None) -> Path:
    """Copy the Python package as a wheel would install it: no data/ next to it, unless given."""
    shutil.copytree(PACKAGE, root / "s2apler", ignore=shutil.ignore_patterns("__pycache__"))
    if path_config is not None:
        (root / "data").mkdir()
        (root / "data" / "path_config.json").write_text(path_config, encoding="utf-8")
    return root


def run_import(root: Path, module: str) -> subprocess.CompletedProcess[str]:
    code = f"import sys; sys.path.insert(0, {str(root)!r}); import {module} as m; print(m.__name__)"
    if module == "s2apler.consts":
        code += "; print(m.CONFIG['main_data_dir'])"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_import_without_path_config_writes_nothing(tmp_path):
    root = installed_layout(tmp_path / "site", None)
    # A read-only install directory: import must not try to create data/ in it.
    os.chmod(root, stat.S_IREAD | stat.S_IEXEC)
    try:
        result = run_import(root, "s2apler.consts")
    finally:
        os.chmod(root, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == str(root / "data")
    assert not (root / "data").exists()


def test_unset_path_config_falls_back_to_the_data_directory(tmp_path):
    root = installed_layout(
        tmp_path / "site", '{"main_data_dir": "absolute path of wherever you downloaded the data to"}'
    )
    result = run_import(root, "s2apler.consts")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == str(root / "data")


def test_configured_missing_data_directory_fails_clearly(tmp_path):
    missing = (tmp_path / "nowhere").as_posix()
    root = installed_layout(tmp_path / "site", f'{{"main_data_dir": "{missing}"}}')
    result = run_import(root, "s2apler.consts")
    assert result.returncode != 0
    assert "doesn't exist" in result.stderr and missing in result.stderr


def test_plotting_helpers_import_without_internal_data_dir(tmp_path):
    root = installed_layout(tmp_path / "site", None)
    result = run_import(root, "s2apler.plotting_utils")
    assert result.returncode == 0, result.stderr


def test_fit_without_search_space_clears_old_trials():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    y = (X[:, 0] > 0.5).astype(int)
    modeler = PairwiseModeler(n_iter=2, search_space={}, random_state=0)
    modeler.hyperopt_trials_store = Trials()  # left over from an earlier fit
    assert modeler.fit(X, y, X, y) is None
    assert modeler.hyperopt_trials_store is None
    assert modeler.best_params == {}
