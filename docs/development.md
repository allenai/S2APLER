# Development

## Core commands

```bash
uv sync --extra dev
uv run pytest -q -n auto
uv run ruff check .
uv run ruff format .
uv run ty check
```

`ty` reads its defaults from `[tool.ty]` in `pyproject.toml`: Python 3.11 and the `linux`
platform used by the GitHub runners. Pass `--python-platform windows` for a Windows check.

## Pair features in Rust

`s2apler.featurizer.many_pairs_featurize` computes pair features in Rust by default.
Python still preprocesses each paper (LaTeX, transliteration, name normalization) and keeps
the texts that n-grams come from. `rust_paper_record` packs each paper once, with those texts as
UTF-32 code points; the extension counts the n-grams and computes all 18 features per pair in
parallel. `_single_pair_featurize` stays as the reference implementation: pass
`backend="python"` to use it. It builds Python `Counter`s from the same texts with
`s2apler.data.with_counters`. `tests/test_rust_featurizer.py` requires the two to agree bit for
bit, NaNs included, on raw Unicode titles and names too.

`Clusterer.make_distance_matrices` handles pairs as arrays: `PDData.constraint_values` applies
`get_constraint`'s rules to whole blocks, and `tests/test_distance_matrices.py` checks both
against the per-pair path. The native scorer scores each distinct row of split bins once.

The counter, string-similarity, thread-pool and build-script code comes from S2AND's native
extension; `s2apler_rust/THIRD_PARTY_NOTICES.md` lists it.

## Native build troubleshooting

[install.md](install.md#build-the-rust-extension-from-source) has the recovery commands. This
section explains why they are sometimes needed.

Syncing `uv run` decides whether to rebuild `s2apler` from the `cache-keys` list in
`pyproject.toml`. Packaged license and notice edits also invalidate the cached distribution.

uv's file-glob keys track the newest matching timestamp, not file contents or the complete file
inventory. Deleting an older file, adding a backdated file, or changing content while
preserving timestamps can therefore miss a rebuild.

The extension embeds a content digest of its native build inputs (`Cargo.toml`, `Cargo.lock`,
`build.rs` and the Rust sources). `tests/conftest.py` recomputes it from the checkout and stops
before collection when they differ, for example after `uv run --no-sync` following a Rust
change. Native input paths must be valid UTF-8, and source directories must not be symlinks or
junctions.

`uv run --active --no-project maturin develop -m s2apler_rust/Cargo.toml` installs an editable
build; the next syncing `uv sync` or `uv run` replaces it with the locked build.

## Local CI mirror

```bash
uv run --no-project python scripts/run_ci_locally.py
```

`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml`:

- lint: `scripts/sync_version.py --check`, `ruff check` and `ruff format --check`, with the
  exact Ruff pin from the `dev` extra in an isolated uv tool environment;
- typecheck-and-test, in order:
  - sync the locked environment, building `s2apler` from source;
  - run `cargo fmt`, Clippy's correctness and suspicious lints, and the Rust unit tests;
  - smoke-test the installed extension and run `ty`;
  - run the full pytest suite with coverage under pytest-xdist (`-n auto`, at most eight
    workers). Pass `--pytest-workers N` to choose a count, or `0` to run in one process.

Run one job with:

```bash
uv run --no-project python scripts/run_ci_locally.py lint
uv run --no-project python scripts/run_ci_locally.py typecheck-and-test
```

PR CI runs Python 3.11–3.13 on Ubuntu and Python 3.11 on Windows and macOS.

## Version bumping

`VERSION` holds the version. One-time hook setup:

```bash
git config core.hooksPath .githooks
```

Bump the version:

```bash
# 1) edit VERSION
# 2) sync manifests and both lockfiles
uv run python scripts/sync_version.py
# 3) rebuild the environment for the new version
uv sync --extra dev
```

- The pre-commit hook runs only when `VERSION` is staged; it runs steps 2 and 3 and stages the
  updated files.
- `scripts/sync_version.py` rewrites `pyproject.toml`, `s2apler_rust/Cargo.toml`, and the
  package entries in `s2apler_rust/Cargo.lock` and `uv.lock`. Cargo spells pre-releases
  `0.3.0-beta.1` where Python uses `0.3.0b1`.
- The loader rejects a native binary whose version differs from the installed package.
