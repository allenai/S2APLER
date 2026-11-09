# s2apler-rust

Rust extension module for `s2apler`. It computes S2APLER's pair features.

## Distribution

The root `s2apler` wheel includes `s2apler._s2apler_rust`.
This directory holds the Rust crate, not a separate Python distribution.
Build and install from the repository root with `uv sync --extra dev`.
Build release wheels and the source archive with `uv build` from the root.

## Local dev build

From the repository root (not this directory):

```bash
uv sync --extra dev
```

`uv sync` builds the package with the crate in `s2apler_rust/`, and `uv run` uses timestamps
to detect ordinary build-input edits. The test guard checks the Rust build inputs by content,
including additions and deletions that uv's timestamp keys can miss. See
[build recovery](../docs/install.md#build-the-rust-extension-from-source) for forced
reinstall and Cargo cache-cleaning commands.
