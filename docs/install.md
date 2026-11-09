# Installation

## Requirements

- Python `3.11`, `3.12`, or `3.13`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Rust, if you work from a source checkout or build the native extension from source:
  [`rustup`](https://www.rust-lang.org/tools/install)

The `s2apler` wheel includes the native extension `s2apler._s2apler_rust`. Installing a
supported wheel needs no Rust compiler. Source installs compile the extension.

If you build from source, install OS prerequisites first.

Ubuntu / Debian / WSL2:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config
```

Windows: install Visual Studio Build Tools with the `Desktop development with C++` workload.

Toolchain check:

```bash
uv --version
rustc --version
cargo --version
```

## Repo checkout

Install the dependencies into the project environment (`.venv`), building the Rust extension:

```bash
git clone https://github.com/allenai/S2APLER.git
cd S2APLER
uv sync               # runtime
uv sync --extra dev   # runtime plus the test, lint, and build tools
```

## Build the Rust extension from source

`uv sync --extra dev` builds the `s2apler` package and installs it into the project
environment, and syncing `uv run` commands rebuild it after ordinary Rust edits. The test suite
stops before collection if the installed extension was built from different native sources
than the checkout.

If a rebuild is missed, force a refresh:

```bash
uv sync --extra dev --reinstall-package s2apler
```

If a forced reinstall still reports a mismatch (for example after edits that preserve
timestamps, or after switching checkouts that share `CARGO_TARGET_DIR`), clear the crate's
cached release artifacts and repeat the forced refresh:

```bash
uv run --no-sync cargo clean -p s2apler_rust --manifest-path s2apler_rust/Cargo.toml --release
```

Notes:

- If you just installed Rust with `rustup` in the current shell, load its environment first.
- Keep the Python code and native extension from the same checkout.
- Editable builds place the extension inside `s2apler/`. After switching revisions, force a reinstall.

[Native build troubleshooting](development.md#native-build-troubleshooting) explains rebuild
detection and the source digest.

## Running repo scripts

```bash
uv run python path/to/script.py
```

The command rebuilds the extension when its inputs change. Use `uv run --no-sync python` only
in an environment that you already synchronized.
