# Releasing s2apler

One `s2apler` distribution holds the Python package and the native extension. A release builds
one source archive and wheels for Linux (manylinux x86_64 and aarch64, musllinux x86_64), macOS
(universal2) and Windows, each for Python 3.11, 3.12 and 3.13.

## Publish a release

1. Set `VERSION`, run `uv run python scripts/sync_version.py`, then `uv sync --extra dev`.
2. Run `uv run --no-project python scripts/run_ci_locally.py`, and merge the change to `main`.
3. Dispatch `.github/workflows/release-rust.yml` from the reviewed `main` commit.
   - Its validate job runs `scripts/run_ci_locally.py`, the same checks as pull-request CI.
   - It fails when PyPI already has the `s2apler` version.
   - Every wheel and the source archive must keep their license notices
     (`scripts/verification/verify_rust_distribution_licenses.py`) and import in a clean
     environment (`scripts/verification/smoke_installed_rust_api.py`).
   - The release smoke installs the Linux wheels outside the checkout and compares Rust with
     Python features (`scripts/verification/smoke_installed_featurizer.py`).
4. Approve the protected `pypi` environment after checking the workflow commit. The publish job
   uploads with PyPI trusted publishing, which the `s2apler` PyPI project must allow for this
   repository and workflow.

## First release

`s2apler` is not on PyPI yet, so before the first dispatch:

1. Create a `pypi` environment in the GitHub repository settings, with required reviewers.
2. On PyPI, add a pending trusted publisher for the `s2apler` project: owner `allenai`,
   repository `S2APLER`, workflow `release-rust.yml`, environment `pypi`.

## Beta pre-releases

A beta lets integrators test a build before a final release.

1. Set `VERSION` to a PEP 440 pre-release such as `0.3.0b1`, and run
   `uv run python scripts/sync_version.py`; Cargo files receive `0.3.0-beta.1`.
2. Push the branch and dispatch `release-rust.yml` on it. A pre-release builds and smoke-tests
   on any branch; the publish job runs only on `main`.
3. Download the run's `release-distributions` artifact, and confirm every file carries the beta
   version.
4. Confirm that `https://pypi.org/pypi/s2apler/<beta version>/json` returns 404. PyPI adds files
   to an existing release, so a reused version would mix two builds. Upload the distributions
   with `uv publish` and an API token.
5. Install `s2apler==<beta version>` in an empty environment and cluster one real request.

Installers skip pre-releases unless the requirement names one. Set `VERSION` back to the final
version before the final release.
