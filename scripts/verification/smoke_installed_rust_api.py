"""Fail-hard import and ABI smoke for an installed ``s2apler`` native extension."""

from __future__ import annotations

from s2apler import _s2apler_rust as s2apler_rust


def main() -> None:
    """Validate the APIs pair featurization needs without loading other dependencies."""

    build_info = s2apler_rust.get_build_info()
    if build_info.get("num_features") != 18:
        raise RuntimeError(f"s2apler_rust reports {build_info.get('num_features')} features, expected 18")
    if not callable(getattr(s2apler_rust.PaperFeaturizer, "featurize", None)):
        raise RuntimeError("s2apler_rust.PaperFeaturizer.featurize is unavailable")
    print(f"Validated installed s2apler_rust {s2apler_rust.__version__} featurizer ABI")


if __name__ == "__main__":
    main()
