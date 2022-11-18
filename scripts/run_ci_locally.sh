# Script for running the CI locally before making a PR
pytest tests/
flake8 s2apler
flake8 scripts/*.py
black s2apler --check --line-length 120
black scripts/*.py --check --line-length 120
pytest tests/ --cov s2apler --cov-fail-under=40