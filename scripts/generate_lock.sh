#!/usr/bin/env bash
# Run from a network-enabled computer, review the resulting lock, commit uv.lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv lock --python 3.12
# Install the locally testable stack first. The Digital RF native extension and
# MultiROCKET full environment can be installed afterward with --extra all.
uv sync --python 3.12 --locked --extra dev --extra xgboost --extra autoencoder --extra serve
printf '\nLockfile ready: %s/uv.lock\nCommit uv.lock to Git for dependency reproducibility.\n' "$PWD"
