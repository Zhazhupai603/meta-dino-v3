#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "${SCRIPT_DIR}"

echo "[1/2] Train artifact localizer"
bash run_artifact_localizer.sh "$@"

echo "[2/2] Train artifact patch classifier"
bash run_artifact_patch_cls.sh "$@"
