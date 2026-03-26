#!/usr/bin/env bash

set -euo pipefail

BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${BACKEND_ROOT}/.." && pwd)"
FRONTEND_DIR="${DATAFACTORY_FRONTEND_DIR:-${WORKSPACE_ROOT}/DataFactory-Frontend}"
FRONTEND_REPO="${DATAFACTORY_FRONTEND_REPO:-https://github.com/SongY123/DataFactory-Frontend.git}"
FRONTEND_UPDATE="${DATAFACTORY_FRONTEND_UPDATE:-0}"

echo "[package] backend root: ${BACKEND_ROOT}"
echo "[package] frontend dir: ${FRONTEND_DIR}"

if [[ -d "${FRONTEND_DIR}/.git" ]]; then
  if [[ "${FRONTEND_UPDATE}" == "1" ]]; then
    if ! git -C "${FRONTEND_DIR}" diff --quiet || ! git -C "${FRONTEND_DIR}" diff --cached --quiet; then
      echo "[package] frontend repo has local changes; refusing to pull with DATAFACTORY_FRONTEND_UPDATE=1" >&2
      exit 1
    fi
    echo "[package] updating existing frontend checkout"
    git -C "${FRONTEND_DIR}" pull --ff-only
  else
    echo "[package] using existing frontend checkout without pulling"
  fi
else
  echo "[package] cloning frontend repo: ${FRONTEND_REPO}"
  git clone "${FRONTEND_REPO}" "${FRONTEND_DIR}"
fi

pushd "${FRONTEND_DIR}" >/dev/null

echo "[package] installing frontend dependencies"
npm ci

echo "[package] packaging Electron desktop app"
npm run desktop:package

echo "[package] build outputs"
find "${FRONTEND_DIR}/dist_electron" -maxdepth 3 -print | sed -n '1,120p'

popd >/dev/null

