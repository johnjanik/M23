#!/usr/bin/env bash
# Fetch the data accompanying [HJLPPZ26] (not redistributed here).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/m23isgalois/.git ]; then
  git clone --depth 1 https://github.com/shaowuz/m23isgalois external/m23isgalois
fi
git -C external/m23isgalois log -1 --format='m23isgalois at %H (%ad)'
