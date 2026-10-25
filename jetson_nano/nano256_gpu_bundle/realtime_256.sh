#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/nano256_realtime/app/nano256_realtime" \
  --provider CUDAExecutionProvider "$@"
