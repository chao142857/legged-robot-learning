#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/deps/tmux/usr/lib/x86_64-linux-gnu"
export TERM=xterm-256color
exec "$ROOT/deps/tmux/usr/bin/tmux" "$@"
