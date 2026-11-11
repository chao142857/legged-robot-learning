#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/go2_piper_native_10k_20261101
export LD_LIBRARY_PATH="$ROOT/deps/tmux/usr/lib/x86_64-linux-gnu"
export TERM=xterm-256color
exec "$ROOT/deps/tmux/usr/bin/tmux" "$@"
