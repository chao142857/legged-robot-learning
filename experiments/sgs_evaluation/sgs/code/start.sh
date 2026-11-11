#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/go2_piper_sgs_20261102
export LD_LIBRARY_PATH=/data/go2_piper_native_10k_20261101/deps/tmux/usr/lib/x86_64-linux-gnu
export TERM=xterm-256color
TMUX_BIN=/data/go2_piper_native_10k_20261101/deps/tmux/usr/bin/tmux
if "$TMUX_BIN" list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq go2-sgs32k; then exit 1; fi
"$TMUX_BIN" new-session -d -s go2-sgs32k -n controller -c "$ROOT" "bash -c 'unset LD_LIBRARY_PATH; exec python3 -u /data/go2_piper_sgs_20261102/code/pipeline.py >> /data/go2_piper_sgs_20261102/pipeline.log 2>&1'"
"$TMUX_BIN" set-option -t go2-sgs32k remain-on-exit on
"$TMUX_BIN" new-window -t go2-sgs32k -n monitor "tail -F /data/go2_piper_sgs_20261102/pipeline.log"
"$TMUX_BIN" list-sessions
