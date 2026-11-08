#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONSTARTUP
export PYTHONPATH="$TASK_ROOT/code:/home/alex/dog/LeggedManip_Lab_upstream_wbc_20261023/source/LeggedManip_Lab"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX=/home/alex/dog/go2_piper_env
export CONDA_DEFAULT_ENV=go2_piper_env
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export OMP_NUM_THREADS=8 ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
exec "$CONDA_PREFIX/bin/python" "$@"
