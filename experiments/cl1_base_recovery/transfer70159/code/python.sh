#!/usr/bin/env bash
set -euo pipefail
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONSTARTUP
export PYTHONPATH=/workspace/code:/e2reference/code:/native/LeggedManip_Lab/source/LeggedManip_Lab
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CONDA_PREFIX=/home/alex/dog/go2_piper_env
export CONDA_DEFAULT_ENV=go2_piper_env
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export OMP_NUM_THREADS=8 ISAACSIM_ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
exec "$CONDA_PREFIX/bin/python" "$@"
