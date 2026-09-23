#!/bin/bash
# End-to-end smoke runs of the driver (OpenFOAM, DAFoam and a GPU needed): every config in
# tests/smoke, 2 iterations each. Usage: tests/smoke/run_all.sh [name ...]; logs in tests/smoke/logs.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOGS=$ROOT/tests/smoke/logs
mkdir -p "$LOGS"
cd "$ROOT"
[ -f env.sh ] && source env.sh
if [ -z "${WM_PROJECT_DIR:-}" ] && [ -f /programs/shared/openfoam/OpenFOAM-v2506/etc/bashrc ]; then
    source /programs/shared/openfoam/OpenFOAM-v2506/etc/bashrc >/dev/null 2>&1
fi
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(drag_deepsdf_openfoam drag_ffd_openfoam drag_deepsdf_dafoam channel_deepsdf_openfoam channel_noise_probe channel_jacobian_probe channel_ffd_openfoam)
rc_all=0
for name in "${names[@]}"; do
    t0=$(date +%s)
    uv run python scripts/optimize.py --config "tests/smoke/$name.json" > "$LOGS/$name.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && rc_all=1
    echo "$(date '+%F %T') $name rc=$rc elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$LOGS/summary.txt"
done
exit $rc_all
