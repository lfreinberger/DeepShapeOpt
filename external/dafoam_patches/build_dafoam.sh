#!/bin/bash
# Build the patched DAFoam (libDASolver + pyDASolvers) in all three modes inside the
# container: plain (primal), ADR (reverse AD: adjoint products) and ADF (forward AD).
# Every mode pyDAFoam loads must contain the same solvers and the same DASolver layout.
#
# The sources are bind-mounted over the image's repo and the libraries land in the
# Apptainer overlay (see README.md); publish them afterwards with
# scripts/publish_dafoam_build.sh.
#
# usage: build_dafoam.sh [build-tree] [modes...]      modes: plain ADR ADF (default: all)
set -eu

B=${1:-/work/lfrei/dafoam-build}                     # has dafoam/ and overlay/
shift || true
MODES=${*:-plain ADR ADF}
SIF=${DAFOAM_SIF:-/usr2/lfrei/containers/dafoam.sif}
HERE=$(cd "$(dirname "$0")" && pwd)

# the powerLawArrhenius model is compiled into libDASolver from DeepShapeOpt's source
ln -sfn "$HERE/../../openfoam/viscosityModels/powerLawArrhenius" \
    "$B/dafoam/src/adjoint/transportModels/powerLawArrhenius"

for mode in $MODES; do
    echo "***************** $mode *****************"
    apptainer exec --cleanenv --overlay "$B/overlay" \
        --bind "/work,/usr2,/workdisk,$B/dafoam:/home/dafoamuser/dafoam/repos/dafoam" \
        "$SIF" bash -c "
        R=/home/dafoamuser/dafoam
        # the environment scripts return non-zero on harmless lookups: no set -e yet
        source \$R/loadDAFoam.sh >/dev/null 2>&1
        if [ $mode != plain ]; then
            sed -i 's/export WM_AD_MODE=.*/export WM_AD_MODE=$mode/' \$R/OpenFOAM/OpenFOAM-AD/etc/bashrc
            source \$R/OpenFOAM/OpenFOAM-AD/etc/bashrc >/dev/null 2>&1
        fi
        set -e
        export WM_QUIET=true
        cd \$R/repos/dafoam/src/adjoint && wmake -j 8 >/dev/null
        cd \$R/repos/dafoam/src/pyDASolvers && ./Allmake >/dev/null
        echo \"built \$(ls \$R/OpenFOAM/sharedLibs/libDASolver\${WM_AD_MODE:-}.so)\"
    "
done
