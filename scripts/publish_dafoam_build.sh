#!/bin/bash
# Publish the patched DAFoam build to shared storage so cluster nodes can use it.
#
# Why: the build tree lives on node-local scratch (/workdisk on neptune), which a slurm
# job on a gpu node cannot see -- an sbatch run fails with "build_source is not a
# directory". This copies the two things a RUN needs onto NFS:
#
#   <root>/dafoam/      the patched python package (PYTHONPATH -> import dafoam)
#   <root>/sharedLibs/  the container's shared libs with our patched libDASolver.so
#
# The full source tree with its object files stays on local scratch; only building needs
# it, and building on NFS is slow.
#
# usage: publish_dafoam_build.sh [build-tree] [publish-root]
set -eu

SRC=${1:-/work/lfrei/dafoam-build}                     # has dafoam/ and overlay/
DST=${2:-${DAFOAM_BUILD_ROOT:-/usr2/lfrei/dafoam-build}}
SIF=${DAFOAM_SIF:-/usr2/lfrei/containers/dafoam.sif}
CONTAINER_LIBS=/home/dafoamuser/dafoam/OpenFOAM/sharedLibs

[ -d "$SRC/dafoam/dafoam" ] || { echo "no patched package at $SRC/dafoam/dafoam"; exit 1; }

mkdir -p "$DST/sharedLibs"

# 1. the container's stock shared libs (only the ones we do not replace)
echo "copying stock shared libs from the image ..."
apptainer exec --cleanenv --bind "$DST:/publish" "$SIF" \
    bash -c "cp -a $CONTAINER_LIBS/. /publish/sharedLibs/"

# 2. our patched libDASolver.so on top (written into the build overlay by wmake)
PATCHED=$(find "$SRC/overlay" -name libDASolver.so -print -quit 2>/dev/null || true)
[ -n "$PATCHED" ] || { echo "no patched libDASolver.so in $SRC/overlay -- build first"; exit 1; }
cp -a "$PATCHED" "$DST/sharedLibs/libDASolver.so"
echo "patched libDASolver.so: $PATCHED"

# 3. the patched python package
rm -rf "$DST/dafoam"
cp -a "$SRC/dafoam/dafoam" "$DST/dafoam"
find "$DST/dafoam" -name __pycache__ -type d -prune -exec rm -rf {} +

echo
echo "published to $DST"
du -sh "$DST/sharedLibs" "$DST/dafoam"
echo
echo "point configs at it with DAFOAM_BUILD_ROOT=$DST (see tests/env.sh),"
echo "or optimization.dafoam.build_root in the config."
