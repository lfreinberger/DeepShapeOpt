# Environment of the DeepShapeOpt runs (bash). Copy to env.sh, adjust the paths, `source env.sh`.
export DEEPSHAPEOPT_DATA_DIR=data                       # input geometries (data/shapes/*.stl), relative to the repo root
export DEEPSHAPEOPT_MODEL_DIR=$HOME/ProjectsPhD/DeepSDFStruct/DeepSDFStruct/trained_models
export DEEPSHAPEOPT_RESULTS_DIR=/storage/$USER/DeepShapeOpt   # heavy debug exports (VTK / STL series)
export DEEPSHAPEOPT_SCRATCH_DIR=/work/$USER/DeepShapeOpt      # transient solver cases (node-local, fast)
export DAFOAM_SIF=$HOME/containers/dafoam.sif             # DAFoam Apptainer image
export DAFOAM_BUILD_ROOT=$HOME/dafoam-build               # published patched DAFoam build (pc_mode fvmatrix)
