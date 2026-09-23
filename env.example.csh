# Environment of the DeepShapeOpt runs (tcsh). Copy to env.csh, adjust the paths, `source env.csh`.
setenv DEEPSHAPEOPT_DATA_DIR data
setenv DEEPSHAPEOPT_MODEL_DIR $HOME/ProjectsPhD/DeepSDFStruct/DeepSDFStruct/trained_models
setenv DEEPSHAPEOPT_RESULTS_DIR /storage/$USER/DeepShapeOpt
setenv DEEPSHAPEOPT_SCRATCH_DIR /work/$USER/DeepShapeOpt
setenv DAFOAM_SIF $HOME/containers/dafoam.sif
setenv DAFOAM_BUILD_ROOT $HOME/dafoam-build
