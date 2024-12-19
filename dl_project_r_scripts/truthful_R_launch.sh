#!/bin/bash
RSCRIPT_PATH="$(which Rscript)"

echo "${RSCRIPT_PATH}"
$RSCRIPT_PATH dl_project_r_scripts/R/truthful_plots.R 
