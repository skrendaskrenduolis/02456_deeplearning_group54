#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- specify gpu memory --
#BSUB -R "select[gpu32gb]"
### -- set the job Name --
#BSUB -J merge_models
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00
# request 8GB of system-memory per core
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 8GB
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo merge_models.out
#BSUB -eo merge_models.err
# -- end of LSF options --

# Load the cuda module
module load cuda/12.4


source ~/miniconda3/bin/activate
# conda activate $PWD/../deeplearning

# export HF_HOME="$PWD/../huggingface"

for file in "$PWD/Merges/truthful_merges"/*; do
    name=$(echo "$file" | gawk -F "/" '{sub(/\.yml$/, "", $NF); print $NF}')
    echo $name
    mergekit-yaml "$PWD/Merges/truthful_merges/$name.yml" "$HF_HOME/hub/$name/" --cuda
done

#conda deactivate
