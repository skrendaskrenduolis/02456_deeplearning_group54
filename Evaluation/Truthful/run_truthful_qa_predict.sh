#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- specify gpu memory --
#BSUB -R "select[gpu40gb]"
### -- set the job Name --
#BSUB -J truthful_qa_complete
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:45
# request 4GB of system-memory per core
#BSUB -R "rusage[mem=1.5GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 2GB
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo truthful_qa_complete.out
#BSUB -eo truthful_qa_complete.err
# -- end of LSF options --

# Load the cuda module
module load cuda/12.4

# Activate your conda env (modify path if necessary)
source ~/miniconda3/bin/activate

conda activate $PWD/../deeplearning

# Export huggingface variables if necessary

export HF_HOME="$PWD/../huggingface"
export HF_HUB_CACHE="$PWD/../huggingface/hub"

for file in $(ls $HF_HUB_CACHE | grep 'biomistral_instruct_*'); do
    echo "$HF_HUB_CACHE/$file"
    python3 $PWD/Evaluation//Truthful/run_models_truthful.py --model_name="$HF_HUB_CACHE/$file"
done

for file in $(ls $HF_HUB_CACHE | grep 'internistai_biomistral_*'); do
    echo "$HF_HUB_CACHE/$file"
    python3 $PWD/Evaluation//Truthful/run_models_truthful.py --model_name="$HF_HUB_CACHE/$file"
done

single_model_array=("mistralai/Mistral-7B-Instruct-v0.1" "mistralai/Mistral-7B-Instruct-v0.2" "BioMistral/BioMistral-7B" "internistai/base-7b-v0.2")
for model in "${single_model_array[@]}"; do
    echo "$model"
    python3 $PWD/Evaluation//Truthful/run_models_truthful.py --model_name="$model"
done

#conda deactivate
