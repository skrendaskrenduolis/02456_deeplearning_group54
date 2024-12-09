#!/bin/sh
# Load the cuda module
module load cuda/12.4

source ~/miniconda3/bin/activate
conda activate /work3/s222858/deeplearning

export HF_HOME="/work3/s222858/huggingface"
export HF_HUB_CACHE="/work3/s222858/huggingface/hub"

for file in $(ls $HF_HUB_CACHE | grep 'biomistral_instruct_*'); do
    echo "$HF_HUB_CACHE/$file"
    python3 run_models_truthful.py --model_name="$HF_HUB_CACHE/$file"
done

for file in $(ls $HF_HUB_CACHE | grep 'internistai_biomistral_*'); do
    echo "$HF_HUB_CACHE/$file"
    python3 run_models_truthful.py --model_name="$HF_HUB_CACHE/$file"
done

single_model_array=("mistralai/Mistral-7B-Instruct-v0.1" "mistralai/Mistral-7B-Instruct-v0.2" "BioMistral/BioMistral-7B" "internistai/base-7b-v0.2")
for model in "${single_model_array[@]}"; do
    echo "$model"
    python3 run_models_truthful.py --model_name="$model"
done
