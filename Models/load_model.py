import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# check the availibility of GPU
print("GPU is available:", torch.cuda.is_available())


# define a public function to load and save model from huggingface
def load_and_save_model(model_name, save_path, cache_dir):
    """
    load pretained model and save into the specified path

    Args:
        model_name (str): the name of pretrained model
        save_path (str): the path to save and model and tokenizer
        cache_dir (str): the path to cache model and tokenizer
    """

    print(f"Start loading model {model_name}")

    # load tokenizer and save
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(save_path)

    # load model and save
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.save_pretrained(save_path)

    print(f"Model and tokenizer for '{model_name}' saved to '{save_path}'")


# define save and cach path
model_dir = "/dtu/blackhole/"
model_save_dir = model_dir + "save/"
model_cache_dir = model_dir + "cache/"

# load and save BioMistral 7B
BioMistral_name = "BioMistral/BioMistral-7B"
BioMistral_save_path = f"{model_save_dir}/BioMistral_BioMistral-7B"
load_and_save_model(BioMistral_name, BioMistral_save_path, model_cache_dir)

# load and save Mistral 7B Instruct
Instruct_name = "mistralai/Mistral-7B-Instruct-v0.1"
Instruct_save_path = f"{model_save_dir}/mistralai_Mistral-7B-Instruct-v0.1"
load_and_save_model(Instruct_name, Instruct_save_path, model_cache_dir)
