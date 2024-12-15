# This code is a modified version of https://github.com/BioMistral/BioMistral/blob/main/Evaluation/TruthfulQA/PredictTruthfulQA.py
# adapted for the TruthfulQA dataset https://huggingface.co/datasets/truthfulqa/truthful_qa
# import libraries
import json
import torch
import argparse
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm
from pqdm.processes import pqdm
from random import seed
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

# import dataset matched to replicate paper
# FIX UNRESOLVED IMPORT
from edit_dataset_improved import merged_dataset, merged_dataset_truthful

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 10

def process(data):

    if "llama" in full_name.lower() :
        tokenizer_thread = LlamaTokenizer.from_pretrained(full_name, trust_remote_code=True)
    else:
        tokenizer_thread = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)
    
    results = []

    for current_data in tqdm(data):

        scores = F.softmax(current_data["scores"], dim=-1)
        max_len = scores.size(dim=1)
        top_k = torch.topk(scores, max_len)

        probs = top_k.values[0].cpu()
        token_str = [tokenizer_thread.decode(top_k.indices[0].cpu()[i]) for i in range(max_len)]
        
        kv = [{"token_str": ts, "probs": tp.item()} for tp, ts in zip(probs, token_str) if ts in current_data["classes"]]

        predictions = {}
        for pair in kv:            
            if pair["token_str"] not in predictions:
                predictions[pair["token_str"]] = pair["probs"]

        results.append({
            "question_id": current_data["question_id"],
            "category": current_data["category"],
            "correct_letter": current_data["correct_letter"],
            "predictions": predictions,
            "idk_letter": current_data["idk_letter"],
            "best_prediction": token_str[0]
        })
    
    return results

def divide_chunks(l, n):
    output_chunks = []
    for i in range(0, len(l), n):  
        output_chunks.append(l[i:i + n])
    return output_chunks

full_name = args["model_name"].rstrip("/")
short_model_name = full_name.split("/")[-1].replace("_","-")
print(short_model_name)

if "llama" in full_name.lower():
    model = LlamaForCausalLM.from_pretrained(full_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = LlamaTokenizer.from_pretrained(full_name, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(full_name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)




torch.set_default_device("cuda")
for i in range(0,3):
    ## Run normal variation
    seed()

    data_threads = []

    with torch.no_grad():    
        for d in tqdm(merged_dataset):
        
            inputs = tokenizer(d[f"formatted_question_choices"], return_tensors = "pt")

            input_ids = inputs["input_ids"].to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            data_threads.append({"scores": outputs.scores[0].to("cpu"), "question_id":d["question_id"], "category": d["category"], "correct_letter": d["correct_answer"], "idk_letter": d["idk_answer"], "classes": d["classes"]})

    data_batches = list(divide_chunks(data_threads, THREADS_NBR))

    all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')
    print(all_thread_result)

    all_results = []
    for thread_result in all_thread_result:
        all_results.extend(thread_result)
    print("Total elements processed: ", len(all_results))

    dir_name = "./results_TruthfulQA_new"

    if not os.path.exists(dir_name):
        # if tmp exists and was not removed, remove it
        #shutil.rmtree(dir_name)
        os.mkdir(dir_name)

    with open(f"{dir_name}/results_{short_model_name}_{i}.json", 'w') as f_out:
        json.dump(all_results, f_out)


    # Run truthful variation
    data_threads = []

    #x = 0
    with torch.no_grad():    
        for d in tqdm(merged_dataset_truthful):
        
            inputs = tokenizer(d[f"formatted_question_choices_truth"], return_tensors = "pt")

            input_ids = inputs["input_ids"].to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            data_threads.append({"scores": outputs.scores[0].to("cpu"), "question_id":d["question_id"], "category": d["category"], "correct_letter": d["correct_answer"], "idk_letter": d["idk_answer"], "classes": d["classes"]})


    data_batches = list(divide_chunks(data_threads, THREADS_NBR))

    all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')
    print(all_thread_result)

    all_results = []
    for thread_result in all_thread_result:
        all_results.extend(thread_result)
    print("Total elements processed: ", len(all_results))

    dir_name = "./results_TruthfulQA_truthful_new"

    if not os.path.exists(dir_name):
        # if tmp exists and was not removed, remove it
        #shutil.rmtree(dir_name)
        os.mkdir(dir_name)

    with open(f"{dir_name}/results_{short_model_name}_{i}_truthful.json", 'w') as f_out:
        json.dump(all_results, f_out)
