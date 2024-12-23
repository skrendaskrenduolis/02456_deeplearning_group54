import pandas as pd
import os

current_dir = os.getcwd()
path_list = [f"{current_dir}/Datasets/results_TruthfulQA_new", f"{current_dir}/Datasets/results_TruthfulQA_truthful_new"]

for path in path_list:
    for file in os.listdir(path):
        if ".json" in file:
            df = pd.read_json(f"{path}/{file}")
            print(file[:-5])
            print(df.head())
            df.to_csv(f"{path}/{file[:-5]}.csv")

#counts = df["category"].value_counts()
#print(counts)
