import pandas as pd
import os

path_list = ["./results_TruthfulQA", "./results_TruthfulQA_fulltruth"]

for path in path_list:
    for file in os.listdir(path):
        if ".json" in file:
            df = pd.read_json(f"{path}/{file}")
            print(file[:-5])
            print(df.head())
            df.to_csv(f"{path}/{file[:-5]}.csv")

#counts = df["category"].value_counts()
#print(counts)
