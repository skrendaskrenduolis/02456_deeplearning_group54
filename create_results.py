import pandas as pd
import os
path = "./results_TruthfulQA"
for file in os.listdir(path):
    if ".json" in file:
        df = pd.read_json(f"{path}/{file}")
        print(file[:-5])
        print(df.head())
        df.to_csv(f"{path}/{file[:-5]}.csv")

#counts = df["category"].value_counts()
#print(counts)
