import json
import numpy as np
import sys
import os

def open_json(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    return data

EVALS = open_json("../results/gpt4_evals.json")

def get_evals(evals_json):
    result = []
    for obj in evals_json:
        result.append(int(obj["chosen_model_index"]))
    
    return np.array(result)

evals = get_evals(EVALS)

print(f"Baseline: {len(evals[np.where(evals == 1)]) / len(evals)}")
print(f"SFT: {len(evals[np.where(evals == 2)]) / len(evals)}")
print(f"RLHF: {len(evals[np.where(evals == 3)]) / len(evals)}")