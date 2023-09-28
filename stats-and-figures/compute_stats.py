import json
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
from global_vars import DATA_PATH


def open_json(file_name):
    with open(file_name, "r") as file:
        # Parse the JSON data
        data = json.load(file)
    return data

def compute_text_metrics(text_lengths):
    print(f"Mean and Var: {np.mean(text_lengths):.3f} $\pm$ {np.std(text_lengths):.3f}")

def get_context_lengths(data):
    result = []
    for obj in data:
        tokens = " ".join(obj["generated_tweets"]).split()
        result.append(len(tokens))

    return result

def get_input_lengths(data):
    result = []
    for obj in data:
        tokens = obj["input"].split()
        result.append(len(tokens))
        
    return result

train_data = open_json(DATA_PATH + '/ptweets_permutations_train.json')
val_data = open_json(DATA_PATH + '/ptweets_permutations_val.json')
test_data = open_json(DATA_PATH + '/ptweets_permutations_test.json')

train_context_lengths = get_context_lengths(train_data)
val_context_lengths = get_context_lengths(val_data)
test_context_lengths = get_context_lengths(test_data)

train_input_lengths = get_input_lengths(train_data)
val_input_lengths = get_input_lengths(val_data)
test_input_lengths = get_input_lengths(test_data)

### context lengths ###
compute_text_metrics(train_context_lengths)
compute_text_metrics(val_context_lengths)
compute_text_metrics(test_context_lengths)

### input lengths ###
compute_text_metrics(train_input_lengths)
compute_text_metrics(val_input_lengths)
compute_text_metrics(test_input_lengths)