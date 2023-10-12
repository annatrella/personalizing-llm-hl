import json
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
from global_vars import DATA_PATH, INTEREST_TYPES, INTERESTS


def open_json(file_name):
    with open(file_name, "r") as file:
        # Parse the JSON data
        data = json.load(file)
    return data

def compute_text_metrics(text_lengths):
    return f"{np.mean(text_lengths):.3f} $\pm$ {np.std(text_lengths):.3f}"

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

def get_text_lengths(text):
    tokens = text.split()
    return len(tokens)

def get_prompt_lengths(data):
    result = []
    for obj in data:
        length = get_text_lengths(obj["prompt"])
        result.append(length)
    
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
print(compute_text_metrics(train_context_lengths))
print(compute_text_metrics(val_context_lengths))
print(compute_text_metrics(test_context_lengths))

### input lengths ###
print(compute_text_metrics(train_input_lengths))
print(compute_text_metrics(val_input_lengths))
print(compute_text_metrics(test_input_lengths))

stats_by_interest = []
for interest in INTEREST_TYPES:
    # get data
    interest_data = [block for block in train_data if block["personal_interest"] == interest] \
        + [block for block in val_data if block["personal_interest"] == interest] \
        + [block for block in test_data if block["personal_interest"] == interest]
    # context length
    context_length = compute_text_metrics(get_context_lengths(interest_data))
    # input length
    input_length = compute_text_metrics(get_input_lengths(interest_data))
    # profile length
    profile_text = INTERESTS[interest]["description"] + " " + INTERESTS[interest]["speaking_style"]
    profile_length = get_text_lengths(profile_text)
    stats_by_interest.append({
        "interest": interest,
        "context_length": context_length,
        "input_length": input_length,
        "profile_length": profile_length,
    })

column_order = ["interest", "context_length", "input_length", "profile_length"]

def print_latex_row(data_row, column_order):
    row = " & ".join([str(data_row[column]) for column in column_order]) + " \\\\"
    print(row)
    print("\\hline")

### Print the table header ###
# header = " & ".join(column_order) + " \\\\"
# print(header)
# print("\\hline")

# Print the data rows
for row_data in stats_by_interest:
    print_latex_row(row_data, column_order)

### baseline vs. HL method ###
base_train_data = open_json(DATA_PATH + '/ptweets_base_train.json')
base_val_data = open_json(DATA_PATH + '/ptweets_base_val.json')
base_test_data = open_json(DATA_PATH + '/ptweets_base_test.json')

low_train_data = open_json(DATA_PATH + '/ptweets_low_train.json')
low_val_data = open_json(DATA_PATH + '/ptweets_low_val.json')
low_test_data = open_json(DATA_PATH + '/ptweets_low_test.json')

base_train_lengths = get_prompt_lengths(base_train_data)
base_val_lengths = get_prompt_lengths(base_val_data)
base_test_lengths = get_prompt_lengths(base_test_data)

low_train_lengths = get_prompt_lengths(low_train_data)
low_val_lengths = get_prompt_lengths(low_val_data)
low_test_lengths = get_prompt_lengths(low_test_data)

stats_by_method = []
stats_by_method.append({
    "dataset": "Train",
    "baseline": compute_text_metrics(base_train_lengths),
    "hierarchical": compute_text_metrics(low_train_lengths)
})
stats_by_method.append({
    "dataset": "Validation",
    "baseline": compute_text_metrics(base_val_lengths),
    "hierarchical": compute_text_metrics(low_val_lengths)
})
stats_by_method.append({
    "dataset": "Test",
    "baseline": compute_text_metrics(base_test_lengths),
    "hierarchical": compute_text_metrics(low_test_lengths)
})

column_order = ["dataset", "baseline", "hierarchical"]

# Print the data rows
for row_data in stats_by_method:
    print_latex_row(row_data, column_order)