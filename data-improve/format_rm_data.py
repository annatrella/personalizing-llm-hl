"""
Formatting dataset for training discriminator-based reward model
"""

import os
import json
import pandas as pd
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH, INTERESTS

SPLIT_RANDOM_SEED = 123 # seed for reproducibility
TRAIN_VAL_SPLIT = 0.2 # percentage of original dataset used for validation
JSON_FILE_NAME = "combined_gpt4_responses.json"
LAMP_DF = pd.read_csv(DATA_PATH + '/LaMP7_train.csv', encoding='utf8')
OUTPUT_NAME = "rm_baseline_data.json"
TASK = "Paraphrase the following tweet in the user's persona and speaking style, without changing the underlying facts presented in the original tweet. "

# Load your JSON file
with open(os.path.join(DATA_PATH, JSON_FILE_NAME), 'r') as json_file:
    JSON_DATA = json.load(json_file)

def random_split(data_list):
    N = len(data_list)
    K = int(TRAIN_VAL_SPLIT * N)
    random.Random(SPLIT_RANDOM_SEED).shuffle(data_list)
    val_data = data_list[:K]
    train_data = data_list[K:]

    return train_data, val_data

def write_prompt(obj):
    try:
        personal_interest = obj['personal_interest']
        description = INTERESTS[personal_interest]["description"]
        speaking_style = INTERESTS[personal_interest]["speaking_style"]
    except Exception as e:
      print(f"Error: {e} for Object: {obj}")

    return TASK + f"Tweet: {obj['input']}, Persona: {description}, Speaking Style: {speaking_style}"

def random_selection(personal_interest, index):
    # randomly select another output from different user with same personal interest
    matching_objects = [obj for obj in JSON_DATA if obj.get("personal_interest") == personal_interest and obj.get("index") != index]
    same_group_wrong_meaning = random.choice(matching_objects)["output"]
    # randomly select another output from different user with different personal interest
    other_objects = [obj for obj in JSON_DATA if obj.get("personal_interest") != personal_interest]
    wrong_group_wrong_meaning = random.choice(other_objects)["output"]

    return same_group_wrong_meaning, wrong_group_wrong_meaning

def format_rm_data():
    result = []

    for item in JSON_DATA:
        user_idx = item["index"]
        positive_sample = item["output"]
        # construct user personal interest
        prompt = write_prompt(item)
        # negative samples
        og_output = LAMP_DF['output'][user_idx]
        same_group_wrong_meaning, wrong_group_wrong_meaning = random_selection(item["personal_interest"], user_idx)
        # incorrect user interest group, correct semantic meaning
        result.append({
            "prompt": prompt,
            "chosen": positive_sample,
            "rejected": og_output
        })
        # correct user interest group, different semantic meaning
        result.append({
            "prompt": prompt,
            "chosen": positive_sample,
            "rejected": same_group_wrong_meaning
        })
        # incorrect user interest group, different semantic meaning
        result.append({
            "prompt": prompt,
            "chosen": positive_sample,
            "rejected": wrong_group_wrong_meaning
        })
    train_data, test_data = random_split(result)
    train_test_split_result = {
        "train": train_data,
        "test": test_data
    }

    return train_test_split_result

## BASELINE PROFILE EXTRACTOR
rm_data = format_rm_data()

# Write the data to a JSON file
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
output_filename = os.path.join(DATA_PATH, OUTPUT_NAME)
with open(output_filename, "w") as outfile:
    json.dump(rm_data, outfile, indent=4)

num_train_pts = len(rm_data["train"])
num_test_pts = len(rm_data["test"])
print(f"Num. Training Points: {num_train_pts}")
print(f"Num. Val Points: {num_test_pts}")
print(f"Total Points: {num_train_pts + num_test_pts}")