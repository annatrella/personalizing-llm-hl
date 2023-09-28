"""
Reads in csv files and splits data into training, validation, and test sets
"""
import json
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH

TRAIN_VAL_SPLIT = 0.1 # percentage of original training data used for validation
RANDOM_SEED = 123 # seed for reproducibility

def process_obj(obj):
    results = [obj]
    original_input = obj['input']
    original_output = obj['output']
    ptweets = obj["generated_tweets"]
    non_ptweets = obj["original_tweets"]
    for i in range(len(ptweets)):
        json_object = {
            "generated_tweets": [original_output] + ptweets[:i] + ptweets[i+1:],
            "index": obj["index"],
            "personal_interest": obj["personal_interest"],
            "input": non_ptweets[i],
            "output": ptweets[i],
            "original_tweets": [original_input] + non_ptweets[:1] + non_ptweets[i+1:]
        }
        results.append(json_object)

    return results

def create_permutations_json(json):
    processed_json_objects = []
    for obj in json:
        permutations = process_obj(obj)
        for permutation in permutations:
            processed_json_objects.append(permutation)

    return processed_json_objects

def random_sample(data_list):
    N = len(data_list)
    K = int(TRAIN_VAL_SPLIT * N)
    random.Random(RANDOM_SEED).shuffle(data_list)
    val_data = data_list[:K]
    train_data = data_list[K:]

    return train_data, val_data

### we will further process the dataset into train and validation ###
with open(DATA_PATH + '/combined_gpt4_responses.json', "r") as file:
    # Parse the JSON data
    original_training_data = json.load(file)

# first split into train and validation
train_data, val_data = random_sample(original_training_data)

permuted_train = create_permutations_json(train_data)
permuted_val = create_permutations_json(val_data)

with open(DATA_PATH + '/ptweets_permutations_train.json', 'w') as json_file:
    json.dump(permuted_train, json_file, indent=4)

with open(DATA_PATH + '/ptweets_permutations_val.json', 'w') as json_file:
    json.dump(permuted_val, json_file, indent=4)

### TEST ###
with open(DATA_PATH + '/gpt4_tweet_test.json', "r") as file:
    # Parse the JSON data
    test_data = json.load(file)

permuted_test = create_permutations_json(test_data)

with open(DATA_PATH + '/ptweets_permutations_test.json', 'w') as json_file:
    json.dump(permuted_test, json_file, indent=4)

### DATA SET STATS ###
print("Num. data pts. in train: ", len(permuted_train))
print("Num. data pts. in validation: ", len(permuted_val))
print("Num. data pts. in test: ", len(permuted_test))

print("Num. users in train: ", len(train_data))
print("Num. users in validation: ", len(val_data))
print("Num. users in test: ", len(test_data))