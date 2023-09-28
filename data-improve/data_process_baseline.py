"""
Reads in csv files and splits data into training, validation, and test sets
"""
import json
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH

TASK = "I will give you a list of previous tweets from the user, paraphrase the following tweet in the user's style. "

def process_obj(obj):
    try:
        json_object = {
            'prompt': TASK + f"Tweet: {obj['input']}, Previous Tweets: {obj['generated_tweets']}",
            'output': obj['output']
        }
    except Exception as e:
      print(f"Error: {e} for Object: {obj}")

    return json_object

def iterate_and_process_json(json):
    processed_json_objects = []

    for obj in json:
        processed_json_objects.append(process_obj(obj))

    return processed_json_objects

### we will further process the dataset into train and validation ###
with open(DATA_PATH + '/ptweets_permutations_train.json', "r") as file:
    # Parse the JSON data
    original_training_data = json.load(file)

with open(DATA_PATH + '/ptweets_permutations_val.json', "r") as file:
    # Parse the JSON data
    original_val_data = json.load(file)

train_result = iterate_and_process_json(original_training_data)
val_result = iterate_and_process_json(original_val_data)

with open(DATA_PATH + '/ptweets_base_train.json', 'w') as json_file:
    json.dump(train_result, json_file, indent=4)

with open(DATA_PATH + '/ptweets_base_val.json', 'w') as json_file:
    json.dump(val_result, json_file, indent=4)

### hold out test set ###
with open(DATA_PATH + '/ptweets_permutations_test.json', "r") as file:
    # Parse the JSON data
    original_test_data = json.load(file)

test_result = iterate_and_process_json(original_test_data)

with open(DATA_PATH + '/ptweets_base_test.json', 'w') as json_file:
    json.dump(test_result, json_file, indent=4)

print("Num. data pts. in train: ", len(train_result))
print("Num. data pts. in validation: ", len(val_result))
print("Num. data pts. in test: ", len(test_result))