"""
Reads in csv files and splits data into training, validation, and test sets
"""
import json
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH, INTERESTS, INTEREST_IDXS

TASK = "Paraphrase the following tweet in the user's persona and speaking style, without changing the underlying facts presented in the original tweet. "

def process_low(obj):
    try:
        personal_interest = obj['personal_interest']
        description = INTERESTS[personal_interest]["description"]
        speaking_style = INTERESTS[personal_interest]["speaking_style"]
        json_object = {
            'prompt': TASK + f"Tweet: {obj['input']}, Persona: {description}, Speaking Style: {speaking_style}",
            'output': obj['output']
        }
    except Exception as e:
      print(f"Error: {e} for Object: {obj}")

    return json_object

def process_high(obj):
    try:
        personal_interest = obj['personal_interest']
        person_interest_idx = INTEREST_IDXS[personal_interest]
        json_object = {
            'input': "".join(obj["generated_tweets"]),
            'label': person_interest_idx
        }
    except Exception as e:
      print(f"Error: {e} for Object: {obj}")

    return json_object

def iterate_and_process_json(json):
    high_objects = []
    low_objects = []

    for obj in json:
        high_objects.append(process_high(obj))
        low_objects.append(process_low(obj))

    return high_objects, low_objects

### we will further process the dataset into train and validation ###
with open(DATA_PATH + '/ptweets_permutations_train.json', "r") as file:
    # Parse the JSON data
    original_training_data = json.load(file)

with open(DATA_PATH + '/ptweets_permutations_val.json', "r") as file:
    # Parse the JSON data
    original_val_data = json.load(file)

train_high, train_low = iterate_and_process_json(original_training_data)
val_high, val_low = iterate_and_process_json(original_val_data)

with open(DATA_PATH + '/ptweets_high_train.json', 'w') as json_file:
    json.dump(train_high, json_file, indent=4)

with open(DATA_PATH + '/ptweets_high_val.json', 'w') as json_file:
    json.dump(val_high, json_file, indent=4)

with open(DATA_PATH + '/ptweets_low_train.json', 'w') as json_file:
    json.dump(train_low, json_file, indent=4)

with open(DATA_PATH + '/ptweets_low_val.json', 'w') as json_file:
    json.dump(val_low, json_file, indent=4)

### hold out test set ###
with open(DATA_PATH + '/ptweets_permutations_test.json', "r") as file:
    # Parse the JSON data
    original_test_data =  json.load(file)
test_high, test_low = iterate_and_process_json(original_test_data)

with open(DATA_PATH + '/ptweets_high_test.json', 'w') as json_file:
    json.dump(test_high, json_file, indent=4)

with open(DATA_PATH + '/ptweets_low_test.json', 'w') as json_file:
    json.dump(test_low, json_file, indent=4)

print("Num. data pts. in high train: ", len(train_high))
print("Num. data pts. in high validation: ", len(val_high))
print("Num. data pts. in low test: ", len(test_high))

print("Num. data pts. in low train: ", len(train_low))
print("Num. data pts. in low validation: ", len(val_low))
print("Num. data pts. in low test: ", len(test_low))