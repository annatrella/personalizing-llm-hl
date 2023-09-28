"""
Randomly samples 1000 data points from model outputs on the test data for model-based evaluation
"""
import random
import json
import os

OUTPUTS_PATH = "/home/ec2-user/personalizing-llm-hl/model_outputs"
RANDOM_SEED = 123 # seed for reproducibility
K = 1000 # number of data points for model-based evaluations
baseline_file = os.path.join(OUTPUTS_PATH, "ptweets-baseline/model_outputs.json")
low_file= os.path.join(OUTPUTS_PATH, "ptweets-low/model_outputs.json")
rl_file = os.path.join(OUTPUTS_PATH, "ptweets-rl-low/model_outputs.json")

def get_json(file_name):
    with open(file_name, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def write_json(data, file_name):
    if not os.path.exists(OUTPUTS_PATH):
        os.makedirs(OUTPUTS_PATH)
    output_filename = os.path.join(OUTPUTS_PATH, file_name)
    with open(output_filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

def get_k_samples(data_list, k=K):
    random.Random(RANDOM_SEED).shuffle(data_list)

    return data_list[:k]

baseline_outs = get_json(baseline_file)
low_outs = get_json(low_file)
rl_outs = get_json(rl_file)
write_json(get_k_samples(baseline_outs, k=K), "ptweets-baseline/base_model_based_evals.json")
write_json(get_k_samples(low_outs, k=K), "ptweets-low/low_model_based_evals.json")
write_json(get_k_samples(rl_outs, k=K), "ptweets-rl-low/rl_model_based_evals.json")