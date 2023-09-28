"""
Use GPT to evaluate which model output is the best
"""
import os
import openai
import json
import datetime
import pickle
import sys
import os

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(grandparent_dir)
from global_vars import MAIN_PATH

def open_json(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    return data

BASELINE_OUTPUTS = open_json(MAIN_PATH + "/model_outputs/ptweets-baseline/base_model_based_evals.json")
SFT_OUTPUTS = open_json(MAIN_PATH + "/model_outputs/ptweets-low/low_model_based_evals.json")
RL_OUTPUTS = open_json(MAIN_PATH + "/model_outputs/ptweets-rl-low/rl_model_based_evals.json")

assert len(BASELINE_OUTPUTS) == len(SFT_OUTPUTS) and len(SFT_OUTPUTS) == len(RL_OUTPUTS)
OUTPUT_PATH = "../results"

# path where we will be reading and writing to
prompt_form_1 = "Choose the best tweet in terms of how well the generated tweet matches the persona and speaking style in the prompt and how well the generated tweet is semantically and contextually relevant to the original tweet in the prompt. no prose, only return 1, 2, or 3. \n"
prompt_form_2 = "Input: {{\"Task\": \"Paraphrase the following tweet in the user's persona and speaking style, without changing the underlying facts presented in the original tweet.\", "
prompt_form_3 = "\"Persona\": \"{}\", \"Original Tweet\": \"{}\", \"Paraphrase 1\": \"{}\", \"Paraphrase 2\": \"{}\", \"Paraphrase 3\": \"{}\"}}"

prompt_template = prompt_form_1 + prompt_form_2 + prompt_form_3

# 1 is baseline, 2 is sft, 3 is rlhf
def create_prompt(index):
   prompt = SFT_OUTPUTS[index]["prompt"]
   original_tweet = prompt.split("Tweet:")[1].split(", Persona: ")[0]
   persona = prompt.split("Tweet:")[1].split(", Persona:")[1]
   base_output = BASELINE_OUTPUTS[index]["model_output"]
   sft_output = SFT_OUTPUTS[index]["model_output"]
   rl_output = RL_OUTPUTS[index]["model_output"]

   return prompt_template.format(persona, original_tweet, base_output, sft_output, rl_output)

BAD_JSONS = []

def add_info_to_json(index, prompt, chosen_model_idx):
    try:
        data = {}
        data['index'] = index
        data['prompt'] = prompt
        data['chosen_model_index'] = chosen_model_idx

        return data
    except Exception as e:
        print(f"Error: {e}, for idx: {index}")
        print(data)
        BAD_JSONS.append(f"index: {index} " + data)

        return None

# need to set export OPENAI_API_KEY={YOUR KEY HERE} before running
openai.api_key = os.getenv("OPENAI_API_KEY")

def prompt_gpt(gpt_prompt):
  message=[{"role": "user", "content": gpt_prompt}]
  response = openai.ChatCompletion.create(
      model="gpt-4",
      messages = message,
      temperature=0.2,
      max_tokens=1000,
      frequency_penalty=0.0
  )
  json_str = response.choices[0]['message']['content']
  
  return json_str

# takes in a list of user_idxs and the dataframe, formats the prompt, and returns the corresponding
# list of responses from gpt4
def get_response_list():
  responses = []
  start_time = datetime.datetime.now()
  for index in range(len(SFT_OUTPUTS)):
    gpt_prompt = create_prompt(index)
    try:
      gpt_response = prompt_gpt(gpt_prompt)
      post_process_response = add_info_to_json(index, gpt_prompt, gpt_response)
      responses.append(post_process_response)
    except Exception as e:
      print(f"Error processing user index {index}: {e}")
      continue
  end_time = datetime.datetime.now()
  print(f"creating response list took {end_time - start_time}")

  return responses

### Testing out what a single prompt looks like ###
# prompt = create_prompt(0)
# print(prompt)

bulk_responses = get_response_list()
with open(os.path.join(OUTPUT_PATH, f"gpt4_evals.p"), 'wb') as file:
    pickle.dump(bulk_responses, file)
   
# save responses as a pickle just in case something goes wrong
with open(os.path.join(OUTPUT_PATH, f"gpt4_bad_evals.p"), 'wb') as file:
    pickle.dump(BAD_JSONS, file)

# Convert JSON strings to Python dictionaries
json_data = []
for json_str in bulk_responses:
    try:
        block = json.loads(json_str)
        json_data.append(block)
    except:
        print("Error converting string to json block")
        print(json_str)
        continue

# Write the combined data to a JSON file
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
output_filename = os.path.join(OUTPUT_PATH, f"gpt4_evals.json")
with open(output_filename, "w") as outfile:
    json.dump(json_data, outfile, indent=4)
print(f"Output to: {output_filename}")