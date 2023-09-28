"""
Given 20 pre-defined twitter user groups, we prompt GPT to annotate the LaMP 7 data set
in order to create a new dataset
"""
import os
import openai
import json
import ast
import pandas as pd
import datetime
import pickle
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH, INTERESTS, INTEREST_TYPES

# path where we will be reading and writing to
prompt_form_1 = "Given a series of existing tweets, Your task rephrase a series of existing tweets to align with the specified Twitter user’s speaking style and persona, without changing the underlying facts presented in the original tweets."
prompt_form_2 = " no prose, only return a valid JSON object with the following format {{\"\"generated_tweets\"\": [\"\", \"\", ... \"\"]}}. JSON, keys and values require double-quotes "
prompt_form_3 = "Persona: {}, Speaking Style: {}, Tweets: {}"

prompt_template = prompt_form_1 + prompt_form_2 + prompt_form_3

def create_prompt(personal_interest, tweets):
   description = INTERESTS[personal_interest]["description"]
   speaking_style = INTERESTS[personal_interest]["speaking_style"]

   return prompt_template.format(description, speaking_style, tweets)

train_df = pd.read_csv(DATA_PATH + '/LaMP7_train.csv', encoding='utf8')

BAD_JSONS = []

def add_info_to_json(json_str, index, personal_interest, input, context):
    try:
        data = json.loads(json_str)
        data['index'] = index
        data['personal_interest'] = personal_interest
        data['input'] = input
        data['output'] = data['generated_tweets'][0]
        data['generated_tweets'] = data['generated_tweets'][1:]
        data['original_tweets'] = context
        # Convert the dictionary back to a JSON string
        return json.dumps(data)
    except Exception as e:
        print(f"Error: {e}, for user_idx: {index}")
        print(json_str)
        BAD_JSONS.append(f"index: {index} " + json_str)
        return None

# need to set export OPENAI_API_KEY={YOUR KEY HERE} before running
openai.api_key = os.getenv("OPENAI_API_KEY")

def prompt_gpt(gpt_prompt):
  message=[{"role": "user", "content": gpt_prompt}]
  response = openai.ChatCompletion.create(
    #   model="gpt-3.5-turbo",
      model="gpt-4",
      messages = message,
      temperature=0.2,
      max_tokens=1000,
      frequency_penalty=0.0
  )
  json_str = response.choices[0]['message']['content']
  
  return json_str

# takes in a list of user_idxs and the dataframe, formats the prompt, and returns the corresponding
# list of responses from gpt3
def get_response_list(df, user_idxs):
  responses = []
  start_time = datetime.datetime.now()
  for i, user_idx in enumerate(user_idxs):
    if i % 50 == 0: 
      print(f"{i} / {len(user_idxs)} have been processed.")
    # randomly select personal interest
    personal_interest = random.choice(INTEREST_TYPES)
    input = df['input'][user_idx].split("Paraphrase the following tweet without any explanation before or after it: ")[1]
    # must do json.dumps then load to correct single quotes to double quotes
    context = json.loads(json.dumps(ast.literal_eval(df['raw profile'][user_idx])[:10]))
    tweets = [input] + context
    gpt_prompt = create_prompt(personal_interest, tweets)
    try:
      gpt_response = prompt_gpt(gpt_prompt)
      post_process_response = add_info_to_json(gpt_response, user_idx, personal_interest, input, context)
      responses.append(post_process_response)
    except Exception as e:
      print(f"Error processing user index {user_idx}: {e}")
      continue
  end_time = datetime.datetime.now()
  print(f"creating response list took {end_time - start_time}")

  return responses

RANDOM_SEED = 123 # seed for reproducibility
TRAIN_DF = pd.read_csv(DATA_PATH + '/LaMP7_train.csv', encoding='utf8')
DEV_DF = pd.read_csv(DATA_PATH + '/LaMP7_dev.csv', encoding='utf8')

### Testing out what a single prompt looks like ###
# test_personal_interest = random.choice(INTEREST_TYPES)
# test_input = TRAIN_DF['input'][200]
# test_context = ast.literal_eval(TRAIN_DF['raw profile'][200])[:10]
# prompt = create_prompt(test_personal_interest, [test_input] + test_context)
# print(prompt)

for j in range(0, len(TRAIN_DF), 2000):
   START_IDX = j
   END_IDX = j + 2000
   random.seed(RANDOM_SEED)
   bulk_responses = get_response_list(TRAIN_DF, range(START_IDX, END_IDX))
   # save responses as a pickle just in case something goes wrong
   with open(os.path.join(DATA_PATH, f"gpt4_tweet_data_{START_IDX}_{END_IDX}.p"), 'wb') as file:
      pickle.dump(bulk_responses, file)
   with open(os.path.join(DATA_PATH, f"gpt4_bad_data_{START_IDX}_{END_IDX}.p"), 'wb') as file:
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
   if not os.path.exists(DATA_PATH):
      os.makedirs(DATA_PATH)
   output_filename = os.path.join(DATA_PATH, f"gpt4_tweet_data_{START_IDX}_{END_IDX}.json")
   with open(output_filename, "w") as outfile:
      json.dump(json_data, outfile, indent=4)
   print(f"Output to: {output_filename}")