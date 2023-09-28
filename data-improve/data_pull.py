import pandas as pd
import json
from io import StringIO
from baseline_user_profile import analyze_style
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH

"""## Create Dataframe
---
"""

def get_text_from_json_column(column_val):
  return [json_obj["text"] for json_obj in column_val]

def create_csv(in_path, out_path, title):
  train_in_df = pd.read_json(in_path)
  train_out_df = pd.read_json(out_path)
  outputs = json.dumps(list(train_out_df['golds']))
  processed_train_out_df = pd.read_json(StringIO(outputs))
  result = train_in_df.merge(processed_train_out_df)
  result['raw profile'] = result['profile'].apply(lambda x: get_text_from_json_column(x))
  # comment this out if we don't want to create the style profile
  result['profile'] = result['raw profile'].apply(lambda x: analyze_style(x))
  result.to_csv(title)

in_path = 'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/train/train_questions.json'
out_path = 'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/train/train_outputs.json'
create_csv(in_path, out_path, DATA_PATH + '/LaMP7_train.csv')

in_path = 'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/dev/dev_questions.json'
out_path = 'https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/dev/dev_outputs.json'
create_csv(in_path, out_path, DATA_PATH + '/LaMP7_dev.csv')

