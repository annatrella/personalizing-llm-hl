from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import json
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH, INTERESTS, INTEREST_TYPES

TASK = "Paraphrase the following tweet in the user's persona and speaking style, without changing the underlying facts presented in the original tweet. "

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
MODEL_PATH = "/model-checkpoint/ptweets-high"
MAX_SEQUENCE_LENGTH = 256
loaded_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
loaded_model = loaded_model.to(DEVICE)
loaded_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
with open(DATA_PATH + '/ptweets_high_test.json', "r") as file:
    high_test_data = json.load(file)
test_texts = [obj["input"] for obj in high_test_data]

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def process_low(obj, pred):
    personal_interest = INTEREST_TYPES[pred]
    description = INTERESTS[personal_interest]["description"]
    speaking_style = INTERESTS[personal_interest]["speaking_style"]
    tweet = obj['prompt'].split("Tweet: ")[1].split(', Persona:')[0]
    json_object = {
        "prompt": TASK + f"Tweet: {tweet}, Persona: {description}, Speaking Style: {speaking_style}",
        "output": obj['output']
    }

    return json_object

def iterate_and_process_json(json, preds):
    assert len(json) == len(preds)
    low_objects = []
     
    for i, obj in enumerate(json):
        low_objects.append(process_low(obj, int(preds[i])))

    return low_objects

def run_classify(test_texts, loaded_model, loaded_tokenizer):
    test_dataset = TextClassificationDataset(test_texts, [], loaded_tokenizer, MAX_SEQUENCE_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    loaded_model.eval()
    test_predictions = []
    N = len(test_loader)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 100 == 0: 
                print(f"{i} / {N} have been processed.")
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = loaded_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Assuming you want to get class predictions (class with the highest probability)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            test_predictions.extend(batch_predictions)

    return test_predictions

test_predictions = run_classify(test_texts, loaded_model, loaded_tokenizer)

### SANITY CHECK ###
def f1_score_func(labels, preds):
    return f1_score(labels, preds, average='weighted')

val_accuracy = accuracy_score([obj['label'] for obj in high_test_data], test_predictions)
f1_score = f1_score_func([obj['label'] for obj in high_test_data], test_predictions)
print(f"Test Accuracy: {val_accuracy}, Test F1 Score: {f1_score}")

with open(DATA_PATH + '/ptweets_low_test.json', "r") as file:
    low_test_data = json.load(file)

test_result = iterate_and_process_json(low_test_data, test_predictions)

# create json file for low level model
with open(DATA_PATH + '/ptweets_user_profiles_test.json', 'w') as json_file:
    json.dump(test_result, json_file, indent=4)