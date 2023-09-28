import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import json
import argparse
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import DATA_PATH, MAIN_PATH

INTEREST_TYPES = ["tech enthusiast", "foodie", "travel explorer", "fashionista", "gamer", \
                  "bookworm", "fitness guru", "film buff", "celebrity gossipmonger", "comedian", \
                  "political commentator", "parenting blogger", "health & wellness influencer", "sports fanatic", "art lover", \
                  "science enthusiast", "music aficionado", "humanitarian", "history buff", "professor"
                  ]
INTEREST_IDXS = {value: index for index, value in enumerate(INTEREST_TYPES)}
DEVICE = torch.device("cuda")
OUTPUT = MAIN_PATH + "/model-checkpoint/ptweets-high"

# Define hyperparameters
batch_size = 16
max_seq_length = 256
num_epochs = 2
learning_rate = 2e-5

# setting random seed for reproducibility
RANDOM_SEED = 17
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()

    assert args.output_dir is not None

    return args

def main():
    args = parse_args()
    with open(DATA_PATH + '/ptweets_high_train.json', "r") as file:
        train_data = json.load(file)

    with open(DATA_PATH + '/ptweets_high_val.json', "r") as file:
        val_data =  json.load(file)

    # Define your training and validation data
    train_texts = [obj["input"] for obj in train_data]  # List of training texts
    train_labels = [obj["label"] for obj in train_data]  # List of training labels
    val_texts = [obj["input"] for obj in val_data]    # List of validation texts
    val_labels = [obj["label"] for obj in val_data]    # List of validation labels

    # Define your dataset class
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
            label = int(self.labels[idx])
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
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Initialize the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(INTEREST_IDXS))

    # Create data loaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}', dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_predictions, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation', dynamic_ncols=True):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu()

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f'Epoch {epoch + 1} - Average Loss: {average_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    print("Saving model.")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

# Evaluate the model on the test set if available
# test_predictions, test_labels = [], []
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc='Test', dynamic_ncols=True):
#         # Similar to validation loop

# Calculate classification report or other metrics on the validation/test set
# classification_rep = classification_report(test_labels, test_predictions, target_names=label_names)
# print(classification_rep)
