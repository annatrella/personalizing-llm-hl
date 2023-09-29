import argparse
import os
import torch
import json
import sys

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(grandparent_dir)
from global_vars import MAIN_PATH
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer

from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data for inference",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args

def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer

def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch

def load_data(input_path):
    with open(input_path, "r") as file:
        data = json.load(file)

    return data

def get_data(input_path):
    data = load_data(input_path)
    train_data = data["train"]
    dev_data = data["test"]
    prompt_list_train = [item["prompt"] for item in train_data]
    prompt_list_dev = [item["prompt"] for item in dev_data]
    good_ans_list_train = [item["chosen"] for item in train_data]
    good_ans_list_dev = [item["chosen"] for item in dev_data]
    bad_ans_list_train = [item["rejected"] for item in train_data]
    bad_ans_list_dev = [item["rejected"] for item in dev_data]

    return prompt_list_train, good_ans_list_train, bad_ans_list_train, prompt_list_dev, good_ans_list_dev, bad_ans_list_dev

def get_rewards(rm_model, tokenizer, device, prompt_list, good_ans_list, bad_ans_list):
    pred_labels = []
    chosen_rewards = []
    rejected_rewards = []
    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        chosen_reward = outputs["chosen_mean_scores"].item()
        rejected_reward = outputs["rejected_mean_scores"].item()
        predicted_label = chosen_reward > rejected_reward
        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)
        pred_labels.append(predicted_label)

    return pred_labels, chosen_rewards, rejected_rewards

def run_pair_comparison():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()
    prompt_list_train, good_ans_list_train, bad_ans_list_train, prompt_list_dev, good_ans_list_dev, bad_ans_list_dev = get_data(args.data_path)

    pred_labels_train, chosen_rewards_train, rejected_rewards_train = get_rewards(rm_model, tokenizer, device, prompt_list_train, good_ans_list_train, bad_ans_list_train)
    pred_labels_dev, chosen_rewards_dev, rejected_rewards_dev = get_rewards(rm_model, tokenizer, device, prompt_list_dev, good_ans_list_dev, bad_ans_list_dev)

    return pred_labels_train, chosen_rewards_train, rejected_rewards_train, pred_labels_dev, chosen_rewards_dev, rejected_rewards_dev


def compute_f1_score(predicted_labels, true_labels):
    f1 = f1_score(true_labels, predicted_labels)
    return f1

def compute_mean_var_reward(chosen_rewards, rejected_rewards, dataset_type):
    chosen_rewards = np.array(chosen_rewards)
    rejected_rewards = np.array(rejected_rewards)
    abs_diff = np.abs(chosen_rewards - rejected_rewards)

    print("Avg. Reward Absolute Difference \n")
    print(f"{dataset_type}: {np.mean(abs_diff):.3f} ({np.var(abs_diff):.3f})")

if __name__ == "__main__":
    pred_labels_train, chosen_rewards_train, rejected_rewards_train, pred_labels_dev, chosen_rewards_dev, rejected_rewards_dev = run_pair_comparison()
    ### SAVING REWARDS ###
    train_obj = {
        "chosen_rewards": chosen_rewards_train,
        "rejected_rewards": rejected_rewards_train,
        "labels": pred_labels_train
    }
    with open(MAIN_PATH + "/model_outputs/rm-eval/rewards_train.json", 'w') as json_file:
        json.dump(train_obj, json_file, indent=4)
    val_obj = {
        "chosen_rewards": chosen_rewards_dev,
        "rejected_rewards": rejected_rewards_dev,
        "labels": pred_labels_dev
    }
    with open(MAIN_PATH + "/model_outputs/rm-eval/rewards_val.json", 'w') as json_file:
        json.dump(val_obj, json_file, indent=4)

    ### COMPUTING STATS ###
    true_labels_train = [1] * len(pred_labels_train)
    true_labels_dev = [1] * len(pred_labels_dev)
    print("Number of Datapoints (Train):", len(pred_labels_train))
    print("Number of Datapoints (Validation):", len(pred_labels_dev))

    # F1 Scores
    f1_score_train = compute_f1_score(pred_labels_train, true_labels_train)
    f1_score_dev = compute_f1_score(pred_labels_dev, true_labels_dev)
    print("F1 Score (Train):", f1_score_train)
    print("F1 Score (Validation):", f1_score_dev)

    # Average and Var Reward
    compute_mean_var_reward(chosen_rewards_train, rejected_rewards_train, "Train")
    compute_mean_var_reward(chosen_rewards_dev, rejected_rewards_dev, "Val")