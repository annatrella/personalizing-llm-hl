# model-based evaluation: average rewards according to reward model
import argparse
import os
import torch
import json
import sys
import numpy as np

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(grandparent_dir)

from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval outputs using the finetued reward model")
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        help="Path to prompts",
        required=True,
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        help="Path to inference outputs of language model",
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

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch

def load_data(input_path):
    with open(input_path, "r") as file:
        data = json.load(file)

    return data

def get_prompts(input_path):
    prompts_data = load_data(input_path)
    return [item["prompt"] for item in prompts_data]

def get_model_outputs(input_path):
    data = load_data(input_path)
    return[item["model_output"] for item in data]

def get_rewards(rm_model, tokenizer, device, prompt_list, ans_list, num_padding_at_beginning):
    rewards = []
    for prompt, ans in zip(prompt_list, ans_list):
        batch = prepare_singlesample(prompt,
                                 ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)

        rm_model.eval()
        # Run inference
        with torch.no_grad():
            outputs = rm_model.forward_value(
                **batch, prompt_length=max(2, num_padding_at_beginning)
            )  # we just need to skip the number of padding tokens at the beginning

        reward = outputs["chosen_end_scores"].item()
        rewards.append(reward)

    return rewards

def run_get_rewards():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.reward_model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()
    prompt_list = get_prompts(args.prompts_path)
    ans_list = get_model_outputs(args.inference_output_path)

    rewards = get_rewards(rm_model, tokenizer, device, prompt_list, ans_list, args.num_padding_at_beginning)

    return rewards

def compute_reward_metrics(rewards):
    print(f"Rewards Mean and Var: {np.mean(rewards):.3f} ({np.var(rewards):.3f})")

if __name__ == "__main__":
    rewards = run_get_rewards()
    compute_reward_metrics(rewards)