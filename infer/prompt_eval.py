# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import datetime
import json

from transformers import (
    AutoModelForCausalLM, )

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to data for inference",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save model outputs",
        required=True,
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        help="EOS token",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  temperature=0.9,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result

def process_response(prompt, response, eos_token):
    output = response[0].split(prompt)[-1]
    output = output.replace(eos_token, "")
    return output

def prompt_eval(args, model, tokenizer, device,
                prompts):
    outputs = []
    N = len(prompts)
    print(f"Starting inference for {N} queries")
    start_time = datetime.datetime.now()
    for i, prompt in enumerate(prompts):
        if i % 100 == 0: 
            print(f"{i} / {N} have been processed.")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # ==========finetune: Greedy=========
        response = generate(model,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        output = process_response(prompt, response, args.eos_token)
        outputs.append(output)
    end_time = datetime.datetime.now()
    print(f"Batch inference finished in {end_time - start_time}")

    return outputs
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
 
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)

        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)

        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)

        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)

def load_data(input_path):
    with open(input_path, "r") as file:
        data = json.load(file)
    return data

def save_output_file(data, model_outputs, output_path, seed=None):
    # Add the new field to each object in the list
    new_field_value = 'new_value'
    assert len(data) == len(model_outputs)
    for i, item in enumerate(data):
        item['model_output'] = model_outputs[i]

    # Save the updated JSON data back to a file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_name = f"model_outputs_{seed}.json" if seed else "model_outputs.json"
    file_path = os.path.join(output_path, output_file_name)

    # Write the JSON data to the file in the specified directory
    print(f"Saving output file to: {file_path}")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def main():
    args = parse_args()

    device = torch.device("cuda:0")
    start_time = datetime.datetime.now()
    print("loading tokenizer and model...")
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True)
    model = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path,
                                     tokenizer, None)
    model.to(device)
    end_time = datetime.datetime.now()
    print(f"tokenizer and model are loaded in {end_time - start_time}")

    data = load_data(args.data_path)
    prompts = [item["prompt"] for item in data]
    model_outputs = prompt_eval(args, model, tokenizer, device,
                prompts)
    save_output_file(data, model_outputs, args.output_path)

if __name__ == "__main__":
    main()