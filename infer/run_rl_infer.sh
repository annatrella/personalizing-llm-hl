#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

MAIN_PATH=/personalizing-llm-hl
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path ${MAIN_PATH}/model-checkpoint/ptweets-rl-low/actor-200 \
    --data_path ${MAIN_PATH}/data/ptweets_low_test.json \
    --output_path ${MAIN_PATH}/model_outputs/rl-ptweets-rl-low-final \
    --eos_token "<|endoftext|></s>"