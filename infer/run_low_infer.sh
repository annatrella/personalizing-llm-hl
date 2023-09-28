#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

MAIN_PATH=/personalizing-llm-hl
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path ${MAIN_PATH}/model-checkpoint/ptweets-low \
    --data_path ${MAIN_PATH}/data/ptweets_user_profiles_test.json \
    --output_path ${MAIN_PATH}/model_outputs/ptweets-low \
    --eos_token "<|endoftext|>"