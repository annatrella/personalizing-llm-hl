#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT="/personalizing-llm-hl/model-checkpoint/reward-model"
ZERO_STAGE=0
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path ptweets/rm-baseline \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
