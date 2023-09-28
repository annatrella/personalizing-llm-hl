#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH="/personalizing-llm-hl/model-checkpoint/ptweets-low"
CRITIC_MODEL_PATH="/personalizing-llm-hl/model-checkpoint/reward-model"
ACTOR_ZERO_STAGE=""
CRITIC_ZERO_STAGE=""
OUTPUT="/personalizing-llm-hl/model-checkpoint/ptweets-rl-low"
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step3_llama2
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

NUM_PADDING_AT_BEGINNING=0 # this is model related
# --enable_hybrid_engine \ was taken out below because of: https://github.com/microsoft/DeepSpeedExamples/issues/385

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path ptweets/ptweets/train_low \
   --data_split 2,4,4 \
   --steps_per_checkpoint 20 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning $NUM_PADDING_AT_BEGINNING \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 2 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_lora_dim 64 \
   --critic_lora_dim 64 \
   --critic_lora_module_name "layers." \
   --actor_lora_module_name "layers." \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log