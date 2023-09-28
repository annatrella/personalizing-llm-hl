OUTPUT="/personalizing-llm-hl/model-checkpoint/ptweets-baseline"
mkdir -p $OUTPUT
ZERO_STAGE=3

deepspeed main.py \
   --sft_only_data_path ptweets/train_baseline \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-350m \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
