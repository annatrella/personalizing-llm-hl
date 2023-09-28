# inference for model selection (validation) of RL fine-tuning of low-level model

start=20
end=200
STEP_SIZE=20
MAIN_PATH=/personalizing-llm-hl
BASE_MODEL_PATH=${MAIN_PATH}/model-checkpoint/ptweets-rl-low/actor
BASE_OUTPUT_PATH=${MAIN_PATH}/model_outputs/ptweets-rl-low

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
for ((i=start; i<=end; i+=${STEP_SIZE})); do
    python prompt_eval.py \
        --model_name_or_path ${BASE_MODEL_PATH}-${i} \
        --data_path ${MAIN_PATH}/data/ptweets_low_val.json \
        --output_path  ${BASE_OUTPUT_PATH}-${i}\
        --eos_token "<|endoftext|></s>"
done

LAST_STEP=213
python prompt_eval.py \
        --model_name_or_path ${BASE_MODEL_PATH}-${LAST_STEP} \
        --data_path ${MAIN_PATH}/data/ptweets_low_val.json \
        --output_path  ${BASE_OUTPUT_PATH}-${LAST_STEP}\
        --eos_token "<|endoftext|></s>"