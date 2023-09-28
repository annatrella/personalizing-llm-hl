# used for validation and model selection for RL fine-tuning of low-level model

start=20
end=180
STEP_SIZE=20
BASE_MODEL_PATH=/personalizing-llm-hl/model-checkpoint/ptweets-rl-low/actor
BASE_OUTPUT_PATH=/personalizing-llm-hl/model_outputs/ptweets-rl-low

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
for ((i=start; i<=end; i+=${STEP_SIZE})); do
    echo ${i}
    python  compute_rewards.py \
    --reward_model_name_or_path /personalizing-llm-hl/model-checkpoint/reward-model \
    --prompts_path ${BASE_OUTPUT_PATH}-${i}/model_outputs.json \
    --inference_output_path ${BASE_OUTPUT_PATH}-${i}/model_outputs.json
done

LAST_STEP=213
echo ${LAST_STEP}
python  compute_rewards.py \
    --reward_model_name_or_path /personalizing-llm-hl/model-checkpoint/reward-model \
    --prompts_path ${BASE_OUTPUT_PATH}-${i}/model_outputs.json \
    --inference_output_path ${BASE_OUTPUT_PATH}-${LAST_STEP}/model_outputs.json