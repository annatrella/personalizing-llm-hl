MAIN_PATH=/personalizing-llm-hl

python  compute_rewards.py \
    --reward_model_name_or_path ${MAIN_PATH}/model-checkpoint/reward-model \
    --prompts_path ${MAIN_PATH}/model_outputs/ptweets-low/model_outputs.json \
    --inference_output_path ${MAIN_PATH}/model_outputs/ptweets-baseline/model_outputs.json

python  compute_rewards.py \
    --reward_model_name_or_path ${MAIN_PATH}/model-checkpoint/reward-model \
    --prompts_path ${MAIN_PATH}/model_outputs/ptweets-low/model_outputs.json \
    --inference_output_path ${MAIN_PATH}/model_outputs/ptweets-low/model_outputs.json

python  compute_rewards.py \
    --reward_model_name_or_path ${MAIN_PATH}/model-checkpoint/reward-model \
    --prompts_path ${MAIN_PATH}/model_outputs/ptweets-low/model_outputs.json \
    --inference_output_path ${MAIN_PATH}/model_outputs/rl-ptweets-rl-low-final/model_outputs.json