OUTPUT="/home/ec2-user/personalizing-llm-hl/model-checkpoint/ptweets-high"
mkdir -p $OUTPUT

python3 main.py \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
