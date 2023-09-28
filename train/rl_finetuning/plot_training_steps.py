import matplotlib.pyplot as plt
import re
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_vars import MAIN_PATH

def extract_values_from_log(file_path):
    step_values = []
    average_reward_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'step:' in line:
                step_match = re.search(r'epoch: 0\|step: (\d+)', line)
                if step_match:
                    step_values.append(int(step_match.group(1)))

            if 'average reward score' in line:
                avg_reward_match = re.search(r'average reward score: ([+-]?\d+\.\d+)', line)
                if avg_reward_match:
                    average_reward_values.append(float(avg_reward_match.group(1)))

    return step_values, average_reward_values

def save_plot_as_pdf(step_values, average_reward_values, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(step_values, average_reward_values, marker='o', linestyle='-')
    plt.title('Average Reward During PPO Training')
    plt.xlabel('Step')
    plt.ylabel('Average Reward Score (On Training Set)')
    plt.grid(True)
    plt.savefig(output_file, format='pdf')
    plt.close()

file_path = MAIN_PATH + '/model-checkpoint/ptweets-rl-low/training.log'
step_values, average_reward_values = extract_values_from_log(file_path)

output_file = 'training.pdf'
save_plot_as_pdf(step_values, average_reward_values, output_file)




