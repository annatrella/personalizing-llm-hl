import numpy as np
import json
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
from global_vars import DATA_PATH

with open(DATA_PATH + '/ptweets_permutations_train.json', "r") as file:
    TRAIN_DATA = json.load(file)

with open(DATA_PATH + '/ptweets_permutations_val.json', "r") as file:
    VAL_DATA = json.load(file)

with open(DATA_PATH + '/ptweets_permutations_test.json', "r") as file:
    TEST_DATA =  json.load(file)

def combine_data(data_list):
    result = []
    for data in data_list:
        for obj in data:
            result.append(obj)

    return result

def compute_class_distribution(data):
    frequency_dict = {}
    for obj in data:
        personal_interest = obj["personal_interest"]
        if personal_interest in frequency_dict:
            frequency_dict[personal_interest] += 1
        else:
            frequency_dict[personal_interest] = 1

    # get normzlied frequency
    total_sum = 0
    N = len(data)
    for string, count in frequency_dict.items():
        frequency_dict[string] = count / N
        total_sum += count / N

    return frequency_dict

def generate_pastel_rainbow_colors(num_colors):
    cmap = plt.get_cmap('rainbow')
    pastel_colors = []
    for i in range(num_colors):
        t = i / (num_colors - 1)  # Interpolation parameter
        color = cmap(t)
        pastel_color = tuple((np.array(color[:3]) + 1) / 2)  # Convert to pastel by scaling
        pastel_colors.append(pastel_color)

    return pastel_colors

def plot_horizontal_bar_chart(data_dict, output_pdf):
    plt.rcParams['ytick.labelsize'] = 14  # adjust fontsize of y labels
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    bar_colors = generate_pastel_rainbow_colors(len(labels))
    
    # Create a horizontal bar chart
    plt.figure(figsize=(6, 8))
    bars = plt.barh(labels, values, color=bar_colors)
    for bar, value in zip(bars, values):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center', ha='left', fontsize=12)
    # plt.ylabel('Personal Interests', fontsize=14)   
    plt.xlabel('Percentage of Users', fontsize=16) 
    plt.xlim(0, 0.073)
    # plt.title('')
    
    # Save the horizontal bar chart as a PDF
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close() 

combined_data = combine_data([TRAIN_DATA, VAL_DATA, TEST_DATA])
combined_dist = compute_class_distribution(combined_data)
print("combined: ", combined_dist)

plot_horizontal_bar_chart(combined_dist, 'data_prop.pdf')

# train_dist = compute_class_distribution(TRAIN_DATA)
# val_dist = compute_class_distribution(VAL_DATA)
# test_dist = compute_class_distribution(TEST_DATA)

# print("train: ", train_dist)
# print("val: ", val_dist) 
# print("test: ", test_dist)      