import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from bachelor.Simulator_E2EDL import Cooking


# %matplotlib inline


def make_autopct(values):
    def my_autopct(percent):
        total = sum(values)
        val = int(round(percent * total / 100.0))
        return '{0:.2f}%  ({1:d})'.format(percent, val)

    return my_autopct


def steering_histogram(hist_labels, title, color, bins):
    plt.figure(figsize=FIGURE_SIZE)
    plt.hist(np.array(hist_labels), bins, density=1, facecolor=color)
    plt.xlabel('Steering Angle')
    plt.ylabel('Normalized Frequency')
    plt.title(title)
    plt.show()


# Data directories
RAW_DATA_DIR = 'Dataset-E2EDL/data_raw'
COOKED_DATA_DIR = 'Dataset-E2EDL/data_cooked'

# The folders to search for data under RAW_DATA_DIR
# For example, the first folder searched will be RAW_DATA_DIR/normal_1
DATA_FOLDERS = ['neighborhood']

# The size of the figures in this notebook
FIGURE_SIZE = (10, 10)

# # Explore .txt files
# sample_txt_path = os.path.join(RAW_DATA_DIR, 'neighborhood/2020-04-13-19-36-47/airsim_rec.txt')
# sample_txt = pd.read_csv(sample_txt_path, sep='\t')
# print(sample_txt.head())
#
# # Explore .png images
# sample_image_path = os.path.join(RAW_DATA_DIR, 'neighborhood/2020-04-16-20-19-34/images'
#                                                '/img_0_0_1587061174141374800.png')
# sample_image = Image.open(sample_image_path)
# plt.title('Sample Image')
# plt.imshow(sample_image)
# plt.show()
#
# # ROI of the image
# sample_image_roi = sample_image.copy()
# fillcolor = (255, 0, 0)
# draw = ImageDraw.Draw(sample_image_roi)
# points = [(0, 76), (0, 135), (255, 135), (255, 76)]
# for i in range(0, len(points), 1):
#     draw.line([points[i], points[(i + 1) % len(points)]], fill=fillcolor, width=3)
# del draw
# plt.title('Image with sample ROI')
# plt.imshow(sample_image_roi)
# plt.show()

# Aggregating non-image data to a single data-frame
path_raw_folders = [os.path.join(RAW_DATA_DIR, f).replace("\\", "/") for f in DATA_FOLDERS]
full_path_raw_folders = []
dataframes = []
for data_folder in path_raw_folders:
    for folder in os.listdir(data_folder):
        current_dataframe = pd.read_csv(os.path.join(os.path.join(data_folder, folder), 'airsim_rec.txt').replace("\\", "/"), sep='\t')
        current_dataframe['Folder'] = os.path.join(data_folder, folder).replace("\\", "/")
        full_path_raw_folders.append(os.path.join(data_folder, folder).replace("\\", "/"))
        dataframes.append(current_dataframe)
dataset = pd.concat(dataframes, axis=0)
print('Number of data points: {0}'.format(dataset.shape[0]))
# print(dataset.head())

# # normal vs swerve driving styles
# min_index = 100
# max_index = 1100
# steering_angles_normal_1 = dataset[dataset['Folder'].apply(lambda v: 'normal_1' in v)]['Steering'][min_index:max_index]
# steering_angles_swerve_1 = dataset[dataset['Folder'].apply(lambda v: 'swerve_1' in v)]['Steering'][min_index:max_index]
# plot_index = [i for i in range(min_index, max_index, 1)]    # List from 100 to 1099
# fig = plt.figure(figsize=FIGURE_SIZE)
# ax1 = fig.add_subplot(111)
# ax1.scatter(plot_index, steering_angles_normal_1, c='b', marker='o', label='normal_1')
# ax1.scatter(plot_index, steering_angles_swerve_1, c='r', marker='o', label='swerve_1')
# plt.legend(loc='upper left')
# plt.title('Steering Angles for normal_1 and swerve_1 runs')
# plt.xlabel('Time')
# plt.ylabel('Steering Angle')
# plt.show()

# # Number of data points in each category
# dataset['Is Swerve'] = dataset.apply(lambda r: 'swerve' in r['Folder'], axis=1)
# grouped = dataset.groupby(by=['Is Swerve']).size().reset_index()
# grouped.columns = ['Is Swerve', 'Count']
# pie_labels = ['Normal', 'Swerve']
# fig, ax = plt.subplots(figsize=FIGURE_SIZE)
# ax.pie(grouped['Count'], labels=pie_labels, autopct=make_autopct(grouped['Count']))
# plt.title('Number of data points per driving strategy')
# plt.show()

# # what the distribution of labels looks like for the two strategies
# bins = np.arange(-1, 1.05, 0.05)
# # normal_labels = dataset[dataset['Is Swerve'] == False]['Steering']
# # swerve_labels = dataset[dataset['Is Swerve']]['Steering']
# steering_labels = dataset['Steering']
# # steering_histogram(normal_labels, 'Normal label distribution', 'g', bins)
# # steering_histogram(swerve_labels, 'Swerve label distribution', 'r', bins)
# steering_histogram(steering_labels, 'Steering label distribution', 'g', bins)

# Processing the data
# train_eval_test_split = [0.7, 0.2, 0.1]
# Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split, chunk_size=32,
#              previous_state=['Steering', 'Throttle', 'Brake', 'Speed'], label=['Steering', 'Throttle', 'Brake'])
