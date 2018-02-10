from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os

files = ['classifier_accuracies.npy', 'features_accuracies.npy']


def transform(data, smooth):
    return np.mean(data.reshape(-1, smooth), axis=1)

def render_plot(data_dict):
    for file,data in data_dict.items():
        plt.plot(range(len(data)),data,label=file)
    plt.legend()
    plt.show()


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('files', nargs='+')
#     parser.add_argument('--smooth', type=int, default=100)
#
#     args = parser.parse_args()
#
data_dict = {}
for file in files:
    data = np.load(file)
    smoothed = transform(data, 1)
    data_dict[file] = smoothed

render_plot(data_dict)