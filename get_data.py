import os
import skimage.data
import skimage.transform

import numpy as np

NUMBER_OF_STATES = 6


def load_data():
    images = []
    labels = []
    for i in range(0, NUMBER_OF_STATES):
        print('Loading in data for state: {}'.format(i))
        dir = 'data/{}'.format(i)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            images.append(skimage.data.imread(file_path))
            labels.append(i)
    return images, labels


def load_resized_data(side_size=32):
    data, labels = load_data()
    resized_data = [skimage.transform.resize(image[:, :, 0], (side_size, side_size), mode='reflect') for image in data]
    # resized_data = [skimage.transform.resize(image, (side_size, side_size), mode='reflect') for image in data]
    flattened_data = [np.array(arr).flatten() for arr in resized_data]
    print(flattened_data[0])
    return flattened_data, labels
