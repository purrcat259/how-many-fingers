import os
import cv2

import numpy as np

NUMBER_OF_STATES = 6


def load_data():
    file_paths = []
    images = []
    labels = []
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for i in range(0, NUMBER_OF_STATES):
        print('Loading in data for state: {}'.format(i))
        dir = 'data/{}'.format(i)
        for file in os.listdir(dir):
            file_path = os.path.join(current_dir, dir, file)
            file_paths.append(file_path)
            labels.append(i)
    print('Resizing images')
    for file_path in file_paths:
        # print(file_path)
        img = cv2.imread(file_path)
        changed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaled_image = cv2.resize(changed_image, (28, 28))
        flat_image = scaled_image.flatten()
        images.append(np.array(flat_image))
    converted_labels = []
    for label in labels:
        converted_labels.append(convert_to_label_array(label))
    return images, converted_labels


def batch_data(data, batch_index=0, batch_size=10):
    low = min(batch_index * batch_size, len(data[0]) - batch_size)
    high = min([len(data[0]), batch_index * batch_size + batch_size])
    return data[0][low:high], data[1][low:high]


def convert_to_label_array(label):
    arr = [0, 0, 0, 0, 0, 0]
    arr[label] = 1
    return arr

if __name__ == '__main__':
    print(load_data())
