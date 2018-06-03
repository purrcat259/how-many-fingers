import os
import skimage.data
import skimage.transform
import tensorflow as tf
import numpy as np

NUMBER_OF_STATES = 6


def load_data():
    file_paths = []
    labels = []
    for i in range(0, NUMBER_OF_STATES):
        print('Loading in data for state: {}'.format(i))
        dir = 'data/{}'.format(i)
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            file_paths.append(file_path)
            labels.append(i)

    image = tf.image.decode_jpeg(content, channels=3)
    image = tf.cast(image, tf.float32)

