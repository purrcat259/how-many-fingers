import os
import skimage.data
import skimage.transform
import skimage.feature
import numpy as np
import tensorflow as tf
import time
import cv2

from skimage import img_as_ubyte
from tqdm import tqdm

from get_data import load_data

image_side_size = 128
start_time = int(time.time())
NUMBER_OF_STATES = 6
ESCAPE_KEY = 27
SPACEBAR_KEY = 32


edge_low_threshold = 400
edge_high_threshold = 500


def main():
    images, labels = load_data()
    # Resize the images

    print('Resizing {} images'.format(len(images)))
    resized_images = [skimage.transform.resize(image, (image_side_size, image_side_size)) for image in images]
    print('Edging images')
    resized_images = [
        skimage.feature.canny(image[:, :, 0], low_threshold=edge_low_threshold, high_threshold=edge_high_threshold) for
        image in resized_images]

    labels_arr = np.array(labels)
    images_arr = np.array(resized_images)

    print('Starting up Neural Network')
    graph = tf.Graph()

    with graph.as_default():
        # Placeholders for inputs and labels
        images_ph = tf.placeholder(tf.float32, [None, image_side_size, image_side_size])
        # images_ph = tf.placeholder(tf.float32, [None, image_side_size, image_side_size, 3])
        labels_ph = tf.placeholder(tf.int32, [None])

        # Flatten the input
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer
        logits = tf.contrib.layers.fully_connected(images_flat, NUMBER_OF_STATES, tf.nn.relu)

        # Convert logits to label indices
        predicted_labels = tf.argmax(logits, 1)

        # Define loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

        # Create training operation
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        init = tf.global_variables_initializer()

    session = tf.Session(graph=graph)

    _ = session.run([init])

    training_cycles = 100
    for i in tqdm(range(training_cycles + 1)):
        _, loss_Value = session.run([train, loss], feed_dict={images_ph: images_arr, labels_ph: labels_arr})

    print('Done in {} seconds'.format(int(time.time()) - start_time))

    # print('Running testing')
    # predicted = session.run([predicted_labels], feed_dict={images_ph: resized_images})[0]
    # match_count = sum([int(y == y_) for y, y_ in zip(labels, predicted)])
    # accuracy = match_count / len(labels)
    # print(accuracy)

    taken_image = None
    print('Running Webcam')
    # Start up the webcam
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        cv2.imshow('Testing window', img)
        if cv2.waitKey(1) == SPACEBAR_KEY:
            print('Image taken')
            taken_image = img
            # Resize the image returned
            resized_image = skimage.transform.resize(taken_image, (image_side_size, image_side_size))
            edged_image = skimage.feature.canny(resized_image[:, :, 0], low_threshold=edge_low_threshold,
                                                high_threshold=edge_high_threshold)
            result = session.run([predicted_labels], feed_dict={images_ph: [edged_image]})
            predicted = result[0]
            print(predicted)
        if cv2.waitKey(1) == ESCAPE_KEY:
            break

    cv2.destroyAllWindows()
    session.close()

if __name__ == '__main__':
    main()
