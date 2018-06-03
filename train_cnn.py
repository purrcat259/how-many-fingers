import tensorflow as tf
import cv2

from get_data import load_data, batch_data, convert_to_label_array

learning_rate = 0.001
num_steps = 500

image_side_size = 28
num_input = image_side_size * image_side_size
num_classes = 6
dropout = 0.0
display_step = 10
batch_size = 10

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def conv_2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool_2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv_2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool_2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv_2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool_2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
data = load_data()
max_batch_index = len(data[0]) // batch_size
batch_index = 0

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = batch_data(data, batch_index=batch_index)
        batch_index += 1
        if batch_index > max_batch_index:
            batch_index = 0
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    testing_accuracy = sess.run(accuracy, feed_dict={X: data[0], Y: data[1], keep_prob: 1.0})
    print("Testing Accuracy:", testing_accuracy)

    print('Running Camera')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_counter = 0
    last_prediction = 0

    while True:
        frame_counter += 1
        ret_val, img = cam.read()
        # Predict every 10th frame
        if frame_counter % 10 == 0:
            changed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled_image = cv2.resize(changed_image, (28, 28))
            flat_image = scaled_image.flatten()
            # for i in range(0, 6):
            label_arr = convert_to_label_array(0)
            predicted_value = prediction.eval(feed_dict={X: [flat_image], Y: [label_arr], keep_prob: 0.5})
            actual_number = list(predicted_value[0]).index(1.0)
            last_prediction = actual_number
        cv2.putText(img, 'Predicted: {}'.format(last_prediction), (10, 300), font, 1, (255, 255, 255))
        cv2.imshow('Recognition Test', img)
        if cv2.waitKey(1) == 27:
            break
