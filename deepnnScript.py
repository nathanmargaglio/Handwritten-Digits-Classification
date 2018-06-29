'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import pickle
import tensorflow as tf
import numpy as np
import logging
from time import time
import os
from datetime import timedelta

# Create model
# Add more hidden layers to create deeper networks
# Remember to connect the final hidden layer to the out_layer
def create_multilayer_perceptron(n_layers=2):
    # Network Parameters
    # Here, we assume every layer has 'n_hidden' nodes
    n_hidden = 256  # 1st layer number of features
    n_input = 2376  # data input
    n_classes = 2

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    for n in range(2, n_layers+1):
        weights['h{}'.format(n)] = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        biases['b{}'.format(n)] = tf.Variable(tf.random_normal([n_hidden]))

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    # Hidden layer with RELU activation
    prev_layer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    prev_layer = tf.nn.relu(prev_layer)

    for n in range(2, n_layers+1):
        layer = tf.add(tf.matmul(prev_layer, weights['h{}'.format(n)]), biases['b{}'.format(n)])
        layer = tf.nn.relu(layer)
        prev_layer = layer
 
    # Output layer with linear activation
    out_layer = tf.matmul(prev_layer, weights['out']) + biases['out']
    return out_layer, x, y

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# check if log dir exists, if not, make it
log_dir = 'logging/'
pickle_dir = log_dir + '/deep_pickle_data/'

if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

        log_count = len([lf for lf in os.listdir(log_dir) if 'deep_log_' in lf])

        fh = logging.FileHandler(log_dir + 'deep_log_{}.txt'.format(log_count))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

total_processing_time = time()
total_processing_time_current = total_processing_time
total_processing_count = 0

logger.info("Starting Tests...")

for n_layers in [3, 5, 7]:
    # Parameters
    learning_rate = 0.0001
    training_epochs = 100
    batch_size = 100

    # Construct model
    pred,x,y = create_multilayer_perceptron(n_layers)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # load data
    train_features, train_labels, valid_features, valid_labels, test_features, test_labels = preprocess()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            print("Epoch:", epoch)
            avg_cost = 0.
            total_batch = int(train_features.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_features[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        training_accuracy = accuracy.eval({x: train_features, y: train_labels})
        validation_accuracy = accuracy.eval({x: valid_features, y: valid_labels})
        test_accuracy = accuracy.eval({x: test_features, y: test_labels})

        print("Train Accuracy:", training_accuracy)
        print("Valid Accuracy:", validation_accuracy)
        print("Test Accuracy :", test_accuracy)

        total_processing_time_iteration = total_processing_time_current
        total_processing_time_current = time()
        total_processing_time_iteration_delta = total_processing_time_current - total_processing_time_iteration
        total_processing_time_delta = total_processing_time_current - total_processing_time
        
        logging_pickle = {
                'n_layers': n_layers,
                'time': total_processing_time_iteration_delta,
                'train_acc': training_accuracy,
                'val_acc': validation_accuracy,
                'test_accuracy': test_accuracy,
                }
        
        with open(pickle_dir + str(n_layers) + '_deep_log.pickle', 'wb') as f:
            pickle.dump(logging_pickle, f)
