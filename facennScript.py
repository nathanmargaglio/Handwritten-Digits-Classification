'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import logging
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import builtins
import logging
from time import time
import pickle
import os
from datetime import timedelta

def initializeWeights(n_in,n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W


def sigmoid(z):
    return 1./(1. + np.exp(-z))


def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    # Set Bias

    b1 = np.ones((len(training_data), 1))
    b2 = np.ones((len(training_data), 1))

    # logging.debug("w1: " + str(w1[0][:10]))

    # Forward Propagation
    X = np.append(training_data, b1, 1)  # append bias
    net1 = X.dot(w1.T)
    o1 = sigmoid(net1)

    # logging.debug("o1 sum: " + str(o1[0]))

    H = np.append(o1, b2, 1)
    net2 = H.dot(w2.T)
    o2 = sigmoid(net2)

    # logging.debug("H: " + str(H[0]))

    # 1-hot encoding
    y = np.zeros(o2.shape)
    y[np.arange(o2.shape[0]), training_label.astype(int)] = 1

    # Regularization
    reg = (lambdaval / (2 * len(training_data))) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

    # Error
    E = (y * np.log(o2) + (1 - y) * np.log(1 - o2))
    obj_val = -(np.sum(E) / len(training_data)) + reg

    # logging.debug("E: " + str(E))

    # logging.debug("obj_val: " + str(obj_val))

    # Gradients
    grad_w2 = (1. / len(training_data)) * (np.dot((o2 - y).T, H) + lambdaval * w2)
    # print("grad_w2: " + str(grad_w2[0]))

    sm = (o2 - y).dot(w2[:, :-1]).T  # note: we remove the bias from w2
    tm = ((1 - o1) * o1).T
    grad_w1 = (1. / len(training_data)) * ((sm * tm).dot(X) + lambdaval * w1)

    # print("grad_w1: " + str(grad_w1[0]))
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):

    labels = np.array([])
    bias = np.ones((len(data), 1))

    # Forward Propagation
    X = np.append(data, bias, 1)  # append bias
    net1 = X.dot(w1.T)
    o1 = sigmoid(net1)

    H = np.append(o1, bias, 1)
    net2 = H.dot(w2.T)
    o2 = sigmoid(net2)

    labels = np.array(np.argmax(o2, axis=1))

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# check if log dir exists, if not, make it
log_dir = 'logging/'
pickle_dir = log_dir + '/pickle_data/'

if not os.path.exists(pickle_dir):
    os.makedirs(pickle_dir)

log_count = len([lf for lf in os.listdir(log_dir) if 'face_log_' in lf])

fh = logging.FileHandler(log_dir + 'face_log_{}.txt'.format(log_count))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

builtins.logger = logger

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 30
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter': 50}    # Preferred value.

total_processing_time = time()
total_processing_time_current = total_processing_time
total_processing_count = 0

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

training_predicted_label = nnPredict(w1, w2, train_data)
training_accuracy = np.mean((training_predicted_label ==
                             train_label).astype(float))
logger.info('Training Set Accuracy:   ' +
            str(100 * training_accuracy) + '%')

validation_predicted_label = nnPredict(w1, w2, validation_data)
validation_accuracy = np.mean((validation_predicted_label ==
                               validation_label).astype(float))
logger.info('Validation Set Accuracy: ' +
            str(100 * validation_accuracy) + '%')

test_predicted_label = nnPredict(w1, w2, test_data)
test_accuracy = np.mean((test_predicted_label ==
                         test_label).astype(float))
logger.info('Test Set Accuracy:       ' +
            str(100 * test_accuracy) + '%')

total_processing_time_iteration = total_processing_time_current
total_processing_time_current = time()
total_processing_time_iteration_delta =\
    total_processing_time_current - total_processing_time_iteration
total_processing_time_delta =\
    total_processing_time_current - total_processing_time

logger.info("Iteration Processing Time: {}".format(str(
    timedelta(seconds=total_processing_time_iteration_delta))))
logger.info("Total Processing Time:     {}".format(str(
    timedelta(seconds=total_processing_time_delta))))

params_pickle = {
    'n_hidden': n_hidden,
    'lambdaval': lambdaval,
    'w1': w1,
    'w2': w2
}

logging_pickle = {
    'n_hidden': n_hidden,
    'lambdaval': lambdaval,
    'time': total_processing_time_iteration_delta,
    'train_acc': training_accuracy,
    'val_acc': validation_accuracy,
    'test_accuracy': test_accuracy
}

with open(pickle_dir + 'face_params.pickle', 'wb') as f:
    pickle.dump(params_pickle, f)

with open(pickle_dir + 'face_log.pickle', 'wb') as f:
    pickle.dump(logging_pickle, f)