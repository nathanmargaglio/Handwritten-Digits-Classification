import numpy as np
import logging
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import time
from datetime import timedelta
import builtins

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1./(1. + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    res = np.all(train_data == train_data[0, :], axis=0)
    removable_indices = np.where(res)
    selected_features = np.where(~res)
    try:
        builtins.selected_features = selected_features
    except Exception as e:
        pass

    train_data = np.delete(train_data, removable_indices, 1)
    validation_data = np.delete(validation_data, removable_indices, 1)
    test_data = np.delete(test_data, removable_indices, 1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    try:
        logger = builtins.logger
        logging_data = builtins.logging_data
        start_time = logging_data['start_time']
        current_time = logging_data['current_time']
        time_delta = time() - current_time
        logging_data['current_time'] = time()
        total_time = time() - start_time

        logger.info(' - Iteration:  ' + str(logging_data['iter']))
        logger.info(' - Time Delta: ' + str(timedelta(seconds=time_delta)))
        logger.info(' - Total Time: ' + str(timedelta(seconds=total_time)))
        logging_data['iter'] += 1
    except Exception as e:
        pass

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
    reg = (lambdaval / (2 * len(training_data))) * (np.sum(w1**2) + np.sum(w2**2))

    # Error
    E = (y * np.log(o2) + (1 - y) * np.log(1 - o2))
    obj_val = -(np.sum(E) / len(training_data)) + reg

    # logging.debug("E: " + str(E))

    # logging.debug("obj_val: " + str(obj_val))

    try:
        logging_data['plt_data'].append(obj_val)
    except Exception as e:
        pass

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
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

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



# to aid with importing
if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    """**************Neural Network Script Starts here********************************"""

    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    #  Train Neural Network

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 50

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}
    plt_data = []
    iteration = 0
    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    """
    # Batches
    opts = {'maxiter': 1}
    iter_weights = initialWeights
    plt_data = []
    for i in range(500):
        if i % 50 == 0: logger.info("Iter: {}".format(i))
        random_sample_index = np.random.choice(np.arange(len(train_data)), 1000)
        args = (n_input, n_hidden, n_class, train_data[random_sample_index], train_label[random_sample_index], lambdaval)
        nn_params = minimize(nnObjFunction, iter_weights, jac=True, args=args, method='CG', options=opts)
        iter_weights = nn_params.x
    """

    # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
    # and nnObjGradient. Check documentation for this function before you proceed.
    # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data[:10])

    # find the accuracy on Training Dataset

    print()
    logger.info('Training set Accuracy: ' + str(100 * np.mean((predicted_label == train_label[:10]).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    # find the accuracy on Validation Dataset
    print()
    logger.info('Validation set Accuracy: ' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    # find the accuracy on Validation Dataset
    print()
    logger.info('Test set Accuracy: ' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    print()
    print('Done!')
