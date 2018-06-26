import logging
from time import time
import pickle
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import builtins
from nnScript import initializeWeights, preprocess, nnObjFunction, nnPredict

# set up logging

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

log_count = len([lf for lf in os.listdir(log_dir) if 'log_' in lf])

fh = logging.FileHandler(log_dir + 'log_{}.txt'.format(log_count))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

builtins.logger = logger

logger.info("Preprocessing...")

train_data, train_label, validation_data,\
    validation_label, test_data, test_label = preprocess()

n_input = train_data.shape[1]
n_class = 10
n_hiddens = [4, 8, 12, 16, 20, 25, 50]
lambdavals = [0, 10, 20, 30, 40, 50, 60]

total_processing_time = time()
total_processing_time_current = total_processing_time
total_processing_count = 0

logger.info("Starting Tests...")
for n_hidden in n_hiddens:
    for lambdaval in lambdavals:
        logger.info("n_hidden:  " + str(n_hidden))
        logger.info("lambdaval: " + str(lambdaval))
        iter_name = "n_{}-l_{}".format(n_hidden, lambdaval)

        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        initialWeights = np.concatenate((initial_w1.flatten(),
                                         initial_w2.flatten()), 0)

        opts = {'maxiter': 50}
        optimize_time = time()
        builtins.logging_data = {
            'iter': 0,
            'plt_data': [],
            'start_time': optimize_time,
            'current_time': optimize_time
        }
        args = (n_input, n_hidden, n_class,
                train_data, train_label, lambdaval)
        nn_params = minimize(nnObjFunction, initialWeights,
                             jac=True, args=args, method='CG', options=opts)

        # plt.plot(builtins.logging_data['plt_data'])
        # plt.title('Training Error')
        # plt.ylabel('error (obj_val)')
        # plt.xlabel('iteration')
        # plt.ylim(0, 10)
        # plt.savefig(error_dir + iter_name + '.png', bbox_inches='tight')
        # plt.cla()

        w1 = nn_params.x[0:n_hidden * (n_input + 1)]\
            .reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):]\
            .reshape((n_class, (n_hidden + 1)))

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

        logger.info("Total Processing Count: " +
                    str(total_processing_count) + "/" +
                    str(len(n_hiddens) * len(lambdavals)))

        total_processing_count += 1

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
            'selected_features': builtins.selected_features,
            'w1': w1,
            'w2': w2
        }

        logging_pickle = {
            'n_hidden': n_hidden,
            'lambdaval': lambdaval,
            'time': total_processing_time_iteration_delta,
            'train_acc': training_accuracy,
            'val_acc': validation_accuracy,
            'test_accuracy': test_accuracy,
            'iter_error': builtins.logging_data['plt_data']
        }

        with open(pickle_dir + iter_name + '_params.pickle', 'wb') as f:
            pickle.dump(params_pickle, f)

        with open(pickle_dir + iter_name + '_log.pickle', 'wb') as f:
            pickle.dump(logging_pickle, f)
