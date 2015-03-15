import pandas as pd
import numpy as np
import pdb
from scipy import optimize
import datetime

## theano library

import theano.tensor as T
from theano import function
from theano import shared
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

start_time = datetime.datetime.now()

#########################################
## data normalization

def data_norm (raw_data):
    
    return raw_data/(1.0 * 255) - np.transpose (np.atleast_2d (np.mean (raw_data/(1.0 * 255), axis = 1)))

#########################################
## functions from theano library (slightly modified from http://deeplearning.net/tutorial/)

class LogisticRegression (object):

    def __init__(self, input, n_in, n_out, prob_dropout_lr = 0.25):

        self.input = dropout (input, prob_dropout_lr)

        self.W = theano.shared (value = np.zeros ((n_in, n_out), dtype = theano.config.floatX), name = 'W', borrow = True)

        self.b = theano.shared (value = np.zeros((n_out, ), dtype = theano.config.floatX), name = 'b', borrow = True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]

    def negative_log_likelihood (self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError ('y should have the same shape as self.y_pred', ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith ('int'):
            return T.mean (T.neq (self.y_pred, y))
        else:
            raise NotImplementedError ()
            
    def predict (self):
        return self.y_pred

class HiddenLayer (object):
    def __init__ (self, rng, input, n_in, n_out, W = None, b = None, activation = T.tanh, prob_dropout_hl = 0.25):

        self.input = dropout(input, prob_dropout_hl)

        if W is None:
            W_values = np.asarray (rng.uniform (low = - np.sqrt(6. / (n_in + n_out)), high = np.sqrt(6. / (n_in + n_out)), size = (n_in, n_out)), dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared (value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = np.zeros ((n_out,), dtype = theano.config.floatX)
            b = theano.shared (value = b_values, name = 'b', borrow = True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        
        self.params = [self.W, self.b]


class LeNetConvPoolLayer (object):

    def __init__ (self, rng, input, filter_shape, image_shape, poolsize, prob_dropout_cnn = 0.25):

        self.input = dropout(input, prob_dropout_cnn)

        fan_in = np.prod (filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))
        
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared (np.asarray (rng.uniform (low = -W_bound, high = W_bound, size = filter_shape), dtype = theano.config.floatX), borrow = True)

        b_values = np.zeros ((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared (value = b_values, borrow = True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d (input = self.input, filters = self.W,
                filter_shape = filter_shape, image_shape = image_shape)

        pooled_out = downsample.max_pool_2d (input = conv_out, ds = poolsize, ignore_border = True)

        self.output = T.tanh (pooled_out + self.b.dimshuffle ('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
    
    def return_output ():
        return self.output

def dropout (X, prob):

    if prob > 0.0:
        rand_str = RandomStreams()
        X *= rand_str.binomial (X.shape, p = 1 - prob, dtype = theano.config.floatX)
        X /= (1 - prob)

    return X

#########################################
## build convolutional neural network

def conv_neural_network (params): 
    ## params = (learning_rate, num_train_sample, num_conv_layer, num_feature_map, mini_batch_size, rand_seed, train_input, train_output, raw_train_output, test_input, num_test_sample, image_size, patch_size, pool_size, full_connected_hidden_units, output_type, max_iter, prob_dropout)

    learning_rate, num_train_sample, num_conv_layer, num_feature_map, mini_batch_size, rand_seed, train_input, train_output, raw_train_output, test_input, num_test_sample, image_size, patch_size, pool_size, full_connected_hidden_units, output_type, max_iter, prob_dropout = params

    rand_generator = np.random.RandomState (rand_seed)
    num_train_batch = num_train_sample/mini_batch_size
    num_test_batch = num_test_sample/mini_batch_size

    index = T.lscalar () 
    x = T.matrix ('x')
    y = T.ivector ('y')

    layer_input = x.reshape ((mini_batch_size, 1, image_size, image_size))

    layer = []
    for loop in np.arange (num_conv_layer):
        layer.append ([])

    layer[0] = LeNetConvPoolLayer (rand_generator, input = layer_input, image_shape = (mini_batch_size, 1, image_size, image_size), filter_shape = (num_feature_map[0], 1, patch_size, patch_size), poolsize = (pool_size, pool_size), prob_dropout_cnn = prob_dropout)

    updated_image_size = (image_size - patch_size + 1)/pool_size

    for loop in range (1, num_conv_layer):
        layer[loop] = LeNetConvPoolLayer (rand_generator, input = layer[loop - 1].output, image_shape = (mini_batch_size, num_feature_map[loop - 1], updated_image_size, updated_image_size), filter_shape = (num_feature_map[loop], num_feature_map[loop - 1], patch_size, patch_size), poolsize = (pool_size, pool_size), prob_dropout_cnn = prob_dropout)
        updated_image_size = (updated_image_size - patch_size + 1)/pool_size

    # pdb.set_trace ()

    fully_connected_input = layer [num_conv_layer - 1].output.flatten (2)
    fully_connected = HiddenLayer(rand_generator, input = fully_connected_input, n_in = num_feature_map [num_conv_layer - 1] * updated_image_size * updated_image_size, n_out = full_connected_hidden_units, activation = T.tanh, prob_dropout_hl = prob_dropout)
    fully_connected_output = LogisticRegression (input = fully_connected.output, n_in = full_connected_hidden_units, n_out = output_type, prob_dropout_lr = prob_dropout)

    cost = fully_connected_output.negative_log_likelihood (y)

    params = fully_connected_output.params + fully_connected.params

    for loop in np.arange (num_conv_layer - 1, -1, -1):
        params = params + layer[loop].params

    grads = T.grad (cost, params)

    # pdb.set_trace ()

    updates = [ (p_i, p_i - learning_rate * g_i) for p_i, g_i in zip (params, grads)]

    # pdb.set_trace ()

    train_model = theano.function ([index], cost, updates = updates,
          givens = {
          x: train_input [index * mini_batch_size: (index + 1) * mini_batch_size], 
          y: train_output [index * mini_batch_size: (index + 1) * mini_batch_size]})

    train_predict = theano.function ([index], fully_connected_output.y_pred, givens = {x: train_input [index * mini_batch_size: (index + 1) * mini_batch_size]})

    test_predict = theano.function ([index], fully_connected_output.y_pred, givens = {x: test_input [index * mini_batch_size: (index + 1) * mini_batch_size]})

    # pdb.set_trace ()

    print "Train the convoutional neural network ~~"

    # pdb.set_trace ()

    iter = 0
    while (iter < max_iter):

        iter = iter + 1

        for batch_index in np.arange (num_train_batch):

            train_cost = train_model (batch_index)
            if batch_index % 100 == 0:
                print "Iter: " + str (iter) + " Sample: " + str (batch_index * mini_batch_size) + " Cost: " + str(train_cost)

    layer [0].prob_dropout_cnn = 0
    layer [1].prob_dropout_cnn = 0
    fully_connected.prob_dropout_hl = 0
    fully_connected_output.prob_dropout_lr = 0

    train_predict_digits = pd.Series (np.concatenate ([train_predict (loop) for loop in np.arange (num_train_batch)]))

    # pdb.set_trace ()

    train_predict_error = sum([output_1 != output_2 for output_1, output_2 in zip(train_predict_digits, raw_train_output)])/(1.0 * num_train_sample)

    print "Train dataset prediction error: " + str (train_predict_error * 100) + " %"

    print "Predict test dataset ~~"

    test_predict_digits = pd.Series (np.concatenate ([test_predict (loop) for loop in np.arange (num_test_batch)]))

    test_output_file = "cnn_dropout.csv"
    with open(test_output_file, "wb") as f:
        f.write(b'Label\n')
        np.savetxt (f, test_predict_digits.astype (int), fmt = '%i')

#########################################
## read data

## train dataset

# data_path = "/Users/xiaoyiliu/Dropbox/Digit_Recognizer/Data/"
data_path = "C:/Users/Xiaoyi Liu/Dropbox/Digit_Recognizer/Data/"

train_data_path = data_path + "train.csv" 
train = pd.read_csv (train_data_path, sep = ",")

train_input = pd.DataFrame.as_matrix (train.iloc[:, range(1, train.shape [1])])
train_output = pd.DataFrame.as_matrix (train.iloc[:, range (0, 1)]).ravel()

raw_train_output = train_output

train_input = data_norm (train_input)
num_train_sample = len (train_input)

train_input = theano.shared (np.asarray(train_input, dtype = theano.config.floatX), borrow = True)
train_output = theano.shared (np.asarray(train_output, dtype = theano.config.floatX), borrow = True)

train_output = T.cast (train_output, 'int32')

## test dataset

test_data_path = data_path + "test.csv" 
test = pd.read_csv (test_data_path, sep = ",")

test_input = pd.DataFrame.as_matrix (test)
test_input = data_norm (test_input)

num_test_sample = len (test_input)

test_input = theano.shared (np.asarray(test_input, dtype = theano.config.floatX), borrow = True)

learning_rate = 0.001
num_conv_layer = 2
num_feature_map = (30, 100)
mini_batch_size = 1
rand_seed = 86027768
image_size = 28
patch_size = 5
pool_size = 2
full_connected_hidden_units = 250
output_type = 10
max_iter = 300
prob_dropout = 0.1


params = (learning_rate, num_train_sample, num_conv_layer, num_feature_map, mini_batch_size, rand_seed, train_input, train_output, raw_train_output, test_input, num_test_sample, image_size, patch_size, pool_size, full_connected_hidden_units, output_type, max_iter, prob_dropout)

# pdb.set_trace ()

conv_neural_network (params)

#########################################
## time consumed

end_time = datetime.datetime.now()

print "Running Time:" + str (end_time - start_time)
