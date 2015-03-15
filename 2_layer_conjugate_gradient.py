import pandas as pd
import numpy as np
import pdb
from scipy import optimize
import datetime

start_time = datetime.datetime.now()

###############################################################################
## activation function

sigmoid = lambda x: 1.0/(1 + np.exp (-x))
sigmoid_derivative = lambda x: x * (1 - x)

tanh = lambda x: np.tanh (x)
tanh_derivative = lambda x: 1 - x * x

###############################################################################
## data normalization

def data_norm (raw_data):
	
	return raw_data/(1.0 * 255) - np.transpose (np.atleast_2d (np.mean (raw_data/(1.0 * 255), axis = 1)))

###############################################################################
## feed forward

def feed_forward (layer_weight, *args):

	ff_layer_weight = layer_weight

	ff_input_data, ff_output, ff_num_layer, ff_num_layer_units, ff_num_train_sample, ff_num_weight_coeffs, ff_active_func, ff_active_func_derivative, ff_regular_lambda, ff_output_matrix = args
	## "ff" for feed forward

	ff_output_layers = []
	ff_output_layers.append (ff_input_data)

	for layer_loop in np.arange (ff_num_layer):
		ff_output_layers.append (np.array(map (ff_active_func, np.dot (np.append (np.ones ([ff_num_train_sample, 1]), ff_output_layers [layer_loop], axis = 1), ff_layer_weight [layer_loop]))))

	return ff_output_layers

###############################################################################
## cost function

def cost_func (weight, *args):

	cf_weight = weight

	cf_input_data, cf_output, cf_num_layer, cf_num_layer_units, cf_num_train_sample, cf_num_weight_coeffs, cf_active_func, cf_active_func_derivative, cf_regular_lambda, cf_output_matrix = args
	## "cf" for cost func

	###############################################################################
	## unpack 1-D weight array into weight matrix at each layer

	cf_layer_weight = []
	for layer_loop in np.arange (cf_num_layer):
		cf_layer_weight.append (np.reshape (cf_weight [sum (cf_num_weight_coeffs[0 : layer_loop + 1]) : sum (cf_num_weight_coeffs [0 : layer_loop + 1]) + cf_num_weight_coeffs [layer_loop + 1]], [cf_num_layer_units [layer_loop] + 1,  cf_num_layer_units [layer_loop + 1]]))

	# pdb.set_trace ()

	###############################################################################
	## feed forward: calculate output and cost 

	cf_output_layers = feed_forward (cf_layer_weight, *args)

	# pdb.set_trace ()

	cost_matrix = - np.multiply (cf_output_matrix, np.log (cf_output_layers [num_layer])) - np.multiply (1 - cf_output_matrix, np.log (1 - cf_output_layers [num_layer]))

	cost = sum(sum (cost_matrix))/(1.0 * cf_num_train_sample)

	regularization = 0

	for layer_loop in np.arange (cf_num_layer):
		regularization = regularization + sum(sum (np.power (cf_layer_weight [layer_loop][1 :cf_num_layer_units [layer_loop] + 2, :], 2.0)))

	cost = cost + cf_regular_lambda * regularization/(2.0 * cf_num_train_sample)

	print "Cost:" + str (cost) + "!"

	return cost

###############################################################################
## conjugate gradient and back propagation

def conjugate_gradient (weight, *args):

	cg_weight = weight

	cg_input_data, cg_output, cg_num_layer, cg_num_layer_units, cg_num_train_sample, cg_num_weight_coeffs, cg_active_func, cg_active_func_derivative, cg_regular_lambda, cg_output_matrix = args
	## "cg" for conjugate gradient 

	cg_layer_weight = []

	for layer_loop in np.arange (cg_num_layer):
		cg_layer_weight.append (np.reshape (cg_weight [sum (cg_num_weight_coeffs[0 : layer_loop + 1]) : sum (cg_num_weight_coeffs[0 : layer_loop + 1]) + cg_num_weight_coeffs [layer_loop + 1]], [cg_num_layer_units [layer_loop] + 1, cg_num_layer_units [layer_loop + 1]]))

	weight_derivative_matrix = []
	weight_derivative_vector = np.zeros (sum (cg_num_weight_coeffs))

	for layer_loop in np.arange (cg_num_layer):
		weight_derivative_matrix.append (np.zeros (cg_layer_weight [layer_loop].shape))

	cg_output_layers = feed_forward (cg_layer_weight, *args)

	delta = []

	for layer_loop in np.arange (cg_num_layer):
		delta.append (np.zeros([cg_num_train_sample, cg_num_layer_units [layer_loop]]))

	delta [cg_num_layer - 1] = cg_output_layers [cg_num_layer] - cg_output_matrix

	for layer_loop in np.arange (cg_num_layer - 2, -1, -1):
		delta [layer_loop] = np.multiply(np.dot(delta [layer_loop + 1], np.transpose(cg_layer_weight [layer_loop + 1])) [:, 1 : cg_num_layer_units [layer_loop + 1] + 1], cg_active_func_derivative (cg_output_layers [layer_loop + 1]))

	for layer_loop in np.arange (cg_num_layer):
		weight_derivative_matrix [layer_loop] = np.dot (np.transpose (np.append (np.ones ([cg_num_train_sample, 1]), cg_output_layers [layer_loop], axis = 1)), delta [layer_loop])/(1.0 * cg_num_train_sample)

	for layer_loop in np.arange (cg_num_layer):
		weight_derivative_matrix [layer_loop] [1 : cg_num_layer_units [layer_loop] + 2, :] = weight_derivative_matrix [layer_loop] [1 :cg_num_layer_units [layer_loop] + 2, :] + cg_regular_lambda * cg_layer_weight [layer_loop] [1 : cg_num_layer_units [layer_loop] + 2, :] /(1.0 * cg_num_train_sample)

	# pdb.set_trace ()

	## re-form weight derivative matrix into vector
	for layer_loop in np.arange (cg_num_layer):
		weight_derivative_vector [sum (cg_num_weight_coeffs [0 : layer_loop + 1]) : sum (cg_num_weight_coeffs [0 : layer_loop + 1]) + cg_num_weight_coeffs [layer_loop + 1]] = np.transpose(np.reshape (weight_derivative_matrix [layer_loop], (cg_num_weight_coeffs [layer_loop + 1], 1)))

	# pdb.set_trace ()

	return weight_derivative_vector

###############################################################################
## read data

data_path_mac = "/Users/xiaoyiliu/Dropbox/Digit_Recognizer/Data/"
data_path_win = "C:/Users/Xiaoyi Liu/Dropbox/Digit_Recognizer/Data/"

train_data_path = data_path_win + "train.csv" 
train = pd.read_csv (train_data_path, sep = ",")

test_data_path = data_path_win + "test.csv" 
test = pd.read_csv (test_data_path, sep = ",")

###############################################################################
## neural network initialization

input_data = pd.DataFrame.as_matrix (train.iloc[:, range(1, train.shape [1])])
output = pd.DataFrame.as_matrix (train.iloc[:, range (0, 1)]).ravel() + 1

input_data = data_norm (input_data)

# pdb.set_trace ()

num_layer = 2
num_layer_units = [784, 500, 10]

num_train_sample = len (input_data)
active_func = sigmoid
active_func_derivative = sigmoid_derivative
regular_lambda = 1.1

## weight initialization

num_weight_coeffs = [0]

for layer_loop in np.arange (num_layer):
	num_weight_coeffs.append((num_layer_units [layer_loop] + 1) * num_layer_units [layer_loop + 1])

weight_initial = np.zeros (sum (num_weight_coeffs))

for layer_loop in np.arange (num_layer):
	uniform_bound = 4.0 * np.sqrt(6.0)/np.sqrt(num_layer_units [layer_loop] + 1.0 +num_layer_units [layer_loop + 1]) 

	weight_initial [sum (num_weight_coeffs[0 : layer_loop + 1]) : sum (num_weight_coeffs[0 : layer_loop + 1]) + num_weight_coeffs [layer_loop + 1]] = np.random.uniform (- uniform_bound, uniform_bound, num_weight_coeffs [layer_loop + 1])

# pdb.set_trace ()

###############################################################################
## re-form output

output_matrix = np.zeros ([num_train_sample * num_layer_units [num_layer], 1])
output_index = list (np.array (output).reshape(-1, )) + num_layer_units [num_layer] * np.arange (num_train_sample) - 1
output_matrix [output_index] = np.ones ([num_train_sample, 1])

output_matrix = output_matrix.reshape ([num_train_sample, num_layer_units [num_layer]])

# pdb.set_trace ()

###############################################################################
## train neural network

args = (input_data, output, num_layer, num_layer_units, num_train_sample, num_weight_coeffs, active_func, active_func_derivative, regular_lambda, output_matrix)

train_neural_network = optimize.fmin_cg (cost_func, weight_initial, fprime = conjugate_gradient, args = args, disp = False)

opt_weight = train_neural_network

opt_layer_weight = []

for layer_loop in np.arange (num_layer):
	opt_layer_weight.append (np.reshape (opt_weight [sum (num_weight_coeffs [0 : layer_loop + 1]) : sum (num_weight_coeffs [0 : layer_loop + 1]) + num_weight_coeffs [layer_loop + 1]], [num_layer_units [layer_loop] + 1, num_layer_units [layer_loop + 1]]))

###############################################################################
## predict train set

opt_train_output_layers = feed_forward (opt_layer_weight, *args)

opt_output_train_label = np.argmax (opt_train_output_layers [num_layer], axis = 1) + 1

prediction_train_error = sum (opt_output_train_label != output) * 1.0/num_train_sample

train_result = sum (opt_output_train_label != output)

print "Prediction Train Set Error: " + str (prediction_train_error * 100) + "%"

###############################################################################
## predict test set

input_data = pd.DataFrame.as_matrix (test)
input_data = data_norm (input_data)

num_train_sample = len (input_data)

test_output_layers = []
test_output_layers.append (input_data)

for layer_loop in np.arange (num_layer):
		test_output_layers.append (np.array(map (active_func, np.dot (np.append (np.ones ([num_train_sample, 1]), test_output_layers [layer_loop], axis = 1), opt_layer_weight [layer_loop]))))

output_test_label = np.argmax (test_output_layers [num_layer], axis = 1)

# pdb.set_trace ()

test_output_file = "test_output_normalized_lambda_110.csv"

with open(test_output_file, "wb") as f:
    f.write(b'Label\n')
    np.savetxt(f, output_test_label.astype(int), fmt = '%i')

end_time = datetime.datetime.now()

print "Running Time:" + str (end_time - start_time)
