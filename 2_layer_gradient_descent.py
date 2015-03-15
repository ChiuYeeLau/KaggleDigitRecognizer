import pandas as pd
import numpy as np
import pdb

import matplotlib.pyplot as plt

## activation function

sigmoid = lambda x: 1.0/(1 + np.exp (-x))
sigmoid_derivative = lambda x: x * (1 - x)

tanh = lambda x: np.tanh (x)
tanh_derivative = lambda x: 1 - x * x

## read data

data_path_mac = "/Users/xiaoyiliu/Dropbox/Digit_Recognizer/Data/"
data_path_win = "C:/Users/Xiaoyi Liu/Dropbox/Digit_Recognizer/Data/"

train_data_path = data_path_mac + "train.csv" 
train = pd.read_csv (train_data_path, sep = ",")

## neural network

class neural_network:
	def __init__ (self, input_data, output_label, nn_parameters):
		print "Neural Network initialization!"
		## input_data: m * n, where m is the number of training samples, n is the number of features
		## output_label: m * 1 output labels
		## nn_parameters: [num_label_type, num_layer, size_each_layer, regularization, active_function, active_function_derivative, max_iteration, learning_rate]
		## num_label_type: 10
		## num_layer: number of weight matrices
		## size_each_layer: starting from input size m
		## active_function: sigmoid, tanh
		## active_function_derivative: sigmoid_derivative, tanh_derivative
		self.input_data = np.array(input_data)
		self.output_label = np.array(output_label)
		self.train_sample_size = len (output_label)
		self.num_label_type = nn_parameters [0]
		self.num_layer = nn_parameters [1]
		self.size_each_layer = nn_parameters [2]
		self.regularization = nn_parameters [3]
		self.active_function = nn_parameters [4]
		self.active_function_derivative = nn_parameters [5]
		self.max_iteration = nn_parameters [6]
		self.learning_rate = nn_parameters [7]

		## initialize the weights for each layer
		self.layer_weight = []
		self.cost = 0
		self.delta = []
		self.weight_derivative = []

		epsilon = []

		for layer_loop in np.arange (self.num_layer):
			epsilon.append (4.0 * np.sqrt(6.0)/np.sqrt(self.size_each_layer [layer_loop] + 1.0 + self.size_each_layer [layer_loop + 1]))

		# pdb.set_trace()

		for layer_loop in np.arange (self.num_layer):
			self.layer_weight.append (np.random.uniform (-epsilon [layer_loop], epsilon [layer_loop], [self.size_each_layer [layer_loop] + 1.0, self.size_each_layer [layer_loop + 1]]))
			self.weight_derivative.append (np.zeros ((self.size_each_layer [layer_loop] + 1.0, self.size_each_layer [layer_loop + 1])))

		self.output_label_matrix = np.zeros ((self.train_sample_size * self.num_label_type, 1))
		output_label_index = list (np.array(self.output_label).reshape(-1, )) + self.num_label_type * np.arange (self.train_sample_size)
		self.output_label_matrix [output_label_index] = np.ones ((self.train_sample_size, 1))
		self.output_label_matrix = self.output_label_matrix.reshape ((self.train_sample_size, self.num_label_type))

		# pdb.set_trace()


	def feed_forward (self):
		## Feed-Forward: Compute the cost function!
		## compute output at each layer

		# pdb.set_trace()

		self.output_each_layer = []
		self.output_each_layer.append (self.input_data)

		for layer_loop in np.arange (self.num_layer):
			input_layer = self.output_each_layer [layer_loop]
			input_layer = np.append (np.ones ([self.train_sample_size, 1]), input_layer, axis = 1)
			self.output_each_layer.append (np.array(map (self.active_function, np.dot (input_layer, self.layer_weight [layer_loop]))))

		# pdb.set_trace ()

	def back_propagation (self):
		## Back-Propagation: Compute the derivative!

		self.delta = []

		for layer_loop in np.arange (self.num_layer):
			self.delta.append ( np.zeros([self.train_sample_size, self.size_each_layer [layer_loop]]))

		self.delta [self.num_layer - 1] = self.output_each_layer [self.num_layer] - np.transpose(np.atleast_2d(self.output_label))

		# pdb.set_trace()

		for layer_loop in np.arange (self.num_layer - 2, -1, -1):
			self.delta [layer_loop] = np.multiply(np.dot(self.delta[layer_loop + 1], np.transpose(self.layer_weight [layer_loop + 1])) [:, 1 : self.size_each_layer [layer_loop + 1] + 1], self.active_function_derivative (self.output_each_layer [layer_loop + 1]))

		for layer_loop in np.arange (self.num_layer):
			self.weight_derivative [layer_loop] = np.dot (np.transpose(np.append (np.ones ([self.train_sample_size, 1]), self.output_each_layer [layer_loop], axis = 1)), self.delta [layer_loop])/(1.0 * self.train_sample_size)

		for layer_loop in np.arange (self.num_layer):
			self.weight_derivative [layer_loop] [:, (range (1, len (self.weight_derivative [layer_loop] [0])))] = self.weight_derivative [layer_loop] [:, (range (1, len (self.weight_derivative [layer_loop] [0])))] + self.regularization * self.layer_weight [layer_loop] [:, (range (1, len (self.weight_derivative [layer_loop] [0])))]/(1.0 * self.train_sample_size)


	def train (self):
		print "Train the Neural Network!"

		for loop in np.arange (self.max_iteration):
			if loop % 10000 == 0: 
				print "Iterations:", loop

			self.feed_forward ()
			self.back_propagation ()
			for layer_loop in np.arange (self.num_layer):
				self.layer_weight [layer_loop] = self.layer_weight [layer_loop] - self.learning_rate * self.weight_derivative [layer_loop]


	def predict (self):
		print "Predict!"

		self.feed_forward ()

		self.prediction = np.max (self.output_each_layer [self.num_layer], axis = 1)
		self.prediction = np.transpose (self.prediction)
		self.prediction_error = 0.0
		for loop in np.arange (self.train_sample_size):
			if self.prediction [loop] != self.output_label [loop]:
				self.prediction_error = self.prediction_error + 1.0

		self.prediction_error = self.prediction_error/(1.0 * self.train_sample_size)
		print self.output_each_layer [self.num_layer]



input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [0, 1, 1, 0]

parameters = [2, 2, [2, 2, 1], 0, sigmoid, sigmoid_derivative, 1000, 0.02]

n = neural_network (input_data, output_data, parameters)
n.train ()
n.predict ()
