import numpy as np
import models as md

class NeuralNet:
	def __init__(self):
		self.layers = []
		self.training_data = None
		self.training_labels = None
		self.test_data = None
		
	def fprop(self, data):
		layer_input = np.array(data)
		for layer in self.layers:
			layer_output = layer.fprop(layer_input)
			layer_input = layer_output
		return layer_output

	def bprop(self, grad_output):
		grad_in = np.array(grad_output)
		for layer in reversed(self.layers):
			grad_out = layer.bprop(grad_in)
			grad_in = grad_out
		return grad_out

	def update_weights(self):
		for layer in self.layers:
			grad_w = layer.get_gradients()
			layer.update_weights(grad_w, md.grad_desc)

	def train(self, data, labels, parameters, method="minibatch"):
		if method == "minibatch":
			self.train_minibatch(data, labels, parameters)
		elif method == "stochastic":
			self.train_stochastic(data, labels, parameters)
		pass


	def train_minibatch(self, data, labels, parameters):
		# Get the relevant parameters
		nEpochs = parameters["nEpochs"]
		nBatch = parameters["nBatch"]
		nClasses = self.layers[-1].nd

		# Get the feature space
		nSamples = data.shape[0]

		# Assign variables
		nBatches = int(np.ceil(float(nSamples) / nBatch))
		mean_err = np.zeros((nBatches,1))

		# Train the network
		print("Training neural network...")
		for epoch in range(0,nEpochs):
			# Use random permutation of the data
			indices = np.random.permutation(np.arange(nSamples))
			mean_err *= 0
			for batch in range(0, nBatches):
				batch_indices = indices[np.arange(batch * nBatch, min((batch + 1) * nBatch, nSamples))]
				mean_err = self.train_net(data[batch_indices,:], labels[batch_indices], nClasses, len(batch_indices))
			print(f"At Epoch: {epoch}\nMean error: {np.mean(mean_err)}")

	def train_stochastic(self, data, labels, parameters):
		# Get the relevant parameters
		nEpochs = parameters["nEpochs"]
		nClasses = self.layers[-1].nd

		# Get the feature space
		nSamples, nD = data.shape

		# Assign extra variables
		mean_err = np.zeros((nSamples,1))

		# Train the network
		print("Training neural network...")
		for epoch in range(0,nEpochs):
			for i in range(0, nSamples):
				mean_err[i] = self.train_net(data[i,:][np.newaxis,:], labels[i], nClasses, 1)
			print(f"At Epoch: {epoch}\nMean error: {np.mean(mean_err)}")

	def add_layer(self, model):
		self.layers.append(model)

	def set_training_data(self, data):
		self.training_data = data

	def set_training_labels(self, labels):
		self.training_labels = labels

	def train_net(self, batch_data, batch_labels, nClasses, nSamples):
		# Doing forward pass through network
		output = self.fprop(batch_data)

		# Create cross entropy error function
		error = md.CrossEntropyLoss(nClasses)
		error.set_labels(batch_labels)
		err = error.fprop(output)
		grad_output = np.ones((nSamples,1)) * 1.0/float(nSamples)
		grad_output = error.bprop(grad_output)

		# Do backward pass through the network
		self.bprop(grad_output)

		# Update the weights
		self.update_weights()

		return np.mean(err)