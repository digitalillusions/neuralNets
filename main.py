import numpy as np
import pandas as pd
import pickle as pkl
import models as md
import network as netw


	
def init_net_2a(**kwargs):
	# Read in training data
	print("Reading in data...")
	train_params = {"nEpochs": 100, "nBatch": 600}
	nEpochs = 100
	nBatch = 600
	nSamples = None
	nClasses = 10
	train_data = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/mnist-train-data.csv", np.float, nSamples)
	train_data /= 255.
	train_labels_temp = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/mnist-train-labels.csv", np.int, nSamples)
	train_labels = np.squeeze(train_labels_temp)

	# Get the feature space
	nSamples, nD = train_data.shape

	# Create a new network and add layers
	print("Creating neural network...")
	net = netw.NeuralNet()
	net.add_layer(md.Linear(nD,200))
	net.add_layer(md.ReLU(200))
	net.add_layer(md.Linear(200,nClasses))
	net.add_layer(md.Softmax(nD))
	net.init_weights(method="glorot")

	# Load model if specified and availiable
	if('load_model' in kwargs and kwargs['load_model']==True):
		if('model_path' in kwargs):
			print("Loading neural network...")
			with open(kwargs['model_path'], 'rb') as output:
				net = pkl.load(output)
		else:
			print("Please specify a path to the model")
	else:
		# Train the network
		print("Training neural network...")
		for epoch in range(0,nEpochs):
			#print("Training in Epoch {0}".format(epoch))

			# Use random permutation of the data
			indices = np.random.permutation(np.arange(nSamples))

			for batch in range(0, int(np.ceil(float(nSamples) / nBatch))):
				batch_indices = indices[np.arange(batch * nBatch, min((batch + 1) * nBatch, nSamples))]
				mean_err = net.train_net(train_data[batch_indices,:], train_labels[batch_indices], nClasses, len(batch_indices))
			print("At Epoch: {0}\nMean error: {1}".format(epoch, mean_err))

	# Save the model if desired
	if('save_model' in kwargs and kwargs['save_model']==True):
		if('model_path' in kwargs):
			print(f"Writing neural network to: {kwargs['model_path']}")
			with open(kwargs['model_path'], 'wb') as output:
				pkl.dump(net, output)
		else:
			print("Please specify a path to the model")
	

	valid_data = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/mnist-valid-data.csv", np.float, nSamples)
	valid_data /= 255.
	valid_labels_temp = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/mnist-valid-labels.csv", np.int, nSamples)
	valid_labels = np.squeeze(valid_labels_temp)

	nValSamples,nDim = valid_data.shape

	classify = net.fprop(valid_data)
	errors = nValSamples - np.count_nonzero(valid_labels == np.argmax(classify, axis=1))
	print(f"Total numer of missclassified data points: {errors}")

def nonLinearTest():
	# Read in training data
	print("Reading in data...")
	train_params = {"nEpochs": 100, "nBatch": 600}
	addFeatures = 20
	nSamples = None
	nClasses = 2
	train_data = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/train_data.csv", np.float, nSamples, ',')
	train_labels_temp = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/train_label.csv", np.int, nSamples, ',')
	train_data /= 10
	train_labels = np.squeeze(train_labels_temp)

	# Get the feature space
	nSamples, nD = train_data.shape
	print(f"Number of samples: {nSamples}; Number of Dimensions: {nD}")

	# Create a new network and add layers
	print("Creating neural network...")
	net = netw.NeuralNet()
	net.add_layer(md.Linear(nD,nD+addFeatures))
	net.add_layer(md.ReLU(nD+addFeatures))
	net.add_layer(md.Linear(nD+addFeatures,nClasses))
	net.add_layer(md.Softmax(nD))
	net.init_weights(method="glorot")

	net.train(train_data, train_labels, train_params, method="minibatch")

	valid_data = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/valid_data.csv", np.float, nSamples, ',')
	valid_labels_temp = readDataFromCsv("/Users/stefanjeske/Downloads/mnist/valid_label.csv", np.int, nSamples, ',')
	valid_data /= 10
	valid_labels = np.squeeze(valid_labels_temp)

	nValSamples,nDim = valid_data.shape

	classify = net.fprop(valid_data)
	errors = nValSamples - np.count_nonzero(valid_labels == np.argmax(classify, axis=1))
	print(f"Total numer of missclassified data points: {errors}")


def readDataFromCsv(filename, dtype=np.float, nrows=None, sep=' '):
	array = pd.read_csv(filename, sep=sep, header=None, nrows=nrows, dtype=dtype)
	return array.values


def main():
	print("Henlo: Welcome to training stupid networks...")
	
	#init_net_2a(save_model=True, model_path='./model_relu.pkl')
	#init_net_2a()
	nonLinearTest()

if __name__ == '__main__':
	main()
