import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
	"""Abstract class for all modular components in the neural network.
	These components contain linear layers as well as non-linear elements
	such as ReLU or tanh.
	"""

	def __init__(self):
		super().__init__()
		pass

	def init_weights(self):
		pass
	
	@abstractmethod
	def fprop(self):
		pass

	@abstractmethod
	def bprop(self):
		pass

	def update_weights(self,*args,**kwargs):
		pass

	def get_gradients(self,*args,**kwargs):
		pass
		

	
class Linear(Module):
	"""The linear element. This elements is simply used to connect the neurons
	of one layer to all of those of another layer. Very simple fully connected.
	"""
	def __init__(self,nd_, ny_):
		"""Setting up the environment for the layer to work properly, when the first 
		fprop is called. The weight matrix will be of size (nd, ny)

		Arguments:
		nd_ -- The number of input dimensions as in the number of input features
		ny_ -- The number of output dimensions, e.g. the number of output classes
		"""
		self.nd = nd_ 														# Number of features 
		self.ny = ny_		 	
		self.W = None		 												# Weight vector (ny x nd)
		self.bias = np.zeros((1,self.ny))									# Bias (ny x 1)
		self.input = None
		self.output = None
		self.grad_w = None
		self.grad_w0 = None

	def init_weights(self, method="glorot"):
		"""Initilize the weights corresponding to some non linearity lateron.

		Keyword arguments:
		method -- Method to use for initilization of weights (default "glorot")
		"""
		if(method=="glorot"):
			self.W = init_weights_glorot(self.nd, self.ny, self.nd, self.ny)
		elif(method=="he"):
			self.W = init_weights_he(self.nd, self.ny, self.nd, self.ny)

	def fprop(self, z):
		""" Compute output of layer with input z assumed to be in (nSamples x nD)"""
		# Get shape and multiply weights on input in the correct order raise error 
		self.input = np.array(z)
		self.output = np.dot(z, self.W) + self.bias

		return np.array(self.output)

	def bprop(self, grad_output):
		""" Compute gradient according to backpropagation grad_output is of dimension  """
		self.grad_w = np.dot(self.input.transpose(), grad_output)
		self.grad_w0 = np.sum(grad_output, 0)
		return np.dot(grad_output, self.W.transpose())

	def get_gradients(self):
		""" Return gradients w.r.t. the weights """
		return np.array(self.grad_w), np.array(self.grad_w0)

	def update_weights(self, weights, up_fun):
		""" Update the weight coefficients """
		self.W = up_fun(self.W, weights[0])
		self.bias = up_fun(self.bias, weights[1])


class Softmax(Module):
	"""Softmax module computing the softmax function. This implementation uses the
	largest input vector in order to reduce the magnitude of the computed 
	exponention as in: softmax(a-b) = softmax(a)
	"""
	def __init__(self, nd_):
		"""Setup the module where nd_ is the input and output dimension"""
		self.nd = nd_
		self.input = None
		self.output = None
		self.grad_w = None
		

	def fprop(self, z):
		""" Compute output of layer with input z assumed to be in (nSamples x nd) """
		self.input = np.array(z)
		inp_max = np.max(z, 1)
		exp = np.exp(z.transpose() - inp_max).transpose()
		exp_col_sum = np.sum(exp, 1)
		softmax = (exp.transpose() / exp_col_sum).transpose()
		self.output = np.array(softmax)
		return softmax

	def bprop(self, grad_output):
		""" Compute the gradient of the output w.r.t. the input """
		grad_input = grad_output - np.sum(np.multiply(grad_output, self.output), axis=1)[:,np.newaxis]
		grad_input = np.multiply(grad_input, self.output)
		return grad_input

class CrossEntropyLoss(Module):
	def __init__(self, nd_):
		self.nd = nd_
		self.input = None
		self.output = None
		self.t_labels = None
		self.nSamples = None
		

	def fprop(self, z):
		self.t_labels = np.squeeze(self.t_labels)
		self.nSamples = z.shape[0]
		self.input = np.array(z)
		output = (-1)*np.log(z[np.arange(self.nSamples), self.t_labels])
		self.output = np.array(output)
		return output

	def bprop(self, grad_output):
		self.t_labels = np.squeeze(self.t_labels)
		grad_l = np.zeros((self.nSamples, self.nd))
		grad_l[np.arange(self.nSamples), self.t_labels] = (-1.) * 1.0/self.input[np.arange(self.nSamples), self.t_labels]
		np.multiply(grad_output, grad_l, grad_l)
		return grad_l

	def set_labels(self, tlabels):
		self.t_labels = np.array(tlabels)

class ReLU(Module):
	def __init__(self, nd_):
		self.nd = nd_
		self.input = None
	
	def fprop(self, z):
		self.input = np.array(z)
		z[z < 0] = 0
		return z

	def bprop(self, grad_output):
		grad_output[self.input < 0] = 0
		grad_input = np.array(grad_output)
		return grad_input

def init_weights_glorot(nin, nout, nd, ny):
	sigma = np.sqrt(2/(nin+nout))
	weights = np.random.normal(0, sigma,((nd, ny)))		 		# Weight vector (nd x ny)
	return weights

def init_weights_he(nin, nout, nd, ny):
	sigma = np.sqrt(2/(nin))
	weights = np.random.normal(0, sigma,((nd, ny)))		 		# Weight vector (nd x ny)
	return weights


def grad_desc(value, grad):
	eta = 0.1
	return (value - eta*grad)