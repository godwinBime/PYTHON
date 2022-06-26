import random
import numpy as np
from past.builtins import xrange

"""Miscellaneous functions"""


def sigmoidPrime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network(object):
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def evaluate(self, testData):
        """
        Return the number of test inputs for which the
        neural network outputs the correct result. Note
        that the neural network's output is assumed to
         be the index of whichever neuron in the first
         layer has the highest activation.
        :param self:
        :param testData:
        :return:
        """
        testResults = [(np.argmax(self.feedForward(x)), y)
                       for (x, y) in testData]
        return sum(int(x == y) for (x, y) in testResults)

    """
    Cost derivative.
    """

    def costDerivative(self, outputActivation, y):
        """
        Return the vector of partial derivatives
        \partial cX/ \partial a for the output activations.
        :param self:
        :param outputActivation:
        :param y:
        :return:
        """
        return outputActivation - y

    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData=None):
        """
       Train the neural network using mini-batch stochastic
       gradient descent. The "trainingData" is a list of tuples
        "(x, y)" representing the training inputs and the
        desired outputs. the other non-optional parameters are
        self-explanatory. If "testData" is provided then the
        network will be evaluated against the test data after
        each epoch, and partial progress printed out. This is
        useful for tracking progress, but slows things down
        substantially.
       :param trainingData:
       :param epochs:
       :param miniBatchSize:
       :param eta:
       :param testData:
       :return:
       """
        nTest = None
        if testData:
            nTest = len(testData)
        n = len(trainingData)
        for j in xrange(epochs):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[k:k + miniBatchSize]
                for k in xrange(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))

    def feedForward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    """
    Update the network's weights and biases by applying 
    gradient descent using backpropagation to a single 
    mini batch. The ''miniBatch'' is a list of tuples 
    ''(x, y)'', and ''eta'' is the learning rate.
    """

    def updateMiniBatch(self, miniBatch, eta):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backProp(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.weights = [w - (eta / len(miniBatch)) * nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (eta / len(miniBatch)) * nb
                       for b, nb in zip(self.biases, nablaB)]

    def backProp(self, x, y):
        """Return a tuple ''(nablaB, nablaW)'' representing
        the gradient for the cost function cX. ''nablaB'' and
        ''nablaW'' are Layer-by-Layer list of numpy arrays,
        similar to ''self.biases'' and ''self.weights''
        """
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        # feedForward
        activation = x
        activations = [x]  # List to store all the activations, layer by layer
        zs = []  # List to store all the z-vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activations[-2].transpose())
        """
        Note that the variable l in the loop below is used
        a little differently to the notation in chapter 2
        of the book. Here, L = 1 means the last layer of 
        neurons, L = 2 is the second to the last layer, 
        and so on. It's a renumbering of the scheme in the 
        book, used here to take advantage of the fact that 
        python can use negative indices in list.
        """
        for l in xrange(2, self.numLayers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nablaB, nablaW
