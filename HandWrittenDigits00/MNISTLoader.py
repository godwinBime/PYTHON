"""A library to load the MNIST image data.
For details of the data structures that are
returned, see the doc string
for ''load_data'' and ''load_data_wrapper''.
In practice, ''load_data_wrapper''
is the function usually called by our
neural network code. """

import pickle
import gzip
import numpy as np
import tensorflow as tf


def loaderData():
    """
    Return the MNIST data as a tuple containing
    the training data, the validation data, and
    the test data. The ''trainingData'' is return
    as a tuple with two entries. This is a numpy
    ndarray with 50,00 entries. Each entry is in
    turn, a numpy ndarray with 784 values, representing
    the 28 * 28 = 784 pixels in a single MNIST image.

    The second entry in the ''trainingData'' tuple is
    a numpy ndarray containing 50,000 entries. Those
    entries are just the digit values (0-9) for the
    corresponding images contained in the first entry
    of the tuple.

    The ''validationData'' and the ''testData'' are
    similar, except each contains only 10,000 images.

    This is a nice data format, but for use in neural
    networks, it's helpful to modify the format of the
    ''trainingData'' a little. That's done in the wrapper
    function ''loadDataWrapper()'' see below
    :return:
    """

    f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    f.close()
    return trainingData, validationData, testData


def loadDataWrapper():
    """
    Return a tuple containing ''(trainingData, validationData,
    testData)''. Based on ''loadData'', but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ''trainingData'' is a list containing 50,000
    2-tuples ''(x, y)''. ''x'' is a 784-dimensional numpy.ndarray
    containing the input image. ''y'' ia a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to
    the correct digit for ''x''.

    ''validationData'' and ''testData'' are lists containing 10,000
    2-tuples ''(x, y)''. In each case, ''x'' is a 784-dimensional
    numpy.ndarray containing the input image, and ''y'' is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ''x''.

    Obviously, this means we're using slightly different formats for
    the training, testing and validation data. These formats turn out
    to be the most convenient for use in our neural network code.
    :return:
    """
    trD, vaD, teD, = loaderData()
    trainingInputs = [np.reshape(x, (784, 1)) for x in trD[0]]
    trainingResults = [vectorizedResult(y) for y in trD[1]]
    trainingData = list(zip(trainingInputs, trainingResults))

    validationInputs = [np.reshape(x, (784, 1)) for x in vaD[0]]
    validationData = list(zip(validationInputs, vaD[1]))

    testInputs = [np.reshape(x, (784, 1)) for x in teD[0]]
    testData = list(zip(testInputs, teD[1]))
    return trainingData, validationData, testData


def vectorizedResult(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeros elsewhere. This is used to convert a digit
    (0-9) into a corresponding desired output from the neural network.
    :param j:
    :return:
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
