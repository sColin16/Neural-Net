'''This file contains classes to create and train Neural Networks,
and classes to preprocess data'''

from copy import deepcopy
from math import tanh, inf
import numpy as np
from time import time

# Experimental Constants
WEIGHT_MODIFIER = 3

'''Stores all the activation functions that a neural layer can use.
All functions must be compute the value for each value in a 2 dimensional
matrix, and accept an array of arguments with *args'''

activation_functions = {
    'ReLu': {
        'activator': lambda x, *args: np.maximum(x*args[0], x),
        'derivative': lambda x, *args: np.array(
            [[args[0] if num < 0 else 1 for num in row] for row in x])
    },
    'tanh': {
        'activator': lambda x, *args: np.tanh(x),
        'derivative': lambda x, *args: 1 - (np.tanh(x))**2
    }
}


class Unit:
    '''Stores a matrix of values, and the values' cooresponding gradients'''

    def __init__(self, values, grads):
        self.values = values
        self.grad = grads


class CostGate:
    '''Handles cost and accuracy-related functions of a
    neural network. Sets the gradients to begin backpropogation'''

    def computeCost(self, predicted_labels, true_labels):
        '''Computes the mean cost of a network's predicted labels,
        given the true labels'''

        return ((0.5 * sum((predicted_labels.values - true_labels)**2)) /
                predicted_labels.values.shape[0])

    def set_gradients(self, predicted_labels, true_labels, learning_rate):
        '''Sets the upper level gradients of a network's output
        according to the derivative of the cost function with
        respect to the predicted labels.'''

        d_predicted_labels = (true_labels - predicted_labels.values)\
            * learning_rate
        predicted_labels.grads = d_predicted_labels

    def classification_percent(self, predicted_labels, true_labels):
        '''Computes the percent of examples that the network classifies
        correctly. Correct classification means the index of the highest
        output matches the true labels'''

        truth_table = (np.argmax(predicted_labels.values, axis=1) ==
                       np.argmax(true_labels, axis=1))
                       
        #print(np.where(truth_table == False)[0])

        return (np.count_nonzero(truth_table))/len(truth_table) * 100


class NeuralLayer:
    '''Basic building block of the neural network that uses matrix
    multiplication for forward and backward backpropogation'''

    def __init__(self, parameters, activation, activator_params=None,
                 learning_rate=1):
        '''Stores parameters ('a'), activation functions, and layer's
        learning rate, which can be modified to prevent vanishing gradients'''

        self.a = Unit(parameters, np.zeros(parameters.shape))
        self.activation = activation_functions[activation]['activator']
        self.activation_prime = activation_functions[activation]['derivative']
        self.activator_params = activator_params or []
        self.learning_rate = learning_rate

    @classmethod
    def new(cls, inputs, outputs, activation, activator_params=None,
            learning_rate=1, weight_multiplier=None):
        '''Initializes a new random neural layer by creating
        a random array of parameters'''

        # Shrinks weights for larger nets to prevent enourmous outputs
        weight_multiplier = weight_multiplier or WEIGHT_MODIFIER/inputs
        parameters = np.random.rand(inputs, outputs) * weight_multiplier

        return cls(parameters, activation, activator_params, learning_rate)

    def forward(self, x):
        '''Uses matrix multiplication and the activation function
        to compute the output of the layer.'''

        self.x = x
        self.z = np.matmul(self.x.values, self.a.values)
        self.output = Unit(self.activation(self.z, *self.activator_params),
                           np.zeros(self.z.shape))

        return self.output

    def backward(self):
        '''Computes the gradients of the cost function with respect to
        z, a, and x, and stores them accordingly. The parameters are also
        updated immediatly. The layer learning rate affects this layer only'''

        dz = np.multiply(self.activation_prime(self.z, *self.activator_params),
                         self.output.grads)
        da = np.matmul(self.x.values.T, dz)
        dx = np.matmul(dz, self.a.values.T)

        self.a.grads = da * self.learning_rate
        self.x.grads = dx

        self.a.values += self.a.grads


class NeuralNetwork:
    '''Stores multiple neural layers to
    form a fully connected neural network'''

    def __init__(self, layers, activation):
        '''Creates a neural network from a list of NeurakLayers'''

        self.activation = activation
        self.layers = layers

    @classmethod
    def from_parameters(cls, parameters, activation, activator_params=None):
        '''Creates a neural network from a list of parameters'''

        layers = [NeuralLayer(parameters[i], activation, activator_params)
                  for i in range(len(parameters))]

        return cls(layers, activation)

    @classmethod
    def new(cls, description, activation, activator_params=None):
        '''Creates a new random neural network by automatically creating'''

        layers = [NeuralLayer.new(description[i], description[i+1], activation,
                  activator_params) for i in range(len(description) - 1)]

        return cls(layers, activation)

    def forward(self, x):
        '''Automates feed forward through every layer'''

        x = Unit(x, np.zeros(x.shape))

        for layer in self.layers:
            x = layer.forward(x)

        self.output = x
        return self.output

    def backward(self):
        '''Automates backpropogation through each layer. Note that
        the upper-level gradients must be set by something
        (e.g. a CostGate) for gradients to be non-zero.'''

        for layer in reversed(self.layers):
            layer.backward()

    def give_parameters(self):
        '''Returns the array of parameters from every layer.
        Can be used to recreate a neural network'''

        return [layer.a.values for layer in self.layers]

    def set_parameters(self, parameters):
        '''Automatically resets a networks parameters.
        used by the Trainer class to reset optimal parameters'''

        for i in range(len(self.layers)):
            self.layers[i].a.values = parameters[i]


class Trainer:
    '''Handles all training related aspects of the Neural Network'''

    def __init__(self, network, training_data, validation_data,
                 classification=False):
        '''Stores the network, a CostGate, and all the data necessary
        for training and validation. Also stores network type to run
        percent success functions for classification tasks'''

        self.network = network
        self.costgate = CostGate()
        self.training_data = training_data
        self.validation_data = validation_data
        self.classification = classification

        self.DEFAULT_TRAINING_SETTINGS = {
            'batch_size': len(self.training_data['inputs']),
            'epoch_blocks': 1,
            'learning_rate': 0.1  # TODO: estmiate appropriate learning rate
        }

        self.DEFAULT_HALT_SETTINGS = {
            'max_epochs': 100000,
            'max_time': 300,
            'min_validation_cost': 0,
            'min_classification_percent': 100,
            'max_stall_blocks': inf
        }

    def train(self, training_settings={}, halt_settings={}):
        '''Trains the neural network with the specified conditions,
        and then reports on the results of training'''

        self.training_settings = set_defaults(training_settings,
                                              self.DEFAULT_TRAINING_SETTINGS)
        self.halt_settings = set_defaults(halt_settings,
                                          self.DEFAULT_HALT_SETTINGS)

        self.epochs = 0
        self.best_validation_cost = inf

        self.start = time()

        while self.epochs < self.halt_settings['max_epochs']:
            for j in range(self.training_settings['epoch_blocks']):
                self.trainEpoch(self.training_settings['batch_size'],
                                self.training_settings['learning_rate'])
                self.epochs += 1

            self.set_diagnostics()
            self.print_diagnostics('Epoch ' + str(self.epochs) + ': ')

            if self.time >= self.halt_settings['max_time']:
                self.exitMessage('Maximum training time acheived')
                return

            if (self.validation_cost <=
                    self.halt_settings['min_validation_cost']):

                self.exitMessage('Validation cost threshold acheived')
                return

            if self.percent > self.halt_settings['min_classification_percent']:
                self.exitMessage('Classification threshold acheived')
                return

            if self.stall_blocks == self.halt_settings['max_stall_blocks']:
                self.exitMessage('Validation training stalled')
                return

        self.exitMessage('Maximum epoch limit reached')

    def trainEpoch(self, batch_size, learning_rate):
        '''Performs stochastic gradient descent with the specified batch size.
        Shuffles the data, and trains the network on peices of the data'''

        p = np.random.permutation(len(self.training_data['inputs']))
        shuffled_inputs = self.training_data['inputs'][p]
        shuffled_labels = self.training_data['labels'][p]

        while len(shuffled_inputs) > 0:
            inputs = shuffled_inputs[:batch_size]
            labels = shuffled_labels[:batch_size]

            shuffled_labels = shuffled_labels[batch_size:]
            shuffled_inputs = shuffled_inputs[batch_size:]

            output = self.network.forward(inputs)
            self.costgate.set_gradients(output, labels, learning_rate)
            self.network.backward()

    def set_diagnostics(self):
        '''Stores statistics related to the performance of the network'''

        self.network.forward(self.validation_data['inputs'])
        self.validation_cost = np.sum(self.costgate.computeCost(
            self.network.output, self.validation_data['labels']))

        self.time = time() - self.start

        if self.classification:
            self.percent = self.costgate.classification_percent(
                self.network.output, self.validation_data['labels'])

        else:
            self.percent = 0

        if self.validation_cost < self.best_validation_cost:
            self.best_validation_params = deepcopy(
                self.network.give_parameters())

            self.best_validation_cost = self.validation_cost

        if self.validation_cost > self.best_validation_cost:
            self.stall_blocks += 1

        else:
            self.stall_blocks = 0

    def print_diagnostics(self, message):
        '''Prints information regarding the performance of the network'''

        print(message, end='')
        print('Validation Cost: ' +
              str(round(self.validation_cost, 5)), end=' ')

        if self.classification:
            print('Validation Percent: ' +
                  str(round(self.percent, 3)) + '%', end='')

        print('')

    def exitMessage(self, message):
        '''Prints information about the training procedure upon completion'''

        print('\nTraining Complete:', message, '\n')

        print(str(self.epochs) + ' epochs/' +
              str(round(self.time, 3)) + ' seconds\n')

        print('Learning Rate: ' + str(self.training_settings['learning_rate']))
        print('Batch Size: ' + str(self.training_settings['batch_size']))
        print('Epoch Blocks: ' + str(self.training_settings['epoch_blocks']))

        self.network.set_parameters(self.best_validation_params)
        self.set_diagnostics()
        self.print_diagnostics('Best Parameters: ')


class Preprocessor:
    '''Transforms data for optimal training convergance.
    Currently only transforming data between -1 and 1
    with z-score normilization followed by scaling'''

    def __init__(self, input_stats={}, label_stats={}, with_bias=False):
        '''Stores all information required to transform inputs
        and labels according to z-score normalization'''

        # These defaults result in no transformation
        self.DEFAULT_STATS = {
            'mean': 0,
            'std': 1,
            'scale_factor': 1
        }

        self.input_stats = set_defaults(input_stats, self.DEFAULT_STATS)

        self.label_stats = set_defaults(label_stats, self.DEFAULT_STATS)
        self.with_bias = with_bias

    @classmethod
    def from_data(cls, data_sample, with_bias=False):
        '''Calculates all parameters necessary
        for Preprocessor automatically'''

        data_sample = deepcopy(data_sample)

        stats = {'inputs': {}, 'labels': {}}

        for key, value in data_sample.items():
            stats[key] = {
                'mean': np.mean(value, axis=0),
                'std': np.std(value, axis=0)
            }

            normalized = cls.normalize(None, value, stats[key]['mean'],
                                       stats[key]['std'])

            stats[key]['scale_factor'] = np.max(np.abs(normalized))

        return cls(stats['inputs'], stats['labels'], with_bias)

    def transform_inputs(self, inputs):
        '''Applys z-score normalization and scaling to inputs
        according to the stats stored by the preprocessor'''

        inputs = self.normalize(inputs, self.input_stats['mean'],
                                self.input_stats['std'])

        inputs = self.scale(inputs, self.input_stats['scale_factor'])

        if self.with_bias:
            inputs = self.append_bias(inputs)

        return inputs

    def transform_labels(self, labels):
        '''Applys z-score normalization and scaling as well,
        but for labels'''

        labels = self.normalize(labels, self.label_stats['mean'],
                                self.label_stats['std'])

        labels = self.scale(labels, self.label_stats['scale_factor'])

        return labels

    def transform_data(self, data):
        '''Convenience function that transforms both inputs and labels'''

        data['inputs'] = self.transform_inputs(data['inputs'])
        data['labels'] = self.transform_labels(data['labels'])

        return data

    def interpret_labels(self, labels):
        '''Reverses the procedure of transforming a label, to allow
        the output of network to be converted to its original meaning'''

        labels = labels/(self.label_stats['scale_factor'])
        labels = (labels * self.label_stats['std']) + self.label_stats['mean']

        return labels

    def normalize(self, data, mean, std):
        '''Normalizes data according to a mean and standard deviation
        Currently just warns users who have features that are call the same'''

        if (std == 0).any():
            zeros = np.count_nonzero(std == 0)
            print('WARNING: ' + str(zeros) + ' features have' +
                  'a standard deviation of 0. Please remove those features.')
            std = [1 if num == 0 else num for num in std]

        return (data - mean)/std

    def scale(self, data, scale_factor):
        '''Scales data so that is falls between -1 and 1,
        according to a scale factor'''

        if (scale_factor == 0).any():
            scale_factor = [1 if num == 0 else num for num in scale_factor]

        return data/scale_factor

    def append_bias(self, data):
        '''Appends a bias array to the data, to help with
        training in some instances'''

        bias_array = np.array([[1] for i in range(data.shape[0])])

        return np.concatenate((data, bias_array), axis=1)


def set_defaults(dictionary, defaults):
    '''Dictionaries are used to pass in arguments for classes for
    expandibility. This function manages setting default settings
    from a given dictionary passed in'''

    defaults_copy = defaults.copy()

    defaults_copy.update(dictionary)

    return defaults_copy


def save_network(network, name):
    '''Basic function that saves a network's parameters
    and activation function as a numpy array'''

    parameters = network.give_parameters()
    activation = network.activation

    array = np.array([np.array([activation]), parameters])

    np.save(name, array)

    print('Network saved as ' + name)


def load_network(name):
    '''Unloads data from a numpy array created using the
    save_network function to recreate the neural network'''

    array = np.load(name)

    activation = array[0][0]
    parameters = array[1]

    network = NeuralNetwork.from_parameters(parameters, activation)

    return network


def save_preprocessor(preprocessor, name):
    '''Basic function that saves preprocessor stats
    in a saved numpy array'''

    input_stats = preprocessor.input_stats
    label_stats = preprocessor.label_stats
    with_bias = preprocessor.with_bias

    array = np.array([input_stats, label_stats, with_bias])

    np.save(name, array)

    print('Preprocessor saved as ' + name)


def load_preprocessor(name):
    '''Unloads preprocessor data from a numpy array
    saved using the save_preprocessor function'''

    array = np.load(name)

    input_stats = array[0]
    label_stats = array[1]
    with_bias = array[2]

    preprocessor = Preprocessor(input_stats, label_stats, with_bias)

    return preprocessor


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    data = {'inputs': x, 'labels': y}

    P = Preprocessor.from_data(data, with_bias=True)
    NN = NeuralNetwork.new([3, 3, 1], 'tanh')

    data = P.transform_data(data)

    trainer = Trainer(NN, data, data)

    trainer.train({'learning_rate': 0.6, 'epoch_blocks': 100},
                  {'min_validation_cost': 0.0001, 'max_epochs': 100000})

    output = NN.forward(data['inputs'])

    print(P.interpret_labels(output.values))

    save_network(NN, 'XOR')
    save_preprocessor(P, 'XORP')
