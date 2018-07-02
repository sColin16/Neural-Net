from NeuralNet import NeuralNetwork, Trainer, Preprocessor
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

x = np.array(digits.data[:100])
y = np.array([[int(i == digit) for i in range(10)] for digit in digits.target[:100]])

validation_x = np.array(digits.data[100:120])
validation_y = np.array([[int(i == digit) for i in range(10)] for digit in digits.target[100:120]])

training_data = {'inputs': x, 'labels': y}
validation_data = {'inputs': validation_x, 'labels': validation_y}

P = Preprocessor.from_data(training_data)
NN = NeuralNetwork.new([64, 10, 10], 'tanh')

training_data = P.transform_data(training_data)
validation_data = P.transform_data(validation_data)

trainer = Trainer(NN, training_data, validation_data, classification = True)

trainer.train({'learning_rate': 0.01, 'epoch_blocks': 10, 'batch_size': 100},
              {'max_epochs': 2000, 'max_stall_blocks': 10})