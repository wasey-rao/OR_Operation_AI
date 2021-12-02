import random

import numpy as np


class Neuron:

    def __init__(self, no_of_inputs):
        self.bias = 0.2
        self.weights = [0.65, 0.30]
        self.learning_rate = 0.1

    def predict(self, input1, input2):
        summation = input1 * self.weights[0] + input2 * self.weights[1] + self.bias
        if summation >= 0:
            return 1
        else:
            return 0

    def train(self, training_ex, actual_label):
        predected_label = self.predict(training_ex[0], training_ex[1])
        print("predicted_label is:", predected_label, "bias is: ", self.bias)
        if predected_label != actual_label:
            error = actual_label - predected_label
            delta_w = self.learning_rate * error
            self.weights[0] += delta_w
            self.weights[1] += delta_w
            self.bias += delta_w
            print("error is: ", error, "new bias is: ", self.bias, "predicted label is: ", predected_label)
            self.train(training_ex, actual_label)
        return actual_label
inputs = []
Y = []
for n in range(99):
    inputs.append([random.randint(0, 100), random.randint(0, 100)])
    number = inputs[n][0] - inputs[n][1]
    if number < 0:
        number = number * -1
    if number % 2 == 0:
        Y.append(0)
    else:
        Y.append(1)


epoch = 20

training_input = np.array(inputs)

neuron = Neuron(2)
for e in range(epoch):
    for j in range(len(training_input)):
        decision = neuron.train(training_input[j], Y[j])
        print("The decision is: ", decision,"\n\n")
