import perceptron

inputs = [1.0,0.75]

neuron = perceptron.perceptron(5,0.1)

print(neuron.pred(inputs))

print(neuron.weights)