import perceptron

inputs = [1.0,0.75]

neuron = perceptron.perceptron(0.1)

for i in range(4):
    print(neuron.pred(inputs))
    print(neuron.weights)
    neuron.train(inputs, 1.0)