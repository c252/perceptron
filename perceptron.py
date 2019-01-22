import random

#like all perceptrons this can only classify 2d linearly separable data

class perceptron(object):
  
  def __init__(self, inputs, lr): #takes 1d vector for inputs and learning rate
    self.lr = lr
    self.inputs = inputs

    self.weights = [0]*2 #intializes the weights for the perceptron, there will always be two 

    for i in range(len(self.weights)): #give the weights of the inputs a random value from -1 to 1
      self.weights[i] = random.uniform(-1,1)

  def activation(self, val): # simple linear activation function
    if val > 0:
      return 1
    else:
      return -1

  def pred(self,inputs):
    sum = 0
    for i in range(len(self.weights)): #caculate the weighted sum
      sum += self.inputs[i] * self.weights[i]

    return self.activation(sum)

  def train(self, inputs, ans):
    guess = self.pred(inputs)
    error = ans - guess

    for i in range(len(self.weights)):
      self.weights[i] += self.lr * error * inputs[i]