import random
 #takes 1d vector for inputs and learning rate
lr = 0.01
weights = [0.0]*2 #intializes the weights for the perceptron, there will always be two 

data = []
result = []
for i in range(50):
    data.append([random.randint(-1,1), random.randint(-1,1)])

for i in data:
    result.append(i[0] & i[1])

for i in range(len(weights)): #give the weights of the inputs a random value from -1 to 1
  weights[i] = random.uniform(-1,1)

def activation(val): # simple linear activation function
    if val > 0.0:
      return 1.0
    else:
      return -1.0

def pred(inputs):
    sum = 0.0
    for i in range(len(weights)): #caculate the weighted sum (dot product)
      sum += inputs[i] * weights[i]

    return activation(sum)

def train(inputs, ans):
    guess = pred(inputs)
    error = ans - guess
    print(weights)

    for i in range(len(weights)):
      weights[i] += lr * error * inputs[i]