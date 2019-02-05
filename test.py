#Cyrus Burt |JAN 2019| MIT LICENSE

#Very simple test to see if the perceptron works
#Tests if the perceptron can classify X OR Y

import perceptron
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

model = perceptron.perceptron(0.5)

data = []

for i in range(100):
    data.append([random.randint(0, 1), random.randint(0, 1)])

results = []
for i in data:
    results.append(i[0] | i[1])

for i in range(len(data)):
    model.train(data[i],results[i])

plt.plot(data, results, marker=".", linestyle="None")
plt.show()

print(model.pred([0,0]))#should be 0
print(model.pred([1,0]))#should be 1
print(model.pred([0,1]))#should be 1
print(model.pred([1,1]))#should be 1
