#Cyrus Burt |JAN 2019| MIT LICENSE

#Very simple test to see if the perceptron works
#Tests if the perceptron can classify X OR Y

import perceptron
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

model = perceptron.perceptron(0.7)

data = pd.read_csv("cars.data")

door_num = list(data.iloc[0:, 2])

safety = list(data.iloc[0:, 5])

#data = shuffle(data)

plt.plot(door_num, safety, linestyle="None", marker=".")
plt.show()