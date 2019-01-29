#Cyrus Burt |JAN 2019| MIT LICENSE

import perceptron
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

model = perceptron.perceptron(0.5) #initialize a new perceptron with a learning rate of 0.1

data = pd.read_csv("iris.data") #read iris csv

data = shuffle(data)

species = list(data.iloc[0:, 4]) #initialize a new array of the iris species, these will get encoded

petal_length = list(data.iloc[0:, 2]) 

encoded_species = []

for i in species:
    if i == "Iris-setosa":
        encoded_species.append(1.0)
    if i == "Iris-versicolor":
        encoded_species.append(-1.0)

for i in range(len(species)):
    model.train(petal_length[i], encoded_species[i])

print(model.pred(1.4))