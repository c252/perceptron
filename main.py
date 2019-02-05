#Cyrus Burt |JAN 2019| MIT LICENSE

"""
main.py
uses my simple perceptron to classify irises

>>> model.pred([1.4,0.2])
1.0
>>> model.pred([4.5,1.6])
-1.0
"""
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
petal_width = list(data.iloc[0:, 3]) 

encoded_species = []

for i in species:
    if i == "Iris-setosa":
        encoded_species.append(1.0)
    if i == "Iris-versicolor":
        encoded_species.append(-1.0)

for _ in range(100):
    for i in range(len(encoded_species)):
        model.train([petal_length[i], petal_width[i]], encoded_species[i])

def classify_iris(petal_l, petal_w):
    if model.pred([petal_l, petal_w]) == 1:
        return "Iris-setosa"

    else:
        return "Iris-versicolor"

if __name__ == "__main__":
    import doctest
    doctest.testmod()
