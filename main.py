#Cyrus Burt |JAN 2019| MIT LICENSE

import perceptron
import pandas as pd
import matplotlib.pyplot as plt
import random

model = perceptron.perceptron(0.1) #initialize a new perceptron with a learning rate of 0.1

data = pd.read_csv("iris.data") #read iris csv

species = data.iloc[0:, 4] #initialize a new array of the iris species, these will get encoded

encoded_species = []

for i in species:
    if i == "Iris-setosa":
        encoded_species.append(1)
    if i == "Iris-versicolor":
        encoded_species.append(0)
    if i == "Iris-virginica":
        encoded_species.append(-1)


plt.plot(data.iloc[0:49, 2], data.iloc[0:49, 0], linestyle="None", marker=".")
plt.plot(data.iloc[49:100, 2], data.iloc[49:100, 0], linestyle="None", marker=".")
plt.plot(data.iloc[100:, 2], data.iloc[100:, 0], linestyle="None", marker=".")

# plt.plot(data.iloc[0:49, 2], encoded_species[0:49], linestyle="None", marker=".")
# plt.plot(data.iloc[49:100, 2], encoded_species[49:100], linestyle="None", marker=".")
# plt.plot(data.iloc[100:, 2], encoded_species[100:], linestyle="None", marker=".")

plt.show()