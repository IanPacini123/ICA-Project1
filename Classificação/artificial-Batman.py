import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataSet = pd.read_csv("DataSets/artificial/batman/data.csv")

x1 = dataSet["X1"]
x2 = dataSet["X2"]
Y = dataSet["Y"]

plt.scatter(x1, x2, c=Y, s = 5)
plt.show()