import numpy as np
import matplotlib.pyplot as plt

ds = np.loadtxt("DataSets/serieTemporal\sunspot/sunspots.dat", delimiter="\t")

timeF = ds[:, 0]
valueF = ds[:, -1]

time = timeF[:100]
value = valueF[:100]

plt.plot(time,value)
plt.show()