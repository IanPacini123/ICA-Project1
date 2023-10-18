import numpy as np
import matplotlib.pyplot as plt

ds = np.loadtxt("DataSets/serieTemporal\sunspot/sunspots.dat", delimiter="\t")

timeFoo = ds[:, 0]
valueFoo = ds[:, -1]

time = timeFoo[:100]
value = valueFoo[:100]

plt.plot(time,value)
plt.show()