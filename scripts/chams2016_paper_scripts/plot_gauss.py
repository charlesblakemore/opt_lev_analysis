import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

xx = np.linspace(-5, 5, 1e3)

plt.plot(xx, np.exp( -xx**2 ) )
plt.ylim([-0.1, 1.1])
plt.savefig("gauss.pdf")

plt.show()
