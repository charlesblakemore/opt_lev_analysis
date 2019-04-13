import numpy as np
import matplotlib.pyplot as plt
import matplotlib


amod = .1
dc = 0.6

Qs = np.linspace(0, 2.*np.pi, 1000)
Es = amod*np.cos(4.*Qs) + dc

matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(dpi = 200, figsize = (4, 2))
ax.plot(Qs/np.pi, Es, linewidth = 5)
ax.axhline(y = dc, linestyle = "--", linewidth = 4, color = 'k', alpha = 0.7, label = r"0.6 $E_{0}$")
ax.set_ylim([0, 1])
ax.set_xlim([0, 2])
ax.set_xlabel(r"$\phi$ [$\pi$]")
ax.set_ylabel(r"E [$E_{0}$]")
ax.legend()
plt.subplots_adjust(top = 0.96, bottom = 0.3, left = 0.2, right = 0.99)
plt.show()


