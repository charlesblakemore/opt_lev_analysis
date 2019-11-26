import sys
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torsion_noise as tn

#np.random.seed(123456)

### Constants
e = 1.602e-19          #  C
p0 = 100 * e * (1e-6)  #  C * m
rhobead = 2000
rbead = 2.4e-6
mbead = (4. / 3.) * np.pi * rbead**3 * rhobead
Ibead = (2. / 5.) * mbead * rbead**2 

### Initial Conditions
theta0 = -0.2   # rad
phi0 = 0            # rad
p0x = p0 * np.sin(theta0) * np.cos(phi0)
p0y = p0 * np.sin(theta0) * np.sin(phi0)
p0z = p0 * np.cos(theta0)

wx = 0.0            # rad/s
wy = 0.0  
wz = 0.0 

xi_init = np.array([p0x, p0y, p0z, wx, wy, wz])

### Integration parameters
dt = 1.0e-3
ti = 0
tf = 200

Fsamp = 1.0 / dt

### Drive parameters
maxvoltage = 10.0
fieldmag = maxvoltage / 4e-3
#fx = 100
#fy = 100
fz = 100

tt = np.arange(ti, tf + dt, dt)
Nsamp = len( tt )

tarrX = tn.torqueNoise(Nsamp, Fsamp)
tarrY = tn.torqueNoise(Nsamp, Fsamp)
tarrZ = tn.torqueNoise(Nsamp, Fsamp)


torque_noise_funcX = interp.interp1d(tt[:len(tarrX)], tarrX, bounds_error=False, \
                                        fill_value='extrapolate')
tarrX = torque_noise_funcX(tt)

torque_noise_funcY = interp.interp1d(tt[:len(tarrY)], tarrY, bounds_error=False, \
                                        fill_value='extrapolate')
tarrY = torque_noise_funcY(tt)

torque_noise_funcZ = interp.interp1d(tt[:len(tarrZ)], tarrZ, bounds_error=False, \
                                        fill_value='extrapolate')
tarrZ = torque_noise_funcZ(tt)


beta = tn.beta()


def Efield(t):
    if t < 100:
        Ex = 0
        Ey = 0
        Ez = 0
    else:
        Ex = 0.0
        Ey = fieldmag * 10
        Ez = 0.0
        #Ez = fieldmag * np.sin(2 * np.pi * fz * t)     # Volt / meter
        #Ez = signal.chirp(t-100, 1, 100, 100)
    return Ex, Ey, Ez


def system(xi, t, tind):

    Ex, Ey, Ez = Efield(t)

    px = xi[0]
    py = xi[1]
    pz = xi[2]

    wx = xi[3]
    wy = xi[4]
    wz = xi[5]

    dpx = py * wz - pz * wy
    dpy = pz * wx - px * wz
    dpz = px * wy - py * wx

    torque = [py * Ez - pz * Ey,  pz * Ex - px * Ez,  px * Ey - py * Ex]
    torque[0] += tarrX[tind] - wx * beta
    torque[1] += tarrY[tind] - wy * beta
    torque[2] += tarrZ[tind] - wz * beta

    return np.array([-dpx, -dpy, -dpz, torque[0] / (Ibead), \
                        torque[1] / (Ibead), torque[2] / (Ibead)])


def rk4(xi_old, t, tind, delt, system, spherical=False):
    k1 = delt * system(xi_old, t, tind)
    k2 = delt * system(xi_old + k1 / 2, t + (delt / 2), tind)
    k3 = delt * system(xi_old + k2 / 2, t + (delt / 2), tind)
    k4 = delt * system(xi_old + k3, t + delt, tind)

    xi_new = xi_old + (1. / 6.) * (k1 + 2*k2 + 2*k3 + k4)

    ptot = np.sqrt(xi_new[0]**2 + xi_new[1]**2 + xi_new[2]**2)
    for ind in [0,1,2]:
        xi_new[ind] *= p0 / ptot
    return xi_new


def stepper(xi_0, ti, tf, delt, system, method, plot=False, spherical=False):
    tt = np.arange(ti, tf + delt, delt)

    # Initialize list of 'points' which will contain the solution to our
    # ODE at each time in our discrete-time array
    points = []
    energy_vec = []
    energy_vec2 = []
    i = 0
    xi_old = xi_0

    ticker = 0
    for tind, t in enumerate(tt):

        #plt.cla()
        #ax.clear()
        # Using the 4th-order Runge-Kutta method, we evaluate the solution
        # iteratively for each time t in our discrete-time array
        xi_new = method(xi_old, t, tind, delt, system, spherical=spherical)

        points.append(xi_new)

        Ex, Ey, Ez = Efield(t)
        # Compute energy as:  (1/2) I omega^2 - p (dot) E
        energy = 0.5 * Ibead * (xi_new[3]**2 + xi_new[4]**2 + xi_new[5]**2) - \
                    (Ex * xi_new[0] + Ey * xi_new[1] + Ez * xi_new[2])

        #energy2 = 0.5 * Ibead * (xi_new[3]**2 + xi_new[4]**2 + xi_new[5]**2)
        energy_vec.append(energy)

        xi_old = xi_new

        if (t / tf) > (i * 0.01):
            print(i, end=' ')
            sys.stdout.flush()
            i += 1
        ticker += 1

    return tt, np.array(points), np.array(energy_vec)





time, points, energy_vec = stepper(xi_init, ti, tf, dt, system, rk4, plot=False, spherical=True)
print(points.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.1*p0, 1.1*p0])
ax.set_ylim([-1.1*p0, 1.1*p0])
ax.set_zlim([-1.1*p0, 1.1*p0])
dipole = np.zeros(6)
xcomp = points[0,0]
ycomp = points[0,1]
zcomp = points[0,2]
vec1 = np.zeros(6)
vec1[3:] = np.array([xcomp, ycomp, zcomp])
vec2 = np.array([0, -1.1*p0, -1.1*p0, xcomp, 0, 0])
vec3 = np.array([1.1*p0, 0, -1.1*p0, 0, ycomp, 0])
vec4 = np.array([1.1*p0, 1.1*p0, 0, 0, 0, zcomp])
arr = np.stack((vec1, vec2, vec3, vec4))
artist = ax.quiver(*np.split(arr, 6, axis=-1))

phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
x = p0*np.sin(phi)*np.cos(theta)
y = p0*np.sin(phi)*np.sin(theta)
z = p0*np.cos(phi)

ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

ax.set_aspect("equal")

onframe = np.argmin( np.abs(tt-100) )

def update(i):
    global artist, ax
    xcomp = points[i,0]
    ycomp = points[i,1]
    zcomp = points[i,2]
    vec1 = np.zeros(6)
    vec1[3:] = np.array([xcomp, ycomp, zcomp])
    vec2 = np.array([0, -1.1*p0, -1.1*p0, xcomp, 0, 0])
    vec3 = np.array([1.1*p0, 0, -1.1*p0, 0, ycomp, 0])
    vec4 = np.array([1.1*p0, 1.1*p0, 0, 0, 0, zcomp])
    # x field arrow
    #vec5 = np.array([-0.5*p0, 1.1*p0, 1.1*p0, p0, 0, 0])
    # y field arrow
    vec5 = np.array([-1.1*p0, -0.5*p0, 1.1*p0, 0, p0, 0])
    # z field arrow
    #vec5 = np.array([-1.1*p0, -1.1*p0, -0.5*p0, 0, 0, p0])
    if i > onframe:
        arr = np.stack((vec1, vec2, vec3, vec4, vec5))
        colors = ['C0', 'C0', 'C0', 'C0', 'C1']
    else: 
        arr = np.stack((vec1, vec2, vec3, vec4))
        colors = ['C0', 'C0', 'C0', 'C0']
    artist.remove()
    artist = ax.quiver(*np.split(arr, 6, axis=-1), color=colors)
    return artist, ax

if __name__ == '__main__':
    initframes = np.arange(len(points[:,0]))
    numframes = 200.
    firstframe = np.argmin( np.abs(tt-95) )
    lastframe = np.argmin( np.abs(tt-105) )
    stride = int((lastframe-firstframe) / numframes)
    cframes = initframes[firstframe:lastframe:stride]
    cframes = cframes.astype(np.int64)
    print(len(cframes))
    print("Building gif...")
    #cframes = np.array([1,10000])
    anim = FuncAnimation(fig, update, frames=cframes, interval=200)

    anim.save('./gif_dipole_pinning_yaxis_strong.gif', dpi=100, writer='imagemagick')

    #plt.show()

