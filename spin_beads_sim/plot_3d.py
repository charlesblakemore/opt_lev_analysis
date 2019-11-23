import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

filepath = '/home/dmartin/Desktop/simulations/libration_3/slow_rot_field/16.5e3Vm_data.npy'
#filepath = '/home/dmartin/Desktop/simulations/libration_2/33e3Vm_5k_data.npy'


frames = 10000
interval = 1

start = 0
stop = 1000


def sine(t):
    
    X = np.cos(t*0.1)
    Y = np.sin(t*0.1)
    
    Q.set_segments([[[0,0,0],[X, Y, 0.]]])


#def plot_3d(t,x,y,z):

 #   X = x[t]
 #   Y = y[t]
 #   Z = z[t]

  #  Q.set_segments([[[0,0,0],[X,Y,Z]]])

def plotter(t,frames, interval,arr, arr1 = []):

    if arr1:
        new_arr = [arr[0],arr[1],arr[2],arr1[0],arr1[1],arr1[2]]
        plt.plot(t, new_arr[3])
        plt.plot(t,new_arr[4])
        plt.plot(t,new_arr[5])
        plt.show()
    
    else:
        new_arr = arr

    def plot_3d(t,x,y,z,s=0,u=0,v=0):
        
        X = x[t]
        Y = y[t]
        Z = z[t]
        Q.set_segments([[[0,0,0],[X,Y,Z]]])
       
        if arr1:
            S = s[t]
            U = u[t]
            V = v[t]
    
            P.set_segments([[[0,0,0],[S,U,V]]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if not arr1:
        Q = ax.quiver(0,0,0,new_arr[0],new_arr[1],new_arr[2])
    
    else:
        P = ax.quiver(0,0,0,new_arr[3]/np.amax(new_arr[3]),\
            new_arr[4]/np.amax(new_arr[4]),new_arr[5]/np.amax(new_arr[5]))

        Q = ax.quiver(0,0,0,new_arr[0]/np.amax(new_arr[0]),\
            new_arr[1]/np.amax(new_arr[1]),new_arr[2]/np.amax(new_arr[2]))
        
    if not arr1:
        ax.set_xlim(-np.amax(arr[0]),np.amax(arr[0]))
        ax.set_ylim(-np.amax(arr[1]),np.amax(arr[1]))
        ax.set_zlim(-np.amax(arr[2]),np.amax(arr[2]))
    else:
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
    

    
    
    ani = animation.FuncAnimation(fig, plot_3d, frames = frames, interval = interval, blit=False, repeat=False, fargs=new_arr)
    plt.show()


data = np.load(filepath)

time = data[:,0]

time = time[start:stop]

efieldx = data[:,-3]
efieldy = data[:,-2]
efieldz = data[:,-1]

px = data[:,-6]
py = data[:,-5]
pz = data[:,-4]

px = px[start:stop]
py = py[start:stop]
pz = pz[start:stop]

p = [px,py,pz]

efieldx = data[:,-3]
efieldy = data[:,-2]
efieldz = data[:,-1]

efieldx = efieldx[start:stop]
efieldy = efieldy[start:stop]
efieldz = efieldz[start:stop]

efield = [efieldx, efieldy, efieldz]


px = 0.2*np.cos(2 * np.pi * 200 * time + 0.8* 0.5*np.pi*np.sin(2*np.pi * 50*time ))
py = 0.2*np.sin(2 * np.pi * 200 * time + 0.8* 0.5*np.pi*np.sin(2*np.pi * 50*time ))
pz = np.zeros_like(px)

fft = np.fft.rfft(px)
freqs = np.fft.rfftfreq(len(px), time[1]-time[0])

plt.plot(freqs, np.abs(fft))
plt.show()

plt.plot(time, px)
plt.plot(time, py)
plt.show()

p = [px,py,0*pz]
plotter(time, frames, interval, p)
