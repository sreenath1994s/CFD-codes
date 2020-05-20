##### Animation Subroutines #####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os, sys
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm



# MPEG codec path

ff_path = os.path.join('C:/', 'Program Files (x86)/ffmpeg/bin/', 'ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = ff_path
if ff_path not in sys.path: sys.path.append(ff_path)
plt.style.use('seaborn-pastel')

# 2D Animation

def animation_2d(x , u_anim):

    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(xlim=(0, 7), ylim=(0, 10)) 
    plt.xlabel("x [m]",   fontname = "serif", fontsize = 10)
    plt.ylabel("u [m/s]", fontname = "serif", fontsize = 10)
    line1, = ax.plot([], [], lw=2)  
    line2, = ax.plot(x, u_anim[0][:].flatten(), lw=2)   # initial condition

    def init():
        
	    line1.set_data([], []) 
	    return line1,

    def update(t):
        ax.set_title(str("t_step = " + str(t)), fontname = "serif", fontsize = 12)
        u_out = u_anim[t][:].flatten()
        x_out = x.flatten()
        line1.set_data(x_out, u_out)
        return line1,

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(u_anim), blit=True) 

    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])

    anim.save('final.mp4', writer=mpeg_writer)

    return 

# 3D Animation

def animation_3d(x, y, z, dt):

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    plt.xlabel("x [m]",   fontname = "serif", fontsize = 10)
    plt.ylabel("y [m]", fontname = "serif", fontsize = 10)
    Y, X = np.meshgrid(y, x) 
    surf2 = ax.plot_surface(X, Y, z[0][:][:], cmap=cm.viridis, linewidth=0, antialiased=False)  
    plt.axis('off')

    def update(t):
        ax.clear()
        ax.set_title(str("t_step = " + str(t)), fontname = "serif", fontsize = 12)
        surf2 = ax.plot_surface(X, Y, z[t][:][:], cmap=cm.viridis, linewidth=0, antialiased=False) 
        plt.axis('off')
        return surf2,

    anim = animation.FuncAnimation(fig, update, frames=len(z), blit=True) 

    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])

    anim.save('final.mp4', writer=mpeg_writer)

    return

def surface_2d(x, y, z, name):

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    Y, X = np.meshgrid(y, x)                            
    surf = ax.plot_surface(X, Y, z, cmap=cm.viridis, linewidth=0, antialiased=True)
    plt.savefig(str(str(name) + ".png"))
    plt.show()
    ax.clear()

    return

