##### Animation routine #####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os, sys


# MPEG codec path

ff_path = os.path.join('C:/', 'Program Files (x86)/ffmpeg/bin/', 'ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = ff_path
if ff_path not in sys.path: sys.path.append(ff_path)
plt.style.use('seaborn-pastel')

def u_animation(x , u_anim):

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

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(u_anim), interval=1.2, blit=True) 

    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])

    anim.save('final.mp4', writer=mpeg_writer)

    return anim