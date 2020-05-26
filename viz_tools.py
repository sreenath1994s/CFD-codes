##### Animation Subroutines #####

import numpy as np
import matplotlib.pyplot as plt
import os, sys

from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import interpolate
from vispy import app, scene
from vispy.gloo.util import _screenshot

# MPEG codec path

ff_path = os.path.join('C:/', 'Program Files (x86)/ffmpeg/bin/', 'ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = ff_path
if ff_path not in sys.path: sys.path.append(ff_path)
plt.style.use('seaborn-pastel')


def surface_2d(x, y, z, name):

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    ax.set_title(str(name), fontname = "serif", fontsize = 12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlim(0, 1)
    
    Y, X = np.meshgrid(y, x)   
    
    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)

    tray1[0,:] = 0.5 ; tray1[:,0] = 0.5 ; tray1[-1,:] = 0.5 ; tray1[:,-1] = 0.5
    tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ;  tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1]

    surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
    surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
    surf3 = ax.plot_surface(X, Y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True)
    
    plt.savefig(str(str(name) + ".png"))
    plt.show()
    ax.clear()

    return

def surface_2d_gpu(x, y, z, name):

    canvas = scene.SceneCanvas(keys='interactive')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.center = (1,1,1)
    view.camera.fov = 50
    view.camera.mode = 'perspective'
    view.camera.distance = 4
    view.camera.azimuth = 30
    view.camera.elevation = 10

    Y, X = np.meshgrid(y, x)   
    
    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)
    
    tray1[0,:] = 1 ; tray1[:,0] = 1 ; tray1[-1,:] = 1 ; tray1[:,-1] = 1
    tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ;  tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1]

    surface  = scene.visuals.SurfacePlot(x, y, z, shading='smooth', color='#289fd2')
    surface1 = scene.visuals.SurfacePlot(x, y, tray1, shading='smooth', color=(0.5,0.5,0.5,0.1))
    surface2 = scene.visuals.SurfacePlot(x, y, tray2, shading='smooth', color='#289fd2')

    view.add(surface)
    view.add(surface2)
    view.add(surface1)
    canvas.show(run=True)

    return 

def animation_3D(cx, cy, Uanim, tanim, cur_itr, t_stop):

    print("\nStarting visualization")

    fps = 23.976
    f_dt = 1/fps

    f_t = np.linspace(0,t_stop,int(t_stop/f_dt)+1)

    z = np.zeros((len(f_t), len(cx), len(cy)))
    
    for i in range (0, len(cx)):
        for j in range (0, len(cy)):
            coeff = interpolate.splrep(tanim[:cur_itr],Uanim[:cur_itr,i,j])
            z[:,i,j] = interpolate.splev(f_t, coeff)

    print("\nInterpolation done")

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    plt.xlabel("x",   fontname = "serif", fontsize = 10)
    plt.ylabel("y", fontname = "serif", fontsize = 10)
    ax.set_zlim(0, 1)
    Y, X = np.meshgrid(cy, cx) 

    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)

    tray1[0,:] = 0.5 ; tray1[:,0] = 0.5 ; tray1[-1,:] = 0.5 ; tray1[:,-1] = 0.5
    tray2[0,:] = z[0,0,:] ; tray2[:,0] = z[0,:,0] ; tray2[-1,:] = z[0,-1,:] ; tray2[:,-1] = z[0,:,-1]

    surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
    surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
    surf3 = ax.plot_surface(X, Y, z[0], rstride=2, cstride=2, linewidth=0.1, antialiased=True)

    def update(t):
        ax.clear()
        ax.set_zlim(0, 1)

        tray2[0,:] = z[t,0,:] ; tray2[:,0] = z[t,:,0] ; tray2[-1,:] = z[t,-1,:] ; tray2[:,-1] = z[t,:,-1] 
       
        surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
        surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
        surf3 = ax.plot_surface(X, Y, z[t], rstride=2, cstride=2, linewidth=0.1, antialiased=True) 
        return surf1,surf2,surf3

    anim = animation.FuncAnimation(fig, update, frames=len(f_t), blit=True) 

    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])

    anim.save('final.mp4', writer=mpeg_writer)

    print("\nVisualization done")

    return