##### Animation Subroutines #####

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import vispy.io as io
import time
import datetime

from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import interpolate
from vispy import app, scene
from vispy.gloo.util import _screenshot
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

time1 = time.time()

# MPEG codec path

ff_path = os.path.join('C:/', 'Program Files (x86)/ffmpeg/bin/', 'ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = ff_path
if ff_path not in sys.path: sys.path.append(ff_path)
plt.style.use('seaborn-pastel')


def surface_3D(x, y, z, name):

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

def surface_3D_gpu(x, y, z, name):

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
    #surface1 = scene.visuals.SurfacePlot(x, y, tray1, shading='smooth', color=(0.5,0.5,0.5,0.1))
    surface2 = scene.visuals.SurfacePlot(x, y, tray2, shading='smooth', color='#289fd2')

    view.add(surface)
    view.add(surface2)
    #view.add(surface1)

    canvas.show(run=True)

    im = _screenshot((0, 0, canvas.size[0], canvas.size[1]))
    io.imsave('vispy_screenshot.png', im)

    return 

def animation_3D_gpu(cx, cy, Uanim, tanim, cur_itr, t_stop):

    print("\nStarting visualization \nInterpolation of variables in progress...")

    canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080))
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.center = (1,1,0)
    view.camera.fov = 50
    view.camera.mode = 'perspective'
    view.camera.distance = 3
    view.camera.azimuth = 30
    view.camera.elevation = 30

    z     = np.zeros((len(cx), len(cy)))

    coeff = np.zeros((len(cx), len(cy)), dtype=object)
    
    for i in range (0, len(cx)):
        for j in range (0, len(cy)):
            coeff[i,j] = interpolate.splrep(tanim[:cur_itr],Uanim[:cur_itr,i,j])
    
    print("\nInterpolation done")

    for i in range (0, len(cx)):
            for j in range (0, len(cy)):
                z[i,j] = interpolate.splev(t_stop, coeff[i,j])

    Y, X = np.meshgrid(cy, cx)   
    
    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)
    
    tray1[0,:] = 1 ; tray1[:,0] = 1 ; tray1[-1,:] = 1 ; tray1[:,-1] = 1
    tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ;  tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1]

    surface  = scene.visuals.SurfacePlot(cx, cy, z, shading='smooth', color='#289fd2')
    surface1 = scene.visuals.SurfacePlot(cx, cy, tray1, shading='smooth', color=(0.5,0.5,0.5,0.1))
    surface2 = scene.visuals.SurfacePlot(cx, cy, tray2, shading='smooth', color='#289fd2')

    view.add(surface)
    view.add(surface2)
    view.add(surface1)

    canvas.show(run=True)

    im = _screenshot((0,0,canvas.size[0],canvas.size[1]))[:,:,:3]
    io.imsave('vispy_screenshot.png', im)

    def update(t):

        for i in range (0, len(cx)):
            for j in range (0, len(cy)):
                z[i,j] = interpolate.splev(t, coeff[i,j])
            
        tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ; tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1] 
        
        surface.set_data(cx, cy, z)
        surface2.set_data(cx, cy, tray2)

        canvas.on_draw(None)

        return _screenshot((0,0,canvas.size[0],canvas.size[1]))[:,:,:3]

    anim = VideoClip(update, duration=t_stop)

    t = time.time()

    anim.write_videofile("3D Vispy.mp4", fps=23.976,  codec = "libx264", bitrate = "24000000")

    print ("Animation with MoviePy : %.02f seconds"%(time.time() - t))

    print("\nVisualization done")

    return 

def animation_3D(cx, cy, Uanim, tanim, cur_itr, t_stop):

    print("\nStarting visualization")

    fps = 23.976
    f_dt = 1/fps

    f_t = np.arange(0, t_stop, f_dt, dtype=None)

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

        global time1

        ax.clear()
        ax.set_zlim(0, 1)

        progress(f_t, t, time1)

        tray2[0,:] = z[t,0,:] ; tray2[:,0] = z[t,:,0] ; tray2[-1,:] = z[t,-1,:] ; tray2[:,-1] = z[t,:,-1] 
       
        surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
        surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
        surf3 = ax.plot_surface(X, Y, z[t], rstride=2, cstride=2, linewidth=0.1, antialiased=True) 

        time1 = time.time()

        return surf1,surf2,surf3

    anim = animation.FuncAnimation(fig, update, frames=len(f_t), blit=True) 

    mpeg_writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])

    anim.save('3D Matplot.mp4', writer=mpeg_writer)

    print("\nVisualization done")

    return

def animation_3D_fast(cx, cy, Uanim, tanim, cur_itr, t_stop):

    print("\nStarting visualization")

    z     = np.zeros((len(cx), len(cy)))

    coeff = np.zeros((len(cx), len(cy)), dtype=object)
    
    for i in range (0, len(cx)):
        for j in range (0, len(cy)):
            coeff[i,j] = interpolate.splrep(tanim[:cur_itr],Uanim[:cur_itr,i,j])
    
    print("\nInterpolation done")

    for i in range (0, len(cx)):
            for j in range (0, len(cy)):
                z[i,j] = interpolate.splev(0, coeff[i,j])

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    plt.xlabel("x",   fontname = "serif", fontsize = 10)
    plt.ylabel("y", fontname = "serif", fontsize = 10)
    ax.set_zlim(0, 1)
    Y, X = np.meshgrid(cy, cx) 

    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)

    tray1[0,:] = 0.5 ; tray1[:,0] = 0.5 ; tray1[-1,:] = 0.5 ; tray1[:,-1] = 0.5
    tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ; tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1]

    surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
    surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
    surf3 = ax.plot_surface(X, Y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True)

    def update(t):

        ax.clear()
        ax.set_zlim(0, 1)

        for i in range (0, len(cx)):
            for j in range (0, len(cy)):
                z[i,j] = interpolate.splev(t, coeff[i,j])
            
        tray2[0,:] = z[0,:] ; tray2[:,0] = z[:,0] ; tray2[-1,:] = z[-1,:] ; tray2[:,-1] = z[:,-1] 
        
        surf1 = ax.plot_surface(X, Y, tray1, rstride=2, cstride=2, linewidth=0, color=(0.5,0.5,0.5,0.1))
        surf2 = ax.plot_surface(X, Y, tray2, rstride=2, cstride=2, linewidth=0, color='#289fd290')
        surf3 = ax.plot_surface(X, Y, z, rstride=2, cstride=2, linewidth=0.1, antialiased=True) 

        return mplfig_to_npimage(fig)

    anim = VideoClip(update, duration=t_stop)

    t = time.time()

    anim.write_videofile("test_.mp4", fps=23.976,  codec = "libx264")

    print ("Animation with MoviePy : %.02f seconds"%(time.time() - t))

    print("\nVisualization done")

    return

def progress(f_t, t, time1):

    t_total = len(f_t)

    ETA     = (time.time()-time1)*(t_total - t + 2)

    if(t>0):
        
        print("Frame ", t," out of ", len(f_t), ". ETA,", str(datetime.timedelta(seconds=ETA)) )

    return


def animation_3D_gpuf(cx, cy, Uanim, tanim, cur_itr, t_stop):

    print("\nStarting visualization \n\nInterpolation of variables in progress...")

    canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080))
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.center = (1,1,0)
    view.camera.fov = 50
    view.camera.mode = 'perspective'
    view.camera.distance = 3
    view.camera.azimuth = 30
    view.camera.elevation = 30

    fps = 23.976
    f_dt = 1/fps

    f_t = np.arange(0, t_stop, f_dt, dtype=None)

    z = np.zeros((len(f_t), len(cx), len(cy)))
    
    for i in range (0, len(cx)):
        for j in range (0, len(cy)):
            coeff = interpolate.splrep(tanim[:cur_itr],Uanim[:cur_itr,i,j])
            z[:,i,j] = interpolate.splev(f_t, coeff)
    
    print("\nInterpolation done")

    Y, X = np.meshgrid(cy, cx)   
    
    tray1  = np.zeros_like(X)
    tray2  = np.zeros_like(X)
    
    tray1[0,:] = 1 ; tray1[:,0] = 1 ; tray1[-1,:] = 1 ; tray1[:,-1] = 1
    tray2[0,:] = z[0,0,:] ; tray2[:,0] = z[0,:,0] ;  tray2[-1,:] = z[0,-1,:] ; tray2[:,-1] = z[0,:,-1]

    surface  = scene.visuals.SurfacePlot(cx, cy, z[0], shading='smooth', color='#289fd2')
    surface1 = scene.visuals.SurfacePlot(cx, cy, tray1, shading='smooth', color=(0.5,0.5,0.5,0.1))
    surface2 = scene.visuals.SurfacePlot(cx, cy, tray2, shading='smooth', color='#289fd2')

    view.add(surface)
    view.add(surface2)
    view.add(surface1)

    canvas.show(run=True)

    im = _screenshot((0,0,canvas.size[0],canvas.size[1]))[:,:,:3]
    io.imsave('vispy_screenshot.png', im)

    def update(t):

        t = int(t/f_dt)
            
        tray2[0,:] = z[t,0,:] ; tray2[:,0] = z[t,:,0] ; tray2[-1,:] = z[t,-1,:] ; tray2[:,-1] = z[t,:,-1] 
        
        surface.set_data(cx, cy, z[t])
        surface2.set_data(cx, cy, tray2)

        canvas.on_draw(None)

        return _screenshot((0,0,canvas.size[0],canvas.size[1]))[:,:,:3]

    anim = VideoClip(update, duration=t_stop)

    t = time.time()

    anim.write_videofile("3D Vispy.mp4", fps=23.976,  codec = "libx264", bitrate = "12000000")

    print ("Animation with MoviePy : %.02f seconds"%(time.time() - t))

    print("\nVisualization done")

    return 