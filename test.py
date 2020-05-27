
#=========


import time
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

fps = 2

f_dt = 1/fps


fig, ax = plt.subplots( figsize=(6,6), facecolor=[1,1,1] )
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x), lw=3)

def make_frame(t):
    line.set_ydata(np.sin(x+2*t))  # update the data
    return mplfig_to_npimage(fig)

anim = VideoClip(make_frame, duration=10)

t = time.time()
anim.write_videofile("test_mpl_mpy.mp4", fps=30)
print ("Animation with MoviePy : %.02f seconds"%(time.time() - t))