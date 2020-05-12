######## 2D Advection Equation #######

### Reference https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/


import numpy as np
import matplotlib.pyplot as plt
import viz_tools, custom
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

# Domain Parameters and Physical Properties

L_x = 2             # Length of the domain in x - direction

L_y = 2             # Length of the domain in y - direction

nu  = .07           # Viscocity

c  = 1              # Wave speed

sigma = 0.2

# Grid - Computational parameters

N_x = 81                       # No: of grid points in x

N_y = 81                       # No: of grid points in y

dx = L_x / (N_x - 1)           # Spacing between grid points

dy = L_y / (N_y - 1)           # Spacing between grid points

x  = np.linspace(0, L_x, N_x)  # Array with x points

y  = np.linspace(0, L_y, N_y)  # Array with x points

nt = 100                       # No: of time steps
                               
dt = dx*sigma                  # Time interval of each time step
                               
# Intial condition setup

u_anim = np.ones((nt,N_x,N_y))

t = 0.0

u1 =  np.ones((N_x, N_y))

u1[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2.0     # Initial solution

fig = plt.figure()
ax  = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)                            
surf = ax.plot_surface(X, Y, u1[:], cmap=cm.viridis)
plt.show()

# Solver 

u2 = np.ones((N_x, N_y))

for t in range(nt):

    for i in range(1, N_x-1):                                      # X End poinits are the left and right side of the grid are used in the BC

        for j in range(1, N_y-1):                                  # Y End poinits are the left and right side of the grid are used in the BC
                                                                   
#       u2[i] = u1[i] - c * dt / dx * (u1[i] - u1[i-1])            # 1D Linear Convection
                                                                   
#       u2[i] = u1[i] - u1[i] * dt / dx * (u1[i] - u1[i-1])        # 1D Non Linear Convection
        
#       u2[i] =  nu * dt / (dx**2) * (u1[i+1] - 2*u1[i] + u1[i-1]) # 1D Pure Diffusion

#       u2[i] = u1[i] - u1[i] * dt / dx * (u1[i] - u1[i-1]) + nu * dt / (dx**2) * (u1[i+1] - 2*u1[i] + u1[i-1])   # Burger's equation. Convection - Diffusion phenomena

            u2[i,j] = u1[i,j] - c * dt / dx * (u1[i,j] - u1[i-1,j]) - c * dt / dy * (u1[i,j] - u1[i,j-1])  # 2D Convection equation

            u_anim[t][i][j] = u1[i,j]

#    u2[0]  = u1[0] - u1[0] * dt / dx * (u1[0] - u1[-2]) + nu * dt / (dx**2) * (u1[1] - 2*u1[0] + u1[-2])   # Boundary conditions burger's equation

#    u2[-1] = u2[0]
    
    u2[0,:]  = 1  # x Boundary conditions
    u2[-1,:] = 1  # 
    u2[:,0]  = 1  # y Boundary conditions
    u2[:,-1] = 1  # 

    u1 = u2.copy()
    

print( "Simulation Finished.. Visualizing data")

#eta_anim = viz_tools.u_animation(x, u_anim)
eta_anim = viz_tools.u_2d_animation(x, y, u_anim)

fig = plt.figure()
ax  = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)                            
surf = ax.plot_surface(X, Y, u1[:], cmap=cm.viridis)
plt.show()