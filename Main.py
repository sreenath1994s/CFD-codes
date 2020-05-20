######## 2D Laplace Equation #######

### Reference https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/


import numpy as np
import matplotlib.pyplot as plt
import viz_tools, custom

#from numba import jit
import time

run_start_time = time.time()                          # Variable used for evaluating program execution time 

# Domain Parameters and Physical Properties

L_x = 2             # Length of the domain in x - direction

L_y = 1             # Length of the domain in y - direction

nu  = .01           # Viscocity

c  = 1              # Wave speed

sigma = 0.0009

# Grid - Computational parameters

N_x = 31                       # No: of grid points in x

N_y = 31                       # No: of grid points in y

dx = L_x / (N_x - 1)           # Spacing between grid points

dy = L_y / (N_y - 1)           # Spacing between grid points

x  = np.linspace(0, L_x, N_x)  # Array with x points

y  = np.linspace(0, L_y, N_y)  # Array with x points

tf = 3                         # Stopping time in seconds
                               
dt = dx*sigma * dy / nu        # Time interval of each time step

nt = 120                       # No: of time steps

                               
# Intial condition setup

u_anim = np.ones((nt,N_x,N_y)) # Variable for storing the animation data

u1 =  np.ones((N_x, N_y))

v1 =  np.ones((N_x, N_y))

u1[int(.5 / dx):int(1 / dx + 1),int(.5 / dy):int(1 / dy + 1)] = 2.0     # Initial solution

v1[int(.5 / dx):int(1 / dx + 1),int(.5 / dy):int(1 / dy + 1)] = 2.0     # Initial solution

p1 = np.zeros((N_x, N_y))

p1[0  , :] = 0.0       # p = 0 at x = 0
p1[-1 , :] = y         # p = 2 at x = 2
p1[:  , 0] = p1[: , 1] # dp/dy = 0 at y = 0
p1[:  ,-1] = p1[: ,-2] # dp/dy = 0 at y = 1

viz_tools.surface_2d(x, y, p1, "P_initial")

# Solver 

u2 = np.ones((N_x, N_y))

v2 = np.ones((N_x, N_y))

#@jit(nopython=True, cache=True)


def laplace2d(p1, y, dx, dy, l1norm_target):

    l1norm = 1
    p2 = np.zeros((N_x, N_y))

    while l1norm > l1norm_target:
        p2[1:-1, 1:-1] = ( dy**2 * (p1[2:, 1:-1] + p1[0:-2, 1:-1]) + dx**2 * (p1[1:-1, 2:] + p1[1:-1, 0:-2])  ) / (2 * (dx**2 + dy**2))

        p2[0  , :] = 0.0       # p = 0 at x = 0
        p2[-1 , :] = y       # p = 2 at x = 2
        p2[:  , 0] = p1[: , 1] # dp/dy = 0 at y = 0
        p2[:  ,-1] = p1[: ,-2] # dp/dy = 0 at y = 1
        
        l1norm = (np.sum(np.abs(p2) - np.abs(p1)) / np.sum(np.abs(p1)))

        p1 = p2.copy()

    return p1

p1 = laplace2d(p1, y, dx, dy, 1e-8)

viz_tools.surface_2d(x, y, p1, "P_final")

def solver(u1,u2,v1,v2,dt,dx,dy,u_anim,nt):

    for t in range(nt):
    
        #for i in range(1, N_x-1):                                      # X End poinits are the left and right side of the grid are used in the BC
        #
        #    for j in range(1, N_y-1):                                  # Y End poinits are the left and right side of the grid are used in the BC
        #   
        #        u2[i,j] = u1[i,j] - c * dt / dx * (u1[i,j] - u1[i-1,j]) - c * dt / dy * (u1[i,j] - u1[i,j-1])  # 2D Convection equation
        #   
        #        u_anim[t][i][j] = u1[i,j]
        
        u2[1:-1,1:-1] = u1[1:-1,1:-1] \
                        - u1[1:-1,1:-1] * dt / dx * (u1[1:-1,1:-1] - u1[0:-2,1:-1]) - v1[1:-1,1:-1] * dt / dy * (u1[1:-1,1:-1] - u1[1:-1,0:-2]) \
                        + nu * dt / (dx**2) * (u1[2:,1:-1] - 2*u1[1:-1,1:-1] + u1[0:-2,1:-1]) + nu * dt / (dy**2) * (u1[1:-1,2:] - 2*u1[1:-1,1:-1] + u1[1:-1,0:-2]) # 2D  Non - Linear Convection equation
    
        v2[1:-1,1:-1] = v1[1:-1,1:-1] \
                        - v1[1:-1,1:-1] * dt / dx * (v1[1:-1,1:-1] - v1[0:-2,1:-1]) - v1[1:-1,1:-1] * dt / dy * (v1[1:-1,1:-1] - v1[1:-1,0:-2]) \
                        + nu * dt / (dx**2) * (v1[2:,1:-1] - 2*v1[1:-1,1:-1] + v1[0:-2,1:-1]) + nu * dt / (dy**2) * (v1[1:-1,2:] - 2*v1[1:-1,1:-1] + v1[1:-1,0:-2])
    
        # u2[1:,1:] = u1[1:,1:] - c * dt / dx * (u1[1:,1:] - u1[:-1,1:]) - c * dt / dy * (u1[1:,1:] - u1[1:,:-1])  # 2D Convection equation optimised version
    
        u_anim[t, 1:, 1:] = u1[1: , 1:]
    
        u2[0,:]  = 1  # x Boundary conditions
        u2[-1,:] = 1  # 
        u2[:,0]  = 1  # y Boundary conditions
        u2[:,-1] = 1  # 
    
        v2[0,:]  = 1  # x Boundary conditions
        v2[-1,:] = 1  # 
        v2[:,0]  = 1  # y Boundary conditions
        v2[:,-1] = 1  # 
    
        u1 = u2.copy()
    
        v1 = v2.copy()

    return u1,u2,v1,v2,dt,dx,dy,u_anim

#u1,u2,v1,v2,dt,dx,dy,u_anim = solver(u1,u2,v1,v2,dt,dx,dy,u_anim,nt)  

print( "Simulation Finished.. Visualizing data")

print("\n Simulation time is --- %s seconds ---" % (time.time() - run_start_time))

viz_tools.animation_3d(x, y, u_anim, dt)

viz_tools.surface_2d(x, y, u1, "fig2")

print(np.sum(u1))