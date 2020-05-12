######## 1D Pure Diffusion - Finite Difference Method #######

### Reference https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/


import numpy as np
import matplotlib.pyplot as plt
import viz_tools

# Domain Parameters and Physical Properties

L_x = 2     # Length of the domain in x - direction

nu  = 0.1   # Viscocity

# Grid - Computational parameters

N_x = 41                       # No: of grid points in x

dx = L_x / (N_x - 1)           # Spacing between grid points

x  = np.linspace(0, L_x, N_x)  # Array with x points

nt = 100                        # No: of time steps

sigma = 0.2
                               
dt = sigma * dx**2 / nu        # Time interval of each time step
                               
c  = 1                         # Wave speed

# Intial condition setup

u_anim = np.ones((nt,N_x))

u1 = np.ones(N_x)

u1[ int(0.5/dx) : int((1 / dx) + 1)] = 2

# Solver 

u2 = np.ones(N_x)

for t in range(nt):

    for i in range(1, N_x - 1):    

#       u2[i] = u1[i] - c * dt / dx * (u1[i] - u1[i-1])       # 1D Linear Convection

        u2[i] = u1[i] - u1[i] * dt / dx * (u1[i] - u1[i-1])   # 1D Non Linear Convection

#        u2[i] = u1[i] + nu * dt / (dx**2) * (u1[i+1] - 2*u1[i] + u1[i-1])   # 1D pure diffusio, central difference approximation for the second order spatial derivative.

        u_anim[t][i] = u1[i]

    u1 = u2.copy()

    u1[0]  = 1.0               # Boundary conditions

    u1[-1] = 1.0
    

print( "Simulation Finished.. Visualizing data")

eta_anim = viz_tools.u_animation(x, u_anim)

fig, ax = plt.subplots()

ax.plot(x,u1)

plt.savefig(str(str(t) + ".png"))