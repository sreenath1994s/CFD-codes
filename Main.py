######## 1D Burgers Equation #######

### Reference https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/


import numpy as np
import matplotlib.pyplot as plt
import viz_tools, custom

# Domain Parameters and Physical Properties

L_x = 2*np.pi       # Length of the domain in x - direction

nu  = .07           # Viscocity

# Grid - Computational parameters

N_x = 101                      # No: of grid points in x

dx = L_x / (N_x - 1)           # Spacing between grid points

x  = np.linspace(0, L_x, N_x)  # Array with x points

nt = 300                       # No: of time steps

sigma = 0.2
                               
dt = dx*nu                     # Time interval of each time step
                               
c  = 1                         # Wave speed

# Intial condition setup

u_anim = np.ones((nt,N_x))

t = 0.0

u1 = custom.init_calc(t, x, nu)

# Solver 

u2 = np.ones(N_x)

for t in range(nt):

    for i in range(1, N_x-1):                                 # End poinits are the left and right side of the grid are used in the BC

#       u2[i] = u1[i] - c * dt / dx * (u1[i] - u1[i-1])       # 1D Linear Convection

#       u2[i] = u1[i] - u1[i] * dt / dx * (u1[i] - u1[i-1])   # 1D Non Linear Convection
        
        u2[i] =  nu * dt / (dx**2) * (u1[i+1] - 2*u1[i] + u1[i-1]) # 1D Pure Diffusion

        u2[i] = u1[i] - u1[i] * dt / dx * (u1[i] - u1[i-1]) + nu * dt / (dx**2) * (u1[i+1] - 2*u1[i] + u1[i-1])   # Burger's equation. Convection - Diffusion phenomena

        u_anim[t][i] = u1[i]

    u2[0]  = u1[0] - u1[0] * dt / dx * (u1[0] - u1[-2]) + nu * dt / (dx**2) * (u1[1] - 2*u1[0] + u1[-2])                # Boundary conditions

    u2[-1] = u2[0]

    u1 = u2.copy()
    

print( "Simulation Finished.. Visualizing data")

eta_anim = viz_tools.u_animation(x, u_anim)

fig, ax = plt.subplots()

ax.plot(x,u1)

plt.savefig(str(str(t) + ".png"))