### Shallow water equation solver based on Finite Volume Method ####
### Written by sreenath subramaniam ###
### https://en.wikipedia.org/wiki/Shallow_water_equations

import numpy as np
import time
import viz_tools
from numba import jit

run_start_time = time.time()                          # Variable used for evaluating program execution time 

# Domain Parameters and Physical Properties

L_x = 2             # Length of the domain in x - direction

L_y = 2             # Length of the domain in y - direction

g = 9.81            # Acceleration due to gravity

theta = np.radians(10)

N_x = 160            # No: of x cells

N_y = 160            # No: of y cells

N_g = 4              # No: of ghost cells

Cou_Num = 0.5

max_itr = 10000     # Maximum no:of iterations
t_start = 0.0       # Starting time of simulation, s
t_stop  = 12.0      # Stopping time of simulation, s

# Data Structure - Grid Data

dx = L_x / (N_x)    # size of x direction cells

dy = L_y / (N_y)    # size of y direction cells

cx  = np.linspace(-N_g*dx/2, L_x + N_g*dx/2 , N_x+2*N_g)  # Centroid of X - Cells

cy  = np.linspace(-N_g*dy/2, L_y + N_g*dy/2 , N_y+2*N_g)  # Centroid of Y - Cells

CY, CX = np.meshgrid(cy, cx) 

U1 =  np.zeros((3, N_x + 2*N_g, N_y + 2*N_g))   # Solution vector  U1[0,:,:] = rho*h ; U1[1,:,:] = rho*h*u ; U1[2,:,:] = rho*h*v 
U2 =  np.zeros_like(U1)                         # Solution vector at n+1 time step

UEast  =  np.zeros_like(U1)  # Right side of a cell
UWest  =  np.zeros_like(U1)  # Left side of a cell
UNorth =  np.zeros_like(U1)  # Top side of a cell
USouth =  np.zeros_like(U1)  # Bottom side of a cell

T_flux =  np.zeros_like(U1)  # Total flux inside a cell

S      =  np.zeros_like(U1)  # Source term

Uanim  = np.zeros((max_itr, 3, N_x + 2*N_g, N_y + 2*N_g))

tanim  = np.zeros((max_itr))

# Intial condition setup

U1[0] = 1 * np.exp(-( ( CX - 1 )**2 / (2*0.04) + (CY - 1 )**2 / (2*0.04) ) )  # Initialising the inital values of height as a gaussian.

U1[1] = 0.0  # Initialising the inital values of u - velocity

U1[2] = 0.0  # Initialising the inital values of v - velocity

#viz_tools.surface_2d(cx, cy, U1[0], "h_initial")

# Time step calculator

@jit(nopython=True, parallel=False, cache=True)
def time_step_calc(U1):

    h = U1[0]
    
    u = U1[1]/h
    
    v = U1[2]/h
    
    c = np.sqrt(g*h)
    
    dtx = dx / (np.abs(u) + c)
    
    dty = dy / (np.abs(u) + c)

    dt  = np.min( 1 / (1/dtx + 1/dty))   

    return dt*Cou_Num


# Boundary condition application

@jit(nopython=True, parallel=False, cache=True)
def boundary_cond(U1):

    #U1[:,0:N_g,:] = U1[:,N_x:N_x+N_g,:]       # Periodic boundary condition for left and right of the boundary 

    #U1[:,-N_g:,:] = U1[:,N_g:2*N_g,:]

    #U1[:,:,0:N_g] = U1[:,:,N_y:N_y+N_g]       # Periodic boundary condition for bottom and top of the boundary 

    #U1[:,:,-N_g:] = U1[:,:,N_g:2*N_g]

    # Vertical wall reflective boundary conditions

    U1[0,0:N_g,:] = U1[0,2*N_g-1:N_g-1:-1,:]      # Boundary condition for height dh / dx = 0 at vertical wall

    U1[0,-N_g:,:] = U1[0,-N_g-1:-2*N_g-1:-1,:]

    U1[1,0:N_g,:] = - U1[1,2*N_g-1:N_g-1:-1,:]    # Boundary condition for u = 0 at vertical wall 

    U1[1,-N_g:,:] = - U1[1,-N_g-1:-2*N_g-1:-1,:]

    U1[2,0:N_g,:] = U1[2,2*N_g-1:N_g-1:-1,:]      # Boundary condition for dv / dx = 0

    U1[2,-N_g:,:] = U1[2,-N_g-1:-2*N_g-1:-1,:]

    # Horizontal wall reflective boundary conditions

    U1[0,:,0:N_g] = U1[0,:,2*N_g-1:N_g-1:-1]      # Boundary condition for height dh / dy = 0 at horizontal wall

    U1[0,:,-N_g:] = U1[0,:,-N_g-1:-2*N_g-1:-1]

    U1[1,:,0:N_g] = U1[1,:,2*N_g-1:N_g-1:-1]      # Boundary condition for du / dy = 0

    U1[1,:,-N_g:] = U1[1,:,-N_g-1:-2*N_g-1:-1]

    U1[2,:,0:N_g] = - U1[2,:,2*N_g-1:N_g-1:-1]    # Boundary condition for v = 0 at horizontal wall

    U1[2,:,-N_g:] = - U1[2,:,-N_g-1:-2*N_g-1:-1]

    return U1

# Variable reconstructor
@jit(nopython=True, parallel=False, cache=True)
def variable_recon(U1, UEast, UWest, UNorth, USouth): 

    UEast [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]      # First order reconstruction 
    UWest [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]      
    UNorth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]
    USouth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]

    return UEast, UWest, UNorth, USouth

# Flux functions
@jit(nopython=True, parallel=False, cache=True)
def F(U):
    h = U[0]
    u = U[1]/h
    v = U[2]/h

    sol = np.zeros_like(U)

    sol[0,:,:] = h*u
    sol[1,:,:] = h*u*u + 0.5*g*h*h
    sol[2,:,:] = h*u*v

    return sol

@jit(nopython=True, parallel=False, cache=True)
def G(U):
    h = U[0]
    u = U[1]/h
    v = U[2]/h

    sol = np.zeros_like(U)

    sol[0,:,:] = h*v
    sol[1,:,:] = h*u*v
    sol[2,:,:] = h*v*v + 0.5*g*h*h

    return sol


# Flux Calculator

@jit(nopython=True, parallel=False, cache=True)
def calculate_flux(U1, UEast, UWest, UNorth, USouth, T_flux):

    T_flux =  np.zeros_like(U1)  # Total flux inside a cell

    ### Vertical edge calculation

    leftvalue  = np.zeros_like(U1)
    rightvalue = np.zeros_like(U1)

    leftvalue  = UEast[:,N_g-1:N_g+N_x,N_g:N_g+N_y]
    rightvalue = UWest[:,N_g:N_g+N_x+1,N_g:N_g+N_y]

    hL = leftvalue[0]
    uL = leftvalue[1] / hL
    cL = np.sqrt(g*hL)
    maxspeedL = np.abs(uL) + cL

    hR = rightvalue[0]
    uR = rightvalue[1] / hR
    cR = np.sqrt(g*hR)
    maxspeedR = np.abs(uR) + cR

    maxSpeed = np.maximum(maxspeedL,maxspeedR)      # Returns array of maximums

    FL = F(leftvalue)
    FR = F(rightvalue)

            ### Lax Friedrichs scheme

    flux = 0.5 * (FL + FR) - 0.5 * maxSpeed * (rightvalue - leftvalue)

    flux *= dy

    T_flux[:,N_g-1:N_g+N_x,N_g:N_g+N_y] -= flux
    T_flux[:,N_g:N_g+N_x+1,N_g:N_g+N_y] += flux

    ### horizontal edge calculation

    leftvalue  = UNorth[:, N_g:N_g+N_x, N_g-1:N_g+N_y ]
    rightvalue = USouth[:, N_g:N_g+N_x, N_g:N_g+N_y+1 ]
    
    hL = leftvalue[0]
    vL = leftvalue[2] / hL
    cL = np.sqrt(g*hL)
    maxspeedL = np.abs(vL) + cL
    
    hR = rightvalue[0]
    vR = rightvalue[2] / hR
    cR = np.sqrt(g*hR)
    maxspeedR = np.abs(vR) + cR
    
    maxSpeed = np.maximum(maxspeedL,maxspeedR)      # Returns array of maximums

    GL = G(leftvalue)
    GR = G(rightvalue)

                ### Lax Friedrichs scheme

    flux = 0.5 * (GL + GR) - 0.5 * maxSpeed * (rightvalue - leftvalue)
    flux *= dx

    T_flux[:, N_g:N_g+N_x, N_g-1:N_g+N_y] -= flux
    T_flux[:, N_g:N_g+N_x, N_g:N_g+N_y+1] += flux

    return T_flux

# Source term
@jit(nopython=True, parallel=False, cache=True)
def source(U1, S, theta):

    S =  np.zeros_like(U1)

    S[1,N_g:N_g+N_x,N_g:N_g+N_y] = g * U1[0,N_g:N_g+N_x,N_g:N_g+N_y] * np.sin(theta) * dx*dy

    return S

# Time integration
@jit(nopython=True, parallel=False, cache=True)
def time_integration(U1, U2, T_flux, dt, S):

    area = dx*dy

    U2[:,N_g:N_g+N_x,N_g:N_g+N_y] = U1[:,N_g:N_g+N_x,N_g:N_g+N_y] + dt / area * ( T_flux[:,N_g:N_g+N_x,N_g:N_g+N_y] + S[:,N_g:N_g+N_x,N_g:N_g+N_y])   # Euler's time stepping

    return U2

# Solver
@jit(nopython=True, parallel=False, cache=True)
def solver(t_start,U1,UEast, UWest, UNorth, USouth, T_flux, S, U2, theta, Uanim, tanim ):

    cur_itr = 0
    
    t       = t_start
    
    lastTimestep = False
    
    while cur_itr < max_itr :
    
        dt = time_step_calc(U1)
    
        print(t,dt)
    
        if(t+dt > t_stop):
            dt = t_stop - t
            lastTimestep = True
    
        ### Save variable for animation ###
    
        Uanim[cur_itr] = U1
        tanim[cur_itr] = t
    
        ### main code here ###
    
        U1 = boundary_cond (U1)  # Boundary condition updater
    
        UEast, UWest, UNorth, USouth = variable_recon(U1, UEast, UWest, UNorth, USouth)  # Reconstruct variables from centroid to the edges
    
        T_flux = calculate_flux(U1, UEast, UWest, UNorth, USouth, T_flux)  # Flux calculation
    
        S = source(U1, S, theta) # Source calculation
        
        U2 = time_integration(U1, U2, T_flux, dt, S) # Time integration
    
        U1[:,N_g:N_g+N_x,N_g:N_g+N_y] = U2[:,N_g:N_g+N_x,N_g:N_g+N_y].copy()
    
        ### main code here ###
    
        if(lastTimestep):
            break
    
        t += dt
        cur_itr += 1

    return U1, Uanim, tanim, cur_itr

U1, Uanim, tanim, cur_itr = solver(t_start,U1,UEast, UWest, UNorth, USouth, T_flux, S, U2, theta, Uanim, tanim )

print ("Simulation finished in ", time.time()-run_start_time)

# Visualisation Section

#viz_tools.surface3D(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], U1[0,N_g:N_g+N_x,N_g:N_g+N_y], str(cur_itr))

#viz_tools.surface_3D_gpu(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], U1[0,N_g:N_g+N_x,N_g:N_g+N_y], str(cur_itr))

#viz_tools.animation_3D(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], Uanim[:,0,N_g:N_g+N_x,N_g:N_g+N_y], tanim, cur_itr, t_stop)

viz_tools.animation_3D_gpu(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], Uanim[:,0,N_g:N_g+N_x,N_g:N_g+N_y], tanim, cur_itr, t_stop)

#viz_tools.animation_3D_fast(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], Uanim[:,0,N_g:N_g+N_x,N_g:N_g+N_y], tanim, cur_itr, t_stop)



