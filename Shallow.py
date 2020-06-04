### Shallow water equation solver based on Finite Volume Method ####
### Written by sreenath subramaniam ###
### https://en.wikipedia.org/wiki/Shallow_water_equations

import numpy as np
import time
import viz_tools
from numba import jit
from weno  import *

run_start_time = time.time()                          # Variable used for evaluating program execution time 


####### Domain Parameters and Physical Properties #########

L_x = 2             # Length of the domain in x - direction

L_y = 2             # Length of the domain in y - direction

g = 9.80665         # Acceleration due to gravity

theta = np.radians(10)  # Angle of tilt of the tank

N_x = 50            # No: of x cells

N_y = 50            # No: of y cells

N_g = 4             # No: of ghost cells

Cou_Num = 0.5       # Courant number must be less than one

max_itr = 10000     # Maximum no:of iterations

t_start = 0.0       # Starting time of simulation, s

t_stop  = 12.0      # Stopping time of simulation, s

RKsteps = 3         # No. of RK steps, min 1, max 3

recons  = 5         # Reconstruction type 1 - 1st , 2 - 2nd ( Lax-Wendroff ), 5 - 5th (WENO)

########## Data Structure - Grid Data ##########

dx = L_x / (N_x)    # size of x direction cells

dy = L_y / (N_y)    # size of y direction cells

cx  = np.linspace(-N_g*dx/2, L_x + N_g*dx/2 , N_x+2*N_g)  # Centroid of X - Cells

cy  = np.linspace(-N_g*dy/2, L_y + N_g*dy/2 , N_y+2*N_g)  # Centroid of Y - Cells

CY, CX = np.meshgrid(cy, cx) 

U1 =  np.zeros((3, N_x + 2*N_g, N_y + 2*N_g))   # Solution vector  U1[0,:,:] = rho*h ; U1[1,:,:] = rho*h*u ; U1[2,:,:] = rho*h*v 

U2 =  np.zeros((RKsteps+1, 3, N_x + 2*N_g, N_y + 2*N_g))    # Solution vector for different RK Steps

UEast  =  np.zeros_like(U1)  # Right side of a cell
UWest  =  np.zeros_like(U1)  # Left side of a cell
UNorth =  np.zeros_like(U1)  # Top side of a cell
USouth =  np.zeros_like(U1)  # Bottom side of a cell

T_flux =  np.zeros_like(U1)  # Total flux inside a cell

S      =  np.zeros_like(U1)  # Source term

Uanim  = np.zeros((max_itr, 3, N_x + 2*N_g, N_y + 2*N_g)) # Arrays for storing animation data.
                                                          #
tanim  = np.zeros((max_itr))                              # 


######## Intial condition setup ########

U1[0] = 0.5+ 1 * np.exp(-( ( CX - 1 )**2 / (2*0.04) + (CY - 1 )**2 / (2*0.04) ) )  # Initialising the inital values of height as a gaussian.

U1[1] = 0.0  # Initialising the inital values of u - velocity

U1[2] = 0.0  # Initialising the inital values of v - velocity

#viz_tools.surface_2d(cx, cy, U1[0], "h_initial")   # Plot initial condtion


###### Time step calculator based on CFL Condition #####        https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

@jit(nopython=True, parallel=False, cache=True)
def time_step_calc(U1):

    h = U1[0,N_g:N_g+N_x,N_g:N_g+N_y]
    
    u = U1[1,N_g:N_g+N_x,N_g:N_g+N_y]/h
    
    v = U1[2,N_g:N_g+N_x,N_g:N_g+N_y]/h
    
    c = np.sqrt(g*h) #Inc
    
    dtx = dx / (np.abs(u) + c) #Inc
    
    dty = dy / (np.abs(u) + c) #Inc

    dt  = np.min( 1.0 / (1.0/dtx + 1.0/dty))

    return dt*Cou_Num


###### Boundary condition application #######

@jit(nopython=True, parallel=False, cache=True)
def boundary_cond(U1):

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

###### Variable reconstructor #######

@jit(nopython=True, parallel=False, cache=True)
def variable_recon(U1, UEast, UWest, UNorth, USouth, recons): 

    # First order reconstruction 

    if (recons==1):

        UEast [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]     
        UWest [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]        
        UNorth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]
        USouth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]
    
    # Second order reconstruction with Lax Wendroff Method and Van - Albada Limiter 1 https://en.wikipedia.org/wiki/Flux_limiter

    if (recons==2):

        for i in range(N_g-1, N_x+N_g+1):
    
            for j in range(N_g-1, N_y+N_g+1):

                for var in range(3):

                    # Flux Limiter calculation

                    num = U1[var, i , j] - U1[var, i-1 , j]                
                    den = U1[var, i+1, j] - U1[var, i , j]                 
                                                                           
                    if (np.abs(den) < 1e-40):                              
                        phi_x = 1.0                                        
                    else:                                                  
                        rx = num/den                                       
                        rx = max(0.0,rx)                                   
                        phi_x = ( rx * rx + rx) / (rx*rx +1.0)             
                                                                           
                    num = U1[var, i , j] - U1[var, i , j-1]                
                    den = U1[var, i, j+1] - U1[var, i , j]                 
                                                                           
                    if (np.abs(den) < 1e-40):                              
                        phi_y = 1.0                                        
                    else:                                                  
                        ry = num/den                                       
                        ry = max(0.0,ry)                                   
                        phi_y = ( ry * ry + ry) / (ry*ry +1.0)                 

                    # Lax Wendroff Reconstruction (Forward - difference to calculate the slope)

                    du_dx = (U1[var, i+1 , j] - U1[var, i , j])/ dx

                    du_dy = (U1[var, i , j+1] - U1[var, i , j])/ dy

                    UEast[var,i,j]  = U1[var, i , j] + phi_x * du_dx * dx/2.0

                    UWest[var,i,j]  = U1[var, i , j] - phi_x * du_dx * dx/2.0

                    UNorth[var,i,j] = U1[var, i , j] + phi_y * du_dy * dy/2.0

                    USouth[var,i,j] = U1[var, i , j] - phi_y * du_dy * dy/2.0

    # 3 Sencil Fifth order reconstruction with WENO

    if (recons==5):

        UEast [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = weno_east (U1, N_g, N_x, N_y)
        UWest [:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = weno_west (U1, N_g, N_x, N_y)
        UNorth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = weno_north(U1, N_g, N_x, N_y)
        USouth[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1] = weno_south(U1, N_g, N_x, N_y)

    return UEast, UWest, UNorth, USouth


###### Flux functions ######

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


####### Numerical Flux Calculator #######

@jit(nopython=True, parallel=False, cache=True)
def calculate_flux(U1, UEast, UWest, UNorth, USouth, T_flux):

    T_flux =  np.zeros_like(U1)  # Total flux inside a cell

    ### Vertical edge calculation

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

            ### Lax Friedrich's flux

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

                ### Lax Friedrich's flux

    flux = 0.5 * (GL + GR) - 0.5 * maxSpeed * (rightvalue - leftvalue)
    flux *= dx

    T_flux[:, N_g:N_g+N_x, N_g-1:N_g+N_y] -= flux
    T_flux[:, N_g:N_g+N_x, N_g:N_g+N_y+1] += flux

    return T_flux

##### Source term #####

@jit(nopython=True, parallel=False, cache=True)
def source(U1, S, theta):

    S =  np.zeros_like(U1)

    S[1,N_g:N_g+N_x,N_g:N_g+N_y] = g * U1[0,N_g:N_g+N_x,N_g:N_g+N_y] * np.sin(theta) * dx*dy    # Inclination about x axis

    return S

##### Time integration ######

@jit(nopython=True, parallel=False, cache=True)
def time_integration(U2, T_flux, dt, S, rkstep):

    area = dx*dy
    
    if(rkstep==1):
    
        U2[rkstep,:,N_g:N_g+N_x,N_g:N_g+N_y] = U2[rkstep-1,:,N_g:N_g+N_x,N_g:N_g+N_y] + dt / area * ( T_flux[:,N_g:N_g+N_x,N_g:N_g+N_y] + S[:,N_g:N_g+N_x,N_g:N_g+N_y])   
    
    if(rkstep==2):
    
        U2[rkstep,:,N_g:N_g+N_x,N_g:N_g+N_y] = (3/4) * U2[rkstep-2,:,N_g:N_g+N_x,N_g:N_g+N_y] + \
                                               (1/4) * U2[rkstep-1,:,N_g:N_g+N_x,N_g:N_g+N_y] + (1/4) * dt / area * ( T_flux[:,N_g:N_g+N_x,N_g:N_g+N_y] + S[:,N_g:N_g+N_x,N_g:N_g+N_y])   
    
    if(rkstep==3):
        
        U2[rkstep,:,N_g:N_g+N_x,N_g:N_g+N_y] = (1/3) * U2[rkstep-3,:,N_g:N_g+N_x,N_g:N_g+N_y] + \
                                               (2/3) * U2[rkstep-1,:,N_g:N_g+N_x,N_g:N_g+N_y] + (2/3) * dt / area * ( T_flux[:,N_g:N_g+N_x,N_g:N_g+N_y] + S[:,N_g:N_g+N_x,N_g:N_g+N_y])   

    return U2 

##### Solver ######

@jit(nopython=True, parallel=False, cache=True)
def solver(t_start, U1, U2, UEast, UWest, UNorth, USouth, recons, T_flux, S, theta, Uanim, tanim  ):

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

        rkstep = 0

        U2[rkstep] = U1.copy()
    
        for rkstep in range(1, RKsteps+1):  # Runge kutta iteration

            U1 = boundary_cond (U1)         # Boundary condition updater
    
            UEast, UWest, UNorth, USouth = variable_recon(U1, UEast, UWest, UNorth, USouth, recons)  # Reconstruct variables from centroid to the edges
    
            T_flux = calculate_flux(U1, UEast, UWest, UNorth, USouth, T_flux)  # Flux calculation

            S = source(U1, S, theta)        # Source calculation
    
            U2 = time_integration(U2, T_flux, dt, S, rkstep) # Time integration
    
            U1 = U2[rkstep].copy()

        ### main code here ###
    
        if(lastTimestep):
            break
    
        t += dt
        cur_itr += 1

    return U1, Uanim, tanim, cur_itr

U1, Uanim, tanim, cur_itr = solver(t_start, U1, U2, UEast, UWest, UNorth, USouth, recons, T_flux, S, theta, Uanim, tanim )

print ("\nSimulation finished in ", time.time()-run_start_time)

####### Visualization Section #######

viz_tools.animation_3D_gpu(cx[N_g:N_g+N_x], cy[N_g:N_g+N_y], Uanim[:,0,N_g:N_g+N_x,N_g:N_g+N_y], tanim, cur_itr, t_stop)



