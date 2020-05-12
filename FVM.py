######## 1D Advection equation #######

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import jitclass 
from numba import float64    # import the types

# Main Configuration variables

NUM_X_CELLS = 100
NUM_GHOST_CELLS = 3
NUM_RK_STEPS = 1
STOPPING_TIME = 0.5
MAX_TIME_ITERATION = 100000
MIN_X = -1.0
MAX_X = 1.0
ADVECTION_VEL = 1.0
COURANT_NUM = 0.5

# Intialise the cell class

spec = [
    ('u', float64[:]),   
    ('uWest', float64 ),         
    ('uEast', float64 ),
    ('cx', float64 ),
    ('dx', float64 ),
    ('totalflux', float64 ),

]

@jitclass(spec)
class Cell:

    def __init__(self, u, uWest, uEast, cx, dx, totalflux):
        
        self.u = np.zeros(NUM_RK_STEPS + 1)
        self.uWest = uWest
        self.uEast = uEast
        self.cx = cx  # Centroid of cell
        self.dx = dx  # length of a cell
        self.totalflux = totalflux


# Solution initialiser

def solution_init(cells):

    rkstep = 0

    for i in range(len(cells)):

        if ((cells[i].cx > - 0.25) and (cells[i].cx < 0.25)):
            cells[i].u[rkstep] = 1.0
        else:
            cells[i].u[rkstep] = 0.0

    return cells


# Time step calculator

def time_step_calc(cells):

    time_step = cells[0].dx / np.abs(ADVECTION_VEL)

    return time_step


# Boundary condition updater


def update_ghost_cells(cells, rkstep):

    for ghostcell_i in range(0, NUM_GHOST_CELLS):

        cells[ghostcell_i].u[rkstep] = cells[NUM_X_CELLS + ghostcell_i].u[rkstep]

        cells[NUM_X_CELLS + NUM_GHOST_CELLS + ghostcell_i].u[rkstep] = cells[NUM_GHOST_CELLS + ghostcell_i].u[rkstep]

    return cells


# Variable reconstructor

def var_recons(cells, rkstep):

    for i in range (NUM_GHOST_CELLS - 1 , NUM_X_CELLS + NUM_GHOST_CELLS + 1  ):

        # First order reconstruction

        cells[i].uWest = cells[i].u[rkstep]

        cells[i].uEast = cells[i].u[rkstep]

    return cells

# Flux Calculator

def flux_calc(cells, rkstep):

    for i in range(0, len(cells)):
        cells[i].totalflux = 0.0

    for edgeindex in range ( NUM_GHOST_CELLS, NUM_X_CELLS + NUM_GHOST_CELLS + 1):

        leftvalue  = cells[edgeindex - 1].uEast
        rightvalue = cells[edgeindex].uWest

        # Lax-Friedrichs-Scheme for calculating the upwind flux

        flux = 0.5 * ADVECTION_VEL * (leftvalue + rightvalue) - 0.5 * abs ( ADVECTION_VEL ) * (rightvalue - leftvalue)

        cells[edgeindex - 1].totalflux -= flux

        cells[edgeindex].totalflux += flux

    return cells

# Time integration

def update_cell_averages(cells, rkstep, dt):

    for i in range ( NUM_GHOST_CELLS, NUM_X_CELLS + NUM_GHOST_CELLS):
        cells[i].u[rkstep + 1] = cells[i].u[rkstep] + dt / cells[i].dx * cells[i].totalflux

    return cells

# Variable Copy

def var_copy(cells):

    for i in range (0, len(cells)):
        cells[i].u[0] = cells[i].u[NUM_RK_STEPS]

    return cells

# Data structure

cells = np.zeros(NUM_X_CELLS + 2*NUM_GHOST_CELLS, dtype=object)


# Populating grid with data

i = 0 

dx = (MAX_X - MIN_X)/NUM_X_CELLS

for i in range(0, len(cells)):

    cx = MIN_X + dx * ( i + 0.5 - NUM_GHOST_CELLS)

    cells[i] = Cell(0,0,0,cx,dx,0)

cells = solution_init(cells)

# Output initialised solution

u_out = np.zeros(len(cells))
x_out = np.zeros(len(cells))

for i in range(0, len(cells)):

    u_out[i] = cells[i].u[0]
    x_out[i] = cells[i].cx

fig, ax = plt.subplots()

ax.plot(x_out,u_out)

plt.show()


# Start solving

time = 0.0 
lasttimestep = False
time_itr = 0

while (time_itr < MAX_TIME_ITERATION):

    dt = time_step_calc(cells)
    
    if(time + dt > STOPPING_TIME):
        dt = STOPPING_TIME - time
        last_timestep = True

    for rkstep in range(0, NUM_RK_STEPS):

        cells = update_ghost_cells(cells, rkstep)            # Apply boundary condition through ghost cells
       
        cells = var_recons(cells, rkstep)                    # Variable reconstruction

        cells = flux_calc(cells, rkstep)                     # Calculate the flux

        cells = update_cell_averages(cells, rkstep, dt)

        cells = var_copy(cells)

    time += dt
    time_itr += 1

    if (lasttimestep):
        break

for i in range(0, len(cells)):

    u_out[i] = cells[i].u[0]
    x_out[i] = cells[i].cx

fig, ax = plt.subplots()

ax.plot(x_out,u_out)

plt.show()