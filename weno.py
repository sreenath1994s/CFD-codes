# WENO Reconstruction Module
# Written in Python by Sreenath S
# Based on:
# C.W. Shu's Lectures notes.

# Domain cells (I{i}) reference:
#
#                |           |   u(i)    |           |
#                |  u(i-1)   |___________|           |
#                |___________|           |   u(i+1)  |
#                |           |           |___________|
#             ...|-----0-----|-----0-----|-----0-----|...
#                |    i-1    |     i     |    i+1    |
#                |-         +|-         +|-         +|
#              i-3/2       i-1/2       i+1/2       i+3/2
#
# WENO stencils (S{r}) reference:
#
#
#                           |___________S2__________|
#                           |                       |
#                   |___________S1__________|       |
#                   |                       |       |
#           |___________S0__________|       |       |
#         ..|---o---|---o---|---o---|---o---|---o---|...
#           | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
#                                  -|
#                                 i+1/2
#
#
#                   |___________S0__________|
#                   |                       |
#                   |       |___________S1__________|
#                   |       |                       |
#                   |       |       |___________S2__________|
#                 ..|---o---|---o---|---o---|---o---|---o---|...
#                   | I{i-1}|  I{i} | I{i+1}| I{i+2}| I{i+3}|
#                                   |+
#                                 i+1/2

import numpy as np
from numba import jit

@jit(nopython=True, parallel=False, cache=True)
def weno_east(U1, N_g, N_x, N_y):      # U_i+1/2,j (-)

    # Stencils

    Umm = U1[:, N_g-3:N_x+N_g-1 ,  N_g-1:N_y+N_g+1]     # U(i-2)
    Um  = U1[:, N_g-2:N_x+N_g   ,  N_g-1:N_y+N_g+1]     # U(i-1)
    Uo  = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]     # U(i)
    Up  = U1[:, N_g  :N_x+N_g+2 ,  N_g-1:N_y+N_g+1]     # U(i+1)
    Upp = U1[:, N_g+1:N_x+N_g+3 ,  N_g-1:N_y+N_g+1]     # U(i+2)

    # Smoothness inidcator

    beta_1 = 13/12 * (Umm - 2*Um + Uo)**2  + 1/4 * (Umm  - 4*Um + 3*Uo)**2
    beta_2 = 13/12 * (Um  - 2*Uo + Up)**2  + 1/4 * (Um   - Up)**2
    beta_3 = 13/12 * (Uo  - 2*Up + Upp)**2 + 1/4 * (3*Uo - 4*Up + Upp )**2

    # Constants

    gamma_1 = 1/10
    gamma_2 = 3/5
    gamma_3 = 3/10

    epsilon = 1e-80

    # Alpha weights

    alpha_1 = gamma_1 / (epsilon + beta_1)**2
    alpha_2 = gamma_2 / (epsilon + beta_2)**2
    alpha_3 = gamma_3 / (epsilon + beta_3)**2

    alpha_sum = alpha_1 + alpha_2 + alpha_3

    # Stencil weights

    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum
    w_3 = alpha_3 / alpha_sum

    # Reconstructed variable 

    UEast =   w_1 * ( 2*Umm - 7*Um + 11*Uo )/6 \
            + w_2 * ( -Um   + 5*Uo + 2*Up  )/6 \
            + w_3 * ( 2*Uo  + 5*Up - Upp   )/6

    return UEast

@jit(nopython=True, parallel=False, cache=True)
def weno_west(U1, N_g, N_x, N_y):      # U_i-1/2,j (+)

    # Stencils

    Umm = U1[:, N_g-3:N_x+N_g-1 ,  N_g-1:N_y+N_g+1]     # U(i-2)
    Um  = U1[:, N_g-2:N_x+N_g   ,  N_g-1:N_y+N_g+1]     # U(i-1)
    Uo  = U1[:, N_g-1:N_x+N_g+1 ,  N_g-1:N_y+N_g+1]     # U(i)
    Up  = U1[:, N_g  :N_x+N_g+2 ,  N_g-1:N_y+N_g+1]     # U(i+1)
    Upp = U1[:, N_g+1:N_x+N_g+3 ,  N_g-1:N_y+N_g+1]     # U(i+2)

    # Smoothness inidcator

    beta_1 = 13/12 * (Umm - 2*Um + Uo)**2  + 1/4 * (Umm  - 4*Um + 3*Uo)**2
    beta_2 = 13/12 * (Um  - 2*Uo + Up)**2  + 1/4 * (Um   - Up)**2
    beta_3 = 13/12 * (Uo  - 2*Up + Upp)**2 + 1/4 * (3*Uo - 4*Up + Upp )**2

    # Linear Weights

    gamma_1 = 3/10
    gamma_2 = 3/5
    gamma_3 = 1/10

    epsilon = 1e-80

    # Alpha weights

    alpha_1 = gamma_1 / (epsilon + beta_1)**2
    alpha_2 = gamma_2 / (epsilon + beta_2)**2
    alpha_3 = gamma_3 / (epsilon + beta_3)**2

    alpha_sum = alpha_1 + alpha_2 + alpha_3

    # Non - Linear Stencil weights

    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum
    w_3 = alpha_3 / alpha_sum

    # Reconstructed variable 

    UWest =   w_1 * ( -Umm  + 5*Um + 2*Uo )/6 \
            + w_2 * ( 2*Um  + 5*Uo - Up   )/6 \
            + w_3 * ( 11*Uo - 7*Up + 2*Upp)/6

    return UWest

@jit(nopython=True, parallel=False, cache=True)
def weno_north(U1, N_g, N_x, N_y):     # U_i,j+1/2 (-)

    # Stencils

    Umm = U1[:, N_g-1:N_x+N_g+1, N_g-3:N_y+N_g-1]     # U(j-2)
    Um  = U1[:, N_g-1:N_x+N_g+1, N_g-2:N_y+N_g  ]     # U(j-1)
    Uo  = U1[:, N_g-1:N_x+N_g+1, N_g-1:N_y+N_g+1]     # U(j)
    Up  = U1[:, N_g-1:N_x+N_g+1, N_g  :N_y+N_g+2]     # U(j+1)
    Upp = U1[:, N_g-1:N_x+N_g+1, N_g+1:N_y+N_g+3]     # U(j+2)

    # Smoothness inidcator

    beta_1 = 13/12 * (Umm - 2*Um + Uo)**2  + 1/4 * (Umm  - 4*Um + 3*Uo)**2
    beta_2 = 13/12 * (Um  - 2*Uo + Up)**2  + 1/4 * (Um   - Up)**2
    beta_3 = 13/12 * (Uo  - 2*Up + Upp)**2 + 1/4 * (3*Uo - 4*Up + Upp )**2

    # Constants

    gamma_1 = 1/10
    gamma_2 = 3/5
    gamma_3 = 3/10

    epsilon = 1e-6

    # Alpha weights

    alpha_1 = gamma_1 / (epsilon + beta_1)**2
    alpha_2 = gamma_2 / (epsilon + beta_2)**2
    alpha_3 = gamma_3 / (epsilon + beta_3)**2

    alpha_sum = alpha_1 + alpha_2 + alpha_3

    # Stencil weights

    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum
    w_3 = alpha_3 / alpha_sum

    # Reconstructed variable 

    UNorth =   w_1 * ( 2*Umm - 7*Um + 11*Uo )/6 \
            + w_2 * ( -Um   + 5*Uo + 2*Up  )/6 \
            + w_3 * ( 2*Uo  + 5*Up - Upp   )/6
    
    return UNorth

@jit(nopython=True, parallel=False, cache=True)
def weno_south(U1, N_g, N_x, N_y):     # U_i,j-1/2 (+)

    # Stencils
    
    Umm = U1[:, N_g-1:N_x+N_g+1, N_g-3:N_y+N_g-1]     # U(j-2)
    Um  = U1[:, N_g-1:N_x+N_g+1, N_g-2:N_y+N_g  ]     # U(j-1)
    Uo  = U1[:, N_g-1:N_x+N_g+1, N_g-1:N_y+N_g+1]     # U(j)
    Up  = U1[:, N_g-1:N_x+N_g+1, N_g  :N_y+N_g+2]     # U(j+1)
    Upp = U1[:, N_g-1:N_x+N_g+1, N_g+1:N_y+N_g+3]     # U(j+2)

    # Smoothness inidcator

    beta_1 = 13/12 * (Umm - 2*Um + Uo)**2  + 1/4 * (Umm  - 4*Um + 3*Uo)**2
    beta_2 = 13/12 * (Um  - 2*Uo + Up)**2  + 1/4 * (Um   - Up)**2
    beta_3 = 13/12 * (Uo  - 2*Up + Upp)**2 + 1/4 * (3*Uo - 4*Up + Upp )**2

    # Linear Weights

    gamma_1 = 3/10
    gamma_2 = 3/5
    gamma_3 = 1/10

    epsilon = 1e-6

    # Alpha weights

    alpha_1 = gamma_1 / (epsilon + beta_1)**2
    alpha_2 = gamma_2 / (epsilon + beta_2)**2
    alpha_3 = gamma_3 / (epsilon + beta_3)**2

    alpha_sum = alpha_1 + alpha_2 + alpha_3

    # Non - Linear Stencil weights

    w_1 = alpha_1 / alpha_sum
    w_2 = alpha_2 / alpha_sum
    w_3 = alpha_3 / alpha_sum

    # Reconstructed variable 

    USouth =   w_1 * ( -Umm  + 5*Um + 2*Uo )/6 \
            + w_2 * ( 2*Um  + 5*Uo - Up   )/6 \
            + w_3 * ( 11*Uo - 7*Up + 2*Upp)/6

    return USouth