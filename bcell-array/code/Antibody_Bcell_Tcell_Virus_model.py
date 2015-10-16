# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Antibody Response Pulse
# https://github.com/blab/antibody-response-pulse
# 
# ### B-cells evolution --- cross-reactive antibody response after influenza virus infection or vaccination
# ### Adaptive immune response for repeated infection

# <codecell>

'''
author: Alvason Zhenhua Li
date:   04/09/2015
'''
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

AlvaFontSize = 23;
AlvaFigSize = (14, 6);
numberingFig = 0;

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24;  
    
###############
numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,6))
plt.axis('off')
plt.title(r'$ Antibody-Bcell-Tcell-Virus \ response \ equations \ (long-term-infection) $'
          , fontsize = AlvaFontSize)
plt.text(0, 5.0/6,r'$ \frac{\partial A_n(t)}{\partial t} = \
         +\mu_a B_{n}(t) - (\phi_{ma} + \phi_{ga})A_{n}(t)V_{n}(t) - (\mu_{ma} + \mu_{ga})A_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6,r'$ \frac{\partial B_n(t)}{\partial t} = \
         +\mu_b + (\alpha_{bn} + \alpha_{bm}) V_{n}(t)C_{n}(t)B_{n}(t) - \mu_b B_n(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6,r'$ \frac{\partial C_n(t)}{\partial t} = \
         +\mu_c + \alpha_c V_n(t)C_{n}(t) - \mu_c C_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6,r'$ \frac{\partial V_n(t)}{\partial t} = \
         +\rho V_n(t)(1 - \frac{V_n(t)}{V_{max}}) - (\phi_{mv} + \phi_{gv}) A_{n}(t)V_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.show()

# define the partial differential equations
def dAdt_array(ABCVxt = [], *args):
    # naming
    A = ABCVxt[0]
    B = ABCVxt[1]
    C = ABCVxt[2]
    V = ABCVxt[3]
    x_totalPoint = ABCVxt.shape[1]
    # there are n dSdt
    dA_dt_array = np.zeros(x_totalPoint)
    # each dVdt with the same equation form
    for xn in range(x_totalPoint):
        dA_dt_array[xn] = +inRateA*B[xn] - (outRateAmV + outRateAgV)*A[xn]*V[xn] - (outRateAm + outRateAg)*A[xn]
    return(dA_dt_array)

def dBdt_array(ABCVxt = [], *args):
    # naming
    A = ABCVxt[0]
    B = ABCVxt[1]
    C = ABCVxt[2]
    V = ABCVxt[3]
    x_totalPoint = ABCVxt.shape[1]
    # there are n dSdt
    dB_dt_array = np.zeros(x_totalPoint)
    # each dCdt with the same equation form
    for xn in range(x_totalPoint):
        dB_dt_array[xn] = +inOutRateB + (actRateB_naive + actRateB_memory)*V[xn]*C[xn]*B[xn] - inOutRateB*B[xn]
    return(dB_dt_array)

def dCdt_array(ABCVxt = [], *args):
    # naming
    A = ABCVxt[0]
    B = ABCVxt[1]
    C = ABCVxt[2]
    V = ABCVxt[3]
    x_totalPoint = ABCVxt.shape[1]
    # there are n dSdt
    dC_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dC_dt_array[xn] = +inOutRateC + actRateC*V[xn]*C[xn] - inOutRateC*C[xn]
    return(dC_dt_array)

def dVdt_array(ABCVxt = [], *args):
    # naming
    A = ABCVxt[0]
    B = ABCVxt[1]
    C = ABCVxt[2]
    V = ABCVxt[3]
    x_totalPoint = ABCVxt.shape[1]
    # there are n dSdt
    dV_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dV_dt_array[xn] = +inRateV*V[xn]*(1 - V[xn]/totalV) - (outRateVg + outRateVm)*A[xn]*V[xn]
    return(dV_dt_array)

# define RK4 for an array (3, n) of coupled differential equations
def AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX_In, maxX_In, totalGPoint_X, minT_In, maxT_In, totalGPoint_T):
    global actRateB_memory
    # primary size of pde equations
    outWay = pde_array.shape[0]
    # initialize the whole memory-space for output and input
    inWay = 1; # one layer is enough for storing "x" and "t" (only two list of variable)
    # define the first part of array as output memory-space
    gridOutIn_array = np.zeros([outWay + inWay, totalGPoint_X, totalGPoint_T])
    # loading starting output values
    for i in range(outWay):
        gridOutIn_array[i, :, :] = startingOut_Value[i, :, :]
    # griding input X value  
    gridingInput_X = np.linspace(minX_In, maxX_In, num = totalGPoint_X, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gridOutIn_array[-inWay, :, 0] = gridingInput_X[0]
    # step-size (increment of input X)
    dx = gridingInput_X[1]
    # griding input T value  
    gridingInput_T = np.linspace(minT_In, maxT_In, num = totalGPoint_T, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gridOutIn_array[-inWay, 0, :] = gridingInput_T[0]
    # step-size (increment of input T)
    dt = gridingInput_T[1]
    # starting
    # initialize the memory-space for local try-step 
    dydt1_array = np.zeros([outWay, totalGPoint_X])
    dydt2_array = np.zeros([outWay, totalGPoint_X])
    dydt3_array = np.zeros([outWay, totalGPoint_X])
    dydt4_array = np.zeros([outWay, totalGPoint_X])
    # initialize the memory-space for keeping current value
    currentOut_Value = np.zeros([outWay, totalGPoint_X])
    for tn in range(totalGPoint_T - 1):
        actRateB_memory = 0
        tn_unit = totalGPoint_T/(maxT_In - minT_In)
        if tn > int(14*day*tn_unit):
            actRateB_memory = float(0.01)*24     
        # setting virus1 = 0 if virus1 < 1
        if gridOutIn_array[3, 0, tn] < 1.0:
            gridOutIn_array[3, 0, tn] = 0.0
        ## 2nd infection
        if tn == int(20*day*tn_unit):
            gridOutIn_array[3, 0, tn] = 1.0 # virus infection
        ### 3rd infection
        if tn == int(2*20*day*tn_unit):
            gridOutIn_array[3, 0, tn] = 1.0 # virus infection
        ### 4rd infection
        if tn == int(50*day*tn_unit):
            gridOutIn_array[3, 0, tn] = 1.0 # virus infection
        # keep initial value at the moment of tn
        currentOut_Value[:, :] = np.copy(gridOutIn_array[:-inWay, :, tn])
        currentIn_T_Value = np.copy(gridOutIn_array[-inWay, 0, tn])
        # first try-step
        for i in range(outWay):
            for xn in range(totalGPoint_X):
                dydt1_array[i, xn] = pde_array[i](gridOutIn_array[:, :, tn])[xn] # computing ratio   
        gridOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt1_array[:, :]*dt/2 # update output
        gridOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # second half try-step
        for i in range(outWay):
            for xn in range(totalGPoint_X):
                dydt2_array[i, xn] = pde_array[i](gridOutIn_array[:, :, tn])[xn] # computing ratio   
        gridOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt2_array[:, :]*dt/2 # update output
        gridOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # third half try-step
        for i in range(outWay):
            for xn in range(totalGPoint_X):
                dydt3_array[i, xn] = pde_array[i](gridOutIn_array[:, :, tn])[xn] # computing ratio   
        gridOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt3_array[:, :]*dt # update output
        gridOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt # update input
        # fourth try-step
        for i in range(outWay):
            for xn in range(totalGPoint_X):
                dydt4_array[i, xn] = pde_array[i](gridOutIn_array[:, :, tn])[xn] # computing ratio 
        # solid step (update the next output) by accumulate all the try-steps with proper adjustment
        gridOutIn_array[:-inWay, :, tn + 1] = currentOut_Value[:, :] + dt*(dydt1_array[:, :]/6 
                                                                                      + dydt2_array[:, :]/3 
                                                                                      + dydt3_array[:, :]/3 
                                                                                      + dydt4_array[:, :]/6)
        # restore to initial value
        gridOutIn_array[:-inWay, :, tn] = np.copy(currentOut_Value[:, :])
        gridOutIn_array[-inWay, 0, tn] = np.copy(currentIn_T_Value)
        # end of loop
    return (gridOutIn_array[:-inWay, :])
##############

inRateA = float(0.3)/hour # growth rate of antibody from B-cell (secretion)
outRateAm = float(0.014)/hour # out rate of Antibody IgM
outRateAg = float(0.048)/hour # out rate of Antibody IgG
outRateAmV = float(4.2*10**(-5))/hour # antibody IgM clearance rate by virus
outRateAgV = float(1.67*10**(-4))/hour # antibody IgG clearance rate by virus

inOutRateB = float(0.037)/hour # birth rate of B-cell
actRateB_naive = float(6.0*10**(-7))/hour # activation rate of naive B-cell
#actRateB_memory = 0*float(0.0012)/hour # activation rate of memory B-cell

inOutRateC = float(0.017)/hour # birth rate of CD4 T-cell
actRateC = float(7.0*10**(-6))/hour # activation rate of CD4 T-cell

totalV = float(5000) # total virion/micro-liter
inRateV = float(0.16)/hour # intrinsic growth rate/hour
outRateVm = float(1.67*10**(-4))/hour # virion clearance rate by IgM
outRateVg = float(6.68*10**(-4))/hour # virion clearance rate by IgG


# time boundary and griding condition
minT = float(0); maxT = float(3*20*day);
totalGPoint_T = int(1*10**4 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

# space boundary and griding condition
minX = float(0); maxX = float(1);
totalGPoint_X = int(1 + 1);
gridX = np.linspace(minX, maxX, totalGPoint_X);
gridingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = gridingX[0]
dx = gridingX[1]

gridA_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridB_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridC_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridV_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridA_array[:, 0] = float(0)
gridB_array[:, 0] = float(0)
gridC_array[0, 0] = float(0)
gridV_array[0, 0] = float(totalV)/10**3

# Runge Kutta numerical solution
pde_array = np.array([dAdt_array, dBdt_array, dCdt_array, dVdt_array])
startingOut_Value = np.array([gridA_array, gridB_array, gridC_array, gridV_array])
gridOut_array = AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridA = gridOut_array[0]  
gridB = gridOut_array[1]
gridC = gridOut_array[2]
gridV = gridOut_array[3]


numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridA[i], color = 'green', label = r'$ A_{%i}(t) $'%(i))
    plt.plot(gridT, gridB[i], color = 'blue', label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridC[i], color = 'gray', label = r'$ C_{%i}(t) $'%(i))
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Antibody-Bcell-Tcell-Virus \ (immune \ response \ for \ primary-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.text(maxT, totalV*6.0/6, r'$ \Omega = %f $'%(totalV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*5.0/6, r'$ \phi = %f $'%(inRateV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*4.0/6, r'$ \xi = %f $'%(inRateA), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*3.0/6, r'$ \mu_b = %f $'%(inOutRateB), fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
#    plt.yscale('log')
    plt.show()

# <codecell>


