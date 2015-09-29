# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Antibody Response Pulse
# https://github.com/alvason/antibody-response-pulse
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
from mpl_toolkits.mplot3d.axes3d import Axes3D

import alva_machinery as alva

AlvaFontSize = 23;
AlvaFigSize = (14, 6);
numberingFig = 0;

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,6))
plt.axis('off')
plt.title(r'$ Antibody-Bcell-Virus \ response \ equations \ (long-term-infection) $'
          , fontsize = AlvaFontSize)
plt.text(0, 5.0/6,r'$ \frac{\partial A_n(t)}{\partial t} = \
         +\mu_a B_{n}(t) - (\phi_{ma} + \phi_{ga})A_{n}(t)V_{n}(t) - (\mu_{ma} + \mu_{ga})A_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6,r'$ \frac{\partial B_n(t)}{\partial t} = \
         +\mu_b + (\alpha_{bn} + \alpha_{bm}) V_{n}(t)B_{n}(t) - \mu_b B_n(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6,r'$ \frac{\partial V_n(t)}{\partial t} = \
         +\rho V_n(t)(1 - \frac{V_n(t)}{V_{max}}) - (\phi_{mv} + \phi_{gv}) A_{n}(t)V_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.show()

# define the partial differential equations
def dAdt_array(ABVxt = [], *args):
    # naming
    A = ABVxt[0]
    B = ABVxt[1]
    V = ABVxt[2]
    x_totalPoint = ABVxt.shape[1]
    # there are n dSdt
    dA_dt_array = np.zeros(x_totalPoint)
    # each dVdt with the same equation form
    for xn in range(x_totalPoint):
        dA_dt_array[xn] = +inRateA*B[xn] - (outRateAmV + outRateAgV)*A[xn]*V[xn] - (outRateAm + outRateAg)*A[xn]
    return(dA_dt_array)

def dBdt_array(ABVxt = [], *args):
    # naming
    A = ABVxt[0]
    B = ABVxt[1]
    V = ABVxt[2]
    x_totalPoint = ABVxt.shape[1]
    # there are n dSdt
    dB_dt_array = np.zeros(x_totalPoint)
    # each dCdt with the same equation form
    for xn in range(x_totalPoint):
        dB_dt_array[xn] = +inOutRateB + (actRateB_naive + alva.actRateB_memory)*V[xn]*B[xn] - inOutRateB*B[xn]
    return(dB_dt_array)

def dVdt_array(ABVxt = [], *args):
    # naming
    A = ABVxt[0]
    B = ABVxt[1]
    V = ABVxt[2]
    x_totalPoint = ABVxt.shape[1]
    # there are n dSdt
    dV_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dV_dt_array[xn] = +inRateV*V[xn]*(1 - V[xn]/totalV) - (outRateVg + outRateVm)*A[xn]*V[xn]
    return(dV_dt_array)

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24; 
 
inRateA = float(0.3)/hour # growth rate of antibody from B-cell (secretion)
outRateAm = float(0.014)/hour # out rate of Antibody IgM
outRateAg = float(0.048)/hour # out rate of Antibody IgG
outRateAmV = float(4.2*10**(-5))/hour # antibody IgM clearance rate by virus
outRateAgV = float(1.67*10**(-4))/hour # antibody IgG clearance rate by virus

inOutRateB = float(0.037)/hour # birth rate of B-cell
actRateB_naive = float(6.0*10**(-5))/hour # activation rate of naive B-cell
#actRateB_memory = 0*float(0.0012)/hour # activation rate of memory B-cell

inOutRateC = float(0.017)/hour # birth rate of CD4 T-cell
actRateC = float(7.0*10**(-6))/hour # activation rate of CD4 T-cell

totalV = float(5000) # total virion/micro-liter
inRateV = float(0.16)/hour # intrinsic growth rate/hour
outRateVm = float(1.67*10**(-4))/hour # virion clearance rate by IgM
outRateVg = float(6.68*10**(-4))/hour # virion clearance rate by IgG


# time boundary and griding condition
minT = float(0); maxT = float(60*day);
totalGPoint_T = int(2*10**4 + 1);
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
gridV_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridA_array[:, 0] = float(0)
gridB_array[:, 0] = float(0)
gridV_array[0, 0] = float(totalV)/10**3

# Runge Kutta numerical solution
pde_array = np.array([dAdt_array, dBdt_array, dVdt_array])
startingOut_Value = np.array([gridA_array, gridB_array, gridV_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridA = gridOut_array[0]  
gridB = gridOut_array[1]
gridV = gridOut_array[2]


numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridA[i], color = 'green', label = r'$ A_{%i}(t) $'%(i))
    plt.plot(gridT, gridB[i], color = 'blue', label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Antibody-Bcell-Tcell-Virus \ (immune \ response \ for \ primary-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.text(maxT, totalV*6.0/6, r'$ \Omega = %f $'%(totalV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*5.0/6, r'$ \phi = %f $'%(inRateV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*3.0/6, r'$ \mu_b = %f $'%(inOutRateB), fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
#    plt.yscale('log')
    plt.show()

# <codecell>

# plotting
gridA = gridOut_array[0]  
gridB = gridOut_array[1]
gridV = gridOut_array[2]


numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridA[i], color = 'green', label = r'$ A_{%i}(t) $'%(i))
    plt.plot(gridT, gridB[i], color = 'blue', label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Antibody-Bcell-Tcell-Virus \ (immune \ response \ for \ repeated-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.text(maxT, totalV*6.0/6, r'$ \Omega = %f $'%(totalV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*5.0/6, r'$ \phi = %f $'%(inRateV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*3.0/6, r'$ \mu_b = %f $'%(inOutRateB), fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
#    plt.yscale('log')
    plt.show()

numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridA[i], color = 'green', label = r'$ A_{%i}(t) $'%(i))
#    plt.plot(gridT, gridB[i], color = 'blue', label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Antibody-Virus \ (immune \ response \ for \ repeated-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.legend(loc = (1,0))
    plt.ylim([10, 10000])
    plt.yscale('log')
    plt.show()


# <codecell>


