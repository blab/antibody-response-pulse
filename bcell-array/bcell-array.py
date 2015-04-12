# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Antibody Response Pulse
# https://github.com/blab/antibody-response-pulse
# 
# ### B-cells evolution --- cross-reactive antibody response after influenza virus infection or vaccination
# ### Adaptive immunity has memory --- primary and secondary immune response

# <codecell>

'''
author: Alvason Zhenhua Li
date:   04/09/2015
'''
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import time
import IPython.display as idisplay
from mpl_toolkits.mplot3d.axes3d import Axes3D

import alva_machinery as alva

AlvaFontSize = 23;
AlvaFigSize = (14, 6);
numberingFig = 0;

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,6))
plt.axis('off')
plt.title(r'$ Virion-Tcell-Bcell-Antibody \ response \ equations \ (primary-infection) $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6,r'$ \frac{\partial V_n(t)}{\partial t} = \
         +\rho V_n(t)(1 - \frac{V_n(t)}{\Omega}) - \phi_v A_{n}(t)V_{n}(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6,r'$ \frac{\partial T_n(t)}{\partial t} = \
         +\mu_t + \alpha_t V_n(t)T_{n}(t) - \mu_t T_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6,r'$ \frac{\partial B_n(t)}{\partial t} = \
         +\mu_b + \alpha_b V_{n}(t)T_{n}(t)B_{n}(t) - \mu_b B_n(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6,r'$ \frac{\partial A_n(t)}{\partial t} = \
         +\xi B_{n}(t) - \phi_a A_{n}(t)V_{n}(t) - \mu_a A_{n}(t) $', fontsize = 1.2*AlvaFontSize)
plt.show()

# define the partial differential equations
def dVdt_array(VCBAxt = [], *args):
    # naming
    V = VCBAxt[0]
    C = VCBAxt[1]
    B = VCBAxt[2]
    A = VCBAxt[3]
    x_totalPoint = VCBAxt.shape[1]
    # there are n dSdt
    dV_dt_array = np.zeros(x_totalPoint)
    # each dVdt with the same equation form
    for xn in range(x_totalPoint):
        dV_dt_array[xn] = +inRateV*V[xn]*(1 - V[xn]/totalV) - outRateV*A[xn]*V[xn]
    return(dV_dt_array)

def dCdt_array(VCBAxt = [], *args):
    # naming
    V = VCBAxt[0]
    C = VCBAxt[1]
    B = VCBAxt[2]
    A = VCBAxt[3]
    x_totalPoint = VCBAxt.shape[1]
    # there are n dSdt
    dC_dt_array = np.zeros(x_totalPoint)
    # each dCdt with the same equation form
    for xn in range(x_totalPoint):
        dC_dt_array[xn] = +inOutRateC + actRateC*V[xn]*C[xn] - inOutRateC*C[xn]
    return(dC_dt_array)

def dBdt_array(VCBAxt = [], *args):
    # naming
    V = VCBAxt[0]
    C = VCBAxt[1]
    B = VCBAxt[2]
    A = VCBAxt[3]
    x_totalPoint = VCBAxt.shape[1]
    # there are n dSdt
    dB_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dB_dt_array[xn] = +inOutRateB + actRateB*V[xn]*C[xn]*B[xn] - inOutRateB*B[xn]
    return(dB_dt_array)

def dAdt_array(VCBAxt = [], *args):
    # naming
    V = VCBAxt[0]
    C = VCBAxt[1]
    B = VCBAxt[2]
    A = VCBAxt[3]
    x_totalPoint = VCBAxt.shape[1]
    # there are n dSdt
    dA_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dA_dt_array[xn] = +inRateA*B[xn] - outRateAmV*A[xn]*V[xn] - outRateAm*A[xn]
    return(dA_dt_array)

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24; 
    
totalV = float(5000) # total virion/micro-liter
inRateV = float(0.16)/hour # intrinsic growth rate/hour
outRateV = float(1.67*10**(-4))/hour # virion clearance rate by IgM
inOutRateC = float(0.017)/hour # birth rate of CD4 T-cell
inOutRateB = float(0.037)/hour # birth rate of B-cell
inRateA = float(0.3)/hour # growth rate of antibody from B-cell (secretion)
outRateAm = float(0.014)/hour # out rate of Antibody IgM
outRateAmV = float(4.2*10**(-5))/hour # antibody IgM clearance rate by virus
actRateC = float(7.0*10**(-6))/hour # activation rate of CD4 T-cell
actRateB = float(6.0*10**(-7))/hour # activation rate of naive B-cell


# time boundary and griding condition
minT = float(0); maxT = float(2000*hour);
totalGPoint_T = int(10**4 + 1);
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

gridV_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridC_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridB_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridA_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridV_array[0:1, 0] = float(totalV)/10**3
gridC_array[0, 0] = float(0)
gridB_array[:, 0] = float(0)
gridA_array[:, 0] = float(1)

# Runge Kutta numerical solution
pde_array = np.array([dVdt_array, dCdt_array, dBdt_array, dAdt_array])
startingOut_Value = np.array([gridV_array, gridC_array, gridB_array, gridA_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridV = gridOut_array[0]  
gridC = gridOut_array[1]
gridB = gridOut_array[2]
gridA = gridOut_array[3]


firstT = np.copy(gridT)
firstV= np.copy(gridV)
firstA= np.copy(gridA)

numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridV[i], label = r'$ V_{%i}(t) $'%(i))
    plt.plot(gridT, gridC[i], label = r'$ C_{%i}(t) $'%(i))
    plt.plot(gridT, gridB[i], label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridA[i], label = r'$ A_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Virion-Tcell-Bcell-Antibody \ (immune \ response \ for \ primary-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.text(maxT, totalV*6.0/6, r'$ \Omega = %f $'%(totalV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*5.0/6, r'$ \phi = %f $'%(inRateV), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*4.0/6, r'$ \xi = %f $'%(inRateA), fontsize = AlvaFontSize)
    plt.text(maxT, totalV*3.0/6, r'$ \mu_b = %f $'%(inOutRateB), fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
    plt.show()

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24; 
    
totalV = float(5000) # total virion/micro-liter
inRateV = float(0.16)/hour # intrinsic growth rate/hour
outRateV = float(6.68*10**(-4))/hour # virion clearance rate by IgG
inOutRateC = float(0.017)/hour # birth rate of CD4 T-cell
inOutRateB = float(0.037)/hour # birth rate of B-cell
inRateA = float(0.3)/hour # growth rate of antibody from B-cell (secretion)
outRateAm = float(0.0048)/hour # out rate of Antibody IgG
outRateAmV = float(1.67*10**(-4))/hour # antibody IgG clearance rate by virus
actRateC = float(7.0*10**(-6))/hour # activation rate of CD4 T-cell
actRateB = float(0.0012)/hour # activation rate of memory B-cell


# time boundary and griding condition
minT = float(0); maxT = float(1000*hour);
totalGPoint_T = int(10**4 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

# space boundary and griding condition
minX = float(0); maxX = float(0);
totalGPoint_X = int(1 + 1);
gridX = np.linspace(minX, maxX, totalGPoint_X);
gridingX = np.linspace(minX, maxX, num = totalGPoint_X, retstep = True)
gridX = gridingX[0]
dx = gridingX[1]

gridV_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridC_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridB_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridA_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridV_array[0:1, 0] = float(totalV)/10**3
gridC_array[0, 0] = float(0)
gridB_array[:, 0] = float(0)
gridA_array[:, 0] = float(1)


# Runge Kutta numerical solution
pde_array = np.array([dVdt_array, dCdt_array, dBdt_array, dAdt_array])
startingOut_Value = np.array([gridV_array, gridC_array, gridB_array, gridA_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridV = gridOut_array[0]  
gridC = gridOut_array[1]
gridB = gridOut_array[2]
gridA = gridOut_array[3]

secondT= np.copy(gridT)
secondV= np.copy(gridV)
secondA= np.copy(gridA)

numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridV[i], label = r'$ V_{%i}(t) $'%(i))
    plt.plot(gridT, gridC[i], label = r'$ C_{%i}(t) $'%(i))
    plt.plot(gridT, gridB[i], label = r'$ B_{%i}(t) $'%(i))
    plt.plot(gridT, gridA[i], label = r'$ A_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ Virion-Tcell-Bcell-Antibody \ (immune \ response \ for \ secondary-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
    plt.legend(loc = (1,0))
    plt.show()

# <codecell>

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(firstT, firstA[0], label = r'$ 1st \ A(t) $', color = 'green', linewidth = 6.0, alpha = 0.3)
plt.plot(firstT, firstV[0], label = r'$ 1st \ V(t) $', color = 'red')
plt.plot(secondT + maxT, secondA[0], label = r'$ 2nd \ A(t)$', color = 'green', linewidth = 6.0, alpha = 0.6)
plt.plot(secondT + maxT, secondV[0], label = r'$ 2nd \ V(t)$', color = 'red')
plt.grid(True)
plt.title(r'$ Antibody \ (immune \ response \ for \ primary \ and \ secondary \ infections) $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.plot(firstT, np.log(firstA[0])/np.log(2), label = r'$ 1st \ A(t) $', color = 'green', linewidth = 6.0, alpha = 0.3)
#plt.plot(firstT[1:], np.log(firstV[0, 1:])/np.log(2), label = r'$ 1st \ V(t) $', color = 'red')
plt.plot(secondT + maxT, np.log(secondA[0])/np.log(2), label = r'$ 2nd \ A(t)$', color = 'green', linewidth = 6.0, alpha = 0.6)
#plt.plot(secondT[1:] + maxT, np.log(secondV[0, 1:])/np.log(2), label = r'$ 2nd \ V(t)$', color = 'red')
plt.grid(True)
plt.title(r'$ Antibody \ (immune \ response \ for \ primary \ and \ secondary \ infections) $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
plt.ylabel(r'$ log_2(Cells/ \mu L) $', fontsize = AlvaFontSize);
plt.legend(loc = (1,0))
plt.show()

# <codecell>


