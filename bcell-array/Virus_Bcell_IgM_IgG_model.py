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
from mpl_toolkits.mplot3d.axes3d import Axes3D

import alva_machinery_VBIg as alva

AlvaFontSize = 23;
AlvaFigSize = (14, 6);
numberingFig = 0;

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12, 5))
plt.axis('off')
plt.title(r'$ Virus-Bcell-IgM-IgG \ equations \ (antibody-response \ for \ repeated-infection) $'
          , fontsize = AlvaFontSize)
plt.text(0, 7.0/8, r'$ \frac{\partial V_n(t)}{\partial t} = \
         +\mu_{v} V_{n}(t)(1 - \frac{V_n(t)}{V_{max}}) - \phi_{m} M_{n}(t) V_{n}(t) - \phi_{g} G_{n}(t) V_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 5.0/8, r'$ \frac{\partial B_n(t)}{\partial t} = \
         +\mu_{b} + (\beta_{m} + \beta_{g}) V_{n}(t) B_{n}(t) - \mu_{b} B_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/8,r'$ \frac{\partial M_n(t)}{\partial t} = \
         +\xi_{m} B_{n}(t) - \phi_{m} M_{n}(t) V_{n}(t) - \mu_{m} M_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/8,r'$ \frac{\partial G_n(t)}{\partial t} = \
         +\xi_{g} B_{n}(t) - \phi_{g} G_{n}(t) V_{n}(t) - \mu_{g} G_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.show()

# define the V-M-G partial differential equations
def dVdt_array(VBMGxt = [], *args):
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dV_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    for xn in range(x_totalPoint):
        dV_dt_array[xn] = +inRateV*V[xn]*(1 - V[xn]/maxV) - killRateVm*M[xn]*V[xn] - killRateVg*G[xn]*V[xn]
    return(dV_dt_array)

def dBdt_array(VBMGxt = [], *args):
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dB_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    for xn in range(x_totalPoint):
        dB_dt_array[xn] = +inRateB + (actRateBm + alva.actRateBg)*V[xn]*B[xn] - outRateB*B[xn]
    return(dB_dt_array)

def dMdt_array(VBMGxt = [], *args):
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dM_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    for xn in range(x_totalPoint):
        dM_dt_array[xn] = +inRateM*B[xn]*actRateBm/(actRateBm + alva.actRateBg) - consumeRateM*M[xn]*V[xn] - outRateM*M[xn]
    return(dM_dt_array)

def dGdt_array(VBMGxt = [], *args):
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dG_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    for xn in range(x_totalPoint):
        dG_dt_array[xn] = +inRateG*B[xn] - consumeRateG*G[xn]*V[xn] - outRateG*G[xn]
    return(dG_dt_array)

# <codecell>


# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24; 
    
maxV = float(1000) # max virus/milli-liter
inRateV = 6.5*maxV/10**4 # in-rate of virus
killRateVm = 1*maxV/10**5 # kill-rate of virus by antibody-IgM
killRateVg = killRateVm/1 # kill-rate of virus by antibody-IgG

inRateB = 3*maxV/10**4 # in-rate of B-cell
outRateB = inRateB # out-rate of B-cell
actRateBm = killRateVm # activation rate of naive B-cell
#actRateBg = killRateVg # activation rate of memory B-cell


inRateM = maxV/10**2  # in-rate of antibody-IgM from naive B-cell
outRateM = inRateM  # out-rate of antibody-IgM from naive B-cell
consumeRateM = killRateVm # consume-rate of antibody-IgM by cleaning virus

inRateG = inRateM/10 # in-rate of antibody-IgG from memory B-cell
outRateG = outRateM/100 # out-rate of antibody-IgG from memory B-cell
consumeRateG = consumeRateM  # consume-rate of antibody-IgG by cleaning virus

# time boundary and griding condition
minT = float(0); maxT = float(300*day);
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
gridV_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridB_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridM_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridG_array = np.zeros([totalGPoint_X, totalGPoint_T])
# initial output condition
gridV_array[0, 0] = float(1)
gridB_array[0, 0] = float(0)
gridM_array[0, 0] = float(0)
gridG_array[0, 0] = float(0)
# Runge Kutta numerical solution
pde_array = np.array([dVdt_array, dBdt_array, dMdt_array, dGdt_array])
startingOut_Value = np.array([gridV_array, gridB_array, gridM_array, gridG_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)
# plotting
gridV = gridOut_array[0]  
gridB = gridOut_array[1] 
gridM = gridOut_array[2]
gridG = gridOut_array[3]

numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.plot(gridT, gridM[i], color = 'blue', label = r'$ IgM_{%i}(t) $'%(i))
    plt.plot(gridT, gridG[i], color = 'green', label = r'$ IgG_{%i}(t) $'%(i))
    plt.plot(gridT, gridM[i] + gridG[i], color = 'gray', linewidth = 5.0, alpha = 0.5, linestyle = 'dashed'
             , label = r'$ IgM_{%i}(t) + IgG_{%i}(t) $'%(i, i))
    plt.grid(True)
    plt.title(r'$ Virus-IgM-IgG \ (immune \ response \ for \ repeated-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ unit/ml $', fontsize = AlvaFontSize);
    plt.legend(loc = (1,0))
#    plt.yscale('log')
    plt.show()

# <codecell>


numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gridT, gridV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.plot(gridT, gridM[i], color = 'blue', label = r'$ IgM_{%i}(t) $'%(i))
    plt.plot(gridT, gridG[i], color = 'green', label = r'$ IgG_{%i}(t) $'%(i))
    plt.plot(gridT, gridM[i] + gridG[i], color = 'gray', linewidth = 5.0, alpha = 0.5, linestyle = 'dashed'
             , label = r'$ IgM_{%i}(t) + IgG_{%i}(t) $'%(i, i))
    plt.grid(True)
    plt.title(r'$ Antibody-Virus \ (immune \ response \ for \ repeated-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Unit/ mL $', fontsize = AlvaFontSize);
    plt.legend(loc = (1,0))
    plt.ylim([1, 10000])
    plt.yscale('log')
    plt.show()



# <codecell>


