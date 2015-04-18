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

import alva_machinery as alva

AlvaFontSize = 23;
AlvaFigSize = (14, 6);
numberingFig = 0;

numberingFig = numberingFig + 1;
plt.figure(numberingFig, figsize=(12,6))
plt.axis('off')
plt.title(r'$ Susceptible-Infected-Antibody-Bcell \ response \ equations \ (primary-infection) $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6,r'$ \frac{\partial S_n(t)}{\partial t} = \
         -\beta \rho I_n(t)S_n(t) $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6,r'$ \frac{\partial I_n(t)}{\partial t} = \
         +\beta \rho I_n(t)S_n(t) - \alpha I_n(t)A_{n}(t) - \xi I_n(t)C_n(t)  $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6,r'$ \frac{\partial A_n(t)}{\partial t} = \
         +\phi_a I_n(t) - \mu_a A_n(t)\ $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/6,r'$ \frac{\partial C_n(t)}{\partial t} = \
         +\phi_c I_n(t)C_n(t) - \mu_c C_n(t)\ $', fontsize = 1.2*AlvaFontSize)
plt.show()

# define the partial differential equations
def dSdt_array(SIACxt = [], *args):
    # naming
    S = SIACxt[0]
    I = SIACxt[1]
    A = SIACxt[2]
    C = SIACxt[3]
    x_totalPoint = SIACxt.shape[1]
    # there are n dSdt
    dS_dt_array = np.zeros(x_totalPoint)
    # each dVdt with the same equation form
    for xn in range(x_totalPoint):
        dS_dt_array[xn] = -infecRate*inRateV*I[xn]*S[xn]
    return(dS_dt_array)

def dIdt_array(SIACxt = [], *args):
    # naming
    S = SIACxt[0]
    I = SIACxt[1]
    A = SIACxt[2]
    C = SIACxt[3]
    x_totalPoint = SIACxt.shape[1]
    # there are n dSdt
    dI_dt_array = np.zeros(x_totalPoint)
    # each dCdt with the same equation form
    for xn in range(x_totalPoint):
        dI_dt_array[xn] = +infecRate*inRateV*I[xn]*S[xn] - killRateA*I[xn]*A[xn] - killRateC*I[xn]*C[xn]
    return(dI_dt_array)

def dAdt_array(SIACxt = [], *args):
    # naming
    S = SIACxt[0]
    I = SIACxt[1]
    A = SIACxt[2]
    C = SIACxt[3]
    x_totalPoint = SIACxt.shape[1]
    # there are n dSdt
    dA_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dA_dt_array[xn] = +inRateA*I[xn] - outRateA*A[xn]
    return(dA_dt_array)

def dCdt_array(SIACxt = [], *args):
    # naming
    S = SIACxt[0]
    I = SIACxt[1]
    A = SIACxt[2]
    C = SIACxt[3]
    x_totalPoint = SIACxt.shape[1]
    # there are n dSdt
    dC_dt_array = np.zeros(x_totalPoint)
    # each dTdt with the same equation form
    for xn in range(x_totalPoint):
        dC_dt_array[xn] = +inRateC*I[xn]*C[xn] - outRateC*C[xn]
    return(dC_dt_array)

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1); day = float(24); 
elif timeUnit == 'day':
    day = float(1); hour = float(1)/24; 
    
infecRate = 1.85*10**(-10)
inRateV = 1.1*10**5
killRateA = 5.74*10**(-4)
killRateC = 3*10**(-5)
inRateA = 0.52
outRateA = 0.07
inRateC = 1.2*10**(-4)
outRateC = 0.1



# time boundary and griding condition
minT = float(0); maxT = float(300*day);
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

gridS_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridI_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridA_array = np.zeros([totalGPoint_X, totalGPoint_T])
gridC_array = np.zeros([totalGPoint_X, totalGPoint_T])

# initial output condition
gridS_array[0:1, 0] = float(3.5*10**5)
gridI_array[0, 0] = float(8.62*10**(-18))
gridA_array[:, 0] = float(0)
gridC_array[:, 0] = float(1000)

# Runge Kutta numerical solution
pde_array = np.array([dSdt_array, dIdt_array, dAdt_array, dCdt_array])
startingOut_Value = np.array([gridS_array, gridI_array, gridA_array, gridC_array])
gridOut_array = alva.AlvaRungeKutta4ArrayXT(pde_array, startingOut_Value, minX, maxX, totalGPoint_X, minT, maxT, totalGPoint_T)

# plotting
gridS = gridOut_array[0]  
gridI = gridOut_array[1]
gridA = gridOut_array[2]
gridC = gridOut_array[3]


numberingFig = numberingFig + 1;
for i in range(1):
    plt.figure(numberingFig, figsize = AlvaFigSize)
 #   plt.plot(gridT, gridS[i], label = r'$ S_{%i}(t) $'%(i))
    plt.plot(gridT, gridI[i], label = r'$ I_{%i}(t) $'%(i))
    plt.plot(gridT, gridA[i], label = r'$ A_{%i}(t) $'%(i))
    plt.plot(gridT, gridC[i], label = r'$ C_{%i}(t) $'%(i))
    plt.grid(True)
    plt.title(r'$ S-I-Antibody-C \ (immune \ response \ for \ primary-infection) $', fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize);
    plt.ylabel(r'$ Cells/ \mu L $', fontsize = AlvaFontSize)
    plt.legend(loc = (1,0))
 #   plt.yscale('log')
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

minT = float(0); maxT = float(100);
totalGPoint_T = int(10**4 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

a = 0.2
c = 1
plt.figure(figsize = AlvaFigSize)
plt.plot(gridT, 1 + np.exp(-(gridT-1)**2) - np.log((1 - np.pi)*(c + gridT)**(-a) + np.pi))
plt.grid(True)
plt.show()

# <codecell>

minT = float(0); maxT = float(20);
totalGPoint_T = int(10**4 + 1);
gridT = np.linspace(minT, maxT, totalGPoint_T);
spacingT = np.linspace(minT, maxT, num = totalGPoint_T, retstep = True)
gridT = spacingT[0]
dt = spacingT[1]

r = 1
saturation = np.exp(r*gridT)/(1 + np.exp(r*gridT))
aaa =np.cos(gridT)/3 + saturation
plt.figure(figsize = AlvaFigSize)
plt.plot(gridT, aaa)
plt.grid(True)
#plt.yscale('log')
plt.show()

# <codecell>


