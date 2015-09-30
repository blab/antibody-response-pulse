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
import os

import alva_machinery_event_OAS as alva

AlvaFontSize = 23
AlvaFigSize = (15, 5)
numberingFig = 0

# plotting
dir_path = '/Users/al/Desktop/GitHub/antibody-response-pulse/bcell-array/figure'
file_name = 'Virus-Bcell-IgM-IgG'
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(12, 5))
plt.axis('off')
plt.title(r'$ Virus-Bcell-IgM-IgG \ equations \ (antibody-response \ for \ repeated-infection) $'
          , fontsize = AlvaFontSize)
plt.text(0, 7.0/9, r'$ \frac{\partial V_n(t)}{\partial t} = \
         +\mu_{v} V_{n}(t)(1 - \frac{V_n(t)}{V_{max}}) - \phi_{m} M_{n}(t) V_{n}(t) - \phi_{g} G_{n}(t) V_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 5.0/9, r'$ \frac{\partial B_n(t)}{\partial t} = \
         +\mu_{b} + (\beta_{m} + \beta_{g}) V_{n}(t) B_{n}(t) - \mu_{b} B_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/9,r'$ \frac{\partial M_n(t)}{\partial t} = \
         +\xi_{m} B_{n}(t) - \phi_{m} M_{n}(t) V_{n}(t) - \mu_{m} M_{n}(t) $'
         , fontsize = 1.2*AlvaFontSize)
plt.text(0, 1.0/9,r'$ \frac{\partial G_n(t)}{\partial t} = \
         +\xi_{g} B_{n}(t) - \phi_{g} G_{n}(t) V_{n}(t) - \mu_{g} G_{n}(t) \
         + m \frac{G_{i-1}(t) - 2G_i(t) + G_{i+1}(t)}{(\Delta i)^2} $'
         , fontsize = 1.2*AlvaFontSize)

plt.savefig(save_figure, dpi = 100)
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
    dV_dt_array[:] = +inRateV*V[:]*(1 - V[:]/maxV) - killRateVm*M[:]*V[:] - killRateVg*G[:]*V[:]
    return(dV_dt_array)

def dBdt_array(VBMGxt = [], *args):
    eName = np.zeros([totalPoint_X])
    eName[0] = 1.0/100 
    eName[1] = alva.eventName
#    eName[2] = 1.0/100 
#    eName[3] = 1.0/100
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dB_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    dB_dt_array[:] = +inRateB + (actRateBm + eName[:])*V[:]*B[:] - outRateB*B[:]
    return(dB_dt_array)

def dMdt_array(VBMGxt = [], *args):
    eName = np.zeros([totalPoint_X])
    eName[0] = 1.0/100 
    eName[1] = alva.eventName
#    eName[2] = 1.0/100 
#    eName[3] = 1.0/100
    # naming
    V = VBMGxt[0]
    B = VBMGxt[1]
    M = VBMGxt[2]
    G = VBMGxt[3]
    x_totalPoint = VBMGxt.shape[1]
    # there are n dSdt
    dM_dt_array = np.zeros(x_totalPoint)
    # each dSdt with the same equation form
    dM_dt_array[:] = +inRateM*B[:]*actRateBm/(actRateBm + eName[:]) - consumeRateM*M[:]*V[:] - outRateM*M[:]
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
    Gcopy = np.copy(G)
    centerX = Gcopy[:]
    leftX = np.roll(Gcopy[:], 1)
    rightX = np.roll(Gcopy[:], -1)
    leftX[0] = centerX[0]
    rightX[-1] = centerX[-1]
    dG_dt_array[:] = +inRateG*B[:] - consumeRateG*G[:]*V[:] - outRateG*G[:] \
                     + mutatRate*(leftX[:] - 2*centerX[:] + rightX[:])/(dx**2)
    return(dG_dt_array)

# <codecell>

# setting parameter
timeUnit = 'day'
if timeUnit == 'hour':
    hour = float(1)
    day = float(24)
elif timeUnit == 'day':
    day = float(1)
    hour = float(1)/24 
elif timeUnit == 'year':
    year = float(1)
    day = float(1)/365
    hour = float(1)/24/365 
    
maxV = float(1000) # max virus/milli-liter
inRateV = 6.5*maxV/10**4/day # in-rate of virus
killRateVm = 1*maxV/10**5/day # kill-rate of virus by antibody-IgM
killRateVg = killRateVm # kill-rate of virus by antibody-IgG

inRateB = 3*maxV/10**4/day # in-rate of B-cell
outRateB = inRateB # out-rate of B-cell
actRateBm = killRateVm # activation rate of naive B-cell
#actRateBg = float(1)/10**2 # activation rate of memory B-cell

inRateM = maxV/10**2/day  # in-rate of antibody-IgM from naive B-cell
outRateM = inRateM  # out-rate of antibody-IgM from naive B-cell
consumeRateM = killRateVm # consume-rate of antibody-IgM by cleaning virus

inRateG = inRateM/20 # in-rate of antibody-IgG from memory B-cell
outRateG = outRateM/600 # out-rate of antibody-IgG from memory B-cell
consumeRateG = consumeRateM  # consume-rate of antibody-IgG by cleaning virus
mutatRate = float(1)/60/day # mutation rate
# time boundary and griding condition
minT = float(0)
maxT = float(2*28*day)
totalPoint_T = int(3*10**3 + 1)
gT = np.linspace(minT, maxT, totalPoint_T)
spacingT = np.linspace(minT, maxT, num = totalPoint_T, retstep = True)
gT = spacingT[0]
dt = spacingT[1]

# space boundary and griding condition
minX = float(0)
maxX = float(3)
totalPoint_X = int(maxX - minX + 1)
gX = np.linspace(minX, maxX, totalPoint_X)
gridingX = np.linspace(minX, maxX, num = totalPoint_X, retstep = True)
gX = gridingX[0]
dx = gridingX[1]
gV_array = np.zeros([totalPoint_X, totalPoint_T])
gB_array = np.zeros([totalPoint_X, totalPoint_T])
gM_array = np.zeros([totalPoint_X, totalPoint_T])
gG_array = np.zeros([totalPoint_X, totalPoint_T])
# initial output condition
gV_array[1, 0] = float(1) # set the first infection with one Virus-1
gB_array[:, 0] = float(0)
gM_array[0, 0] = float(0)
gG_array[0, 0] = float(0)

event_tn_In = np.array([[0*day, 1/10**2/day], [14*day, 1/1/day]])

# Runge Kutta numerical solution
pde_array = np.array([dVdt_array, dBdt_array, dMdt_array, dGdt_array])
initial_Out = np.array([gV_array, gB_array, gM_array, gG_array])
gOut_array = alva.AlvaRungeKutta4XT(pde_array, initial_Out, minX, maxX, totalPoint_X, minT, maxT, totalPoint_T, event_tn_In)

# plotting
gV = gOut_array[0]  
gB = gOut_array[1] 
gM = gOut_array[2]
gG = gOut_array[3]

numberingFig = numberingFig + 1
for i in range(totalPoint_X):
    figure_name = '-response-%i'%(i)
    figure_suffix = '.png'
    save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)
    plt.figure(numberingFig, figsize = AlvaFigSize)
    plt.plot(gT, gV[i], color = 'red', label = r'$ V_{%i}(t) $'%(i))
    plt.plot(gT, gM[i], color = 'blue', label = r'$ IgM_{%i}(t) $'%(i))
    plt.plot(gT, gG[i], color = 'green', label = r'$ IgG_{%i}(t) $'%(i))
    plt.plot(gT, gM[i] + gG[i], color = 'gray', linewidth = 5.0, alpha = 0.5, linestyle = 'dashed'
             , label = r'$ IgM_{%i}(t) + IgG_{%i}(t) $'%(i, i))
    plt.grid(True, which = 'both')
    plt.title(r'$ Antibody \ from \ Virus-{%i} $'%(i), fontsize = AlvaFontSize)
    plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
    plt.ylabel(r'$ Neutralization \ \ titer $', fontsize = AlvaFontSize)
    plt.xlim([minT, maxT])
    plt.xticks(fontsize = AlvaFontSize*0.6)
    plt.yticks(fontsize = AlvaFontSize*0.6) 
    plt.ylim([2**0, 2**11])
    plt.yscale('log', basey = 2)
    plt.legend(loc = (1,0), fontsize = AlvaFontSize)
    plt.savefig(save_figure, dpi = 100)
    plt.show()

# <codecell>

# Experimental lab data from OAS paper
gT_lab = np.array([28, 28 + 7, 28 + 14, 28 + 28])
gPR8_lab = np.array([2**(7 + 1.0/2), 2**9, 2**(9 + 1.0/3), 2**(9 - 1.0/4)]) 
gFM1_lab = np.array([2**(6 + 2.0/5), 2**(7 - 1.0/4), 2**(7 + 1.0/4), 2**(8 - 1.0/4)])
# Sequential immunization graph
figure_name = '-Original-Antigenic-Sin'
figure_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)
numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize = (6, 6))
plt.plot(gT, gM[1] + gG[1], linewidth = 5.0, alpha = 0.5, color = 'gray'
         , label = r'$ Origin-virus $')
plt.plot(gT, gM[2] + gG[2], linewidth = 5.0, alpha = 0.5, color = 'red'
         , label = r'$ Subsequence-virus $')

plt.title(r'$ Original \ Antigenic \ Sin $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
plt.ylabel(r'$ Neutralization \ \ titer $', fontsize = AlvaFontSize)
plt.xlim([28, maxT])
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.ylim([2**5, 2**11])
plt.yscale('log', basey = 2)
plt.legend(loc = (1, 0), fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 100, bbox_inches='tight')
plt.show()

# <codecell>

# Normalization stacked graph
numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize = AlvaFigSize)
plt.stackplot(gT, gM + gG, alpha = 0.3)
plt.title(r'$ Stacked-graph \ of \ Antibody $', fontsize = AlvaFontSize)
plt.xlabel(r'$time \ (%s)$'%(timeUnit), fontsize = AlvaFontSize)
plt.ylabel(r'$ Neutralization \ \ titer $', fontsize = AlvaFontSize)
plt.ylim([2**0, 2**12])
plt.yscale('log', basey = 2)
plt.show()

# <codecell>


