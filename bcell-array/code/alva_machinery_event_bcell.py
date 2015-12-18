
# coding: utf-8

# # Infectious Pulse
# https://github.com/blab/antibody-response-pulse/
# 
# ## Homemade machinery for solving partial differential equations
# ### Runge-kutta algorithm for a array of coupled partial differential equation --- Bcell events

# In[1]:

'''
author: Alvason Zhenhua Li
date:   03/23/2015

Home-made machinery for solving partial differential equations --- Bcell events
'''
import numpy as np

# define RK4 for an array (3, n) of coupled differential equations
def AlvaRungeKutta4XT(pde_array, initial_Out, minX_In, maxX_In, totalPoint_X, minT_In, maxT_In, totalPoint_T, event_table):
    global event_recovered; event_recovered = 0.0
    global event_OAS_boost; event_OAS_boost = 0.0
    global event_OAS_press; event_OAS_press = 0.0
    global event_recoveredV; event_recoveredV = 0.0
    global event_OAS_boostV; event_OAS_boostV = 0.0
    global event_OAS_pressV; event_OAS_pressV = 0.0
    global event_OAS_slowV; event_OAS_slowV = 0.0 
    # primary size of pde equations
    outWay = pde_array.shape[0]
    # initialize the whole memory-space for output and input
    inWay = 1; # one layer is enough for storing "x" and "t" (only two list of variable)
    # define the first part of array as output memory-space
    gOutIn_array = np.zeros([outWay + inWay, totalPoint_X, totalPoint_T])
    # loading starting output values
    for i in range(outWay):
        gOutIn_array[i, :, :] = initial_Out[i, :, :]
    # griding input X value  
    gridingInput_X = np.linspace(minX_In, maxX_In, num = totalPoint_X, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gOutIn_array[-inWay, :, 0] = gridingInput_X[0]
    # step-size (increment of input X)
    dx = gridingInput_X[1]
    # griding input T value  
    gridingInput_T = np.linspace(minT_In, maxT_In, num = totalPoint_T, retstep = True)
    # loading input values to (define the final array as input memory-space)
    gOutIn_array[-inWay, 0, :] = gridingInput_T[0]
    # step-size (increment of input T)
    dt = gridingInput_T[1]
    # starting
    # initialize the memory-space for local try-step 
    dydt1_array = np.zeros([outWay, totalPoint_X])
    dydt2_array = np.zeros([outWay, totalPoint_X])
    dydt3_array = np.zeros([outWay, totalPoint_X])
    dydt4_array = np.zeros([outWay, totalPoint_X])
    # initialize the memory-space for keeping current value
    currentOut_Value = np.zeros([outWay, totalPoint_X])
    for tn in range(totalPoint_T - 1):
        eqV = int(0) # index for VBMG equation list
        eqB = int(1) # index for VBMG equation list
        numberIn = int(0) # event_table index for viral loading number
        timeIn = int(1) # event_table index for viral loading time
        event_parameter = event_table[0]
        event_infect = event_table[1]
        event_repeat = event_table[2]
        event_vaccine = event_table[3]
        tn_unit = totalPoint_T/(maxT_In - minT_In)
        originVirus = int(event_parameter[0, 0])
        currentVirus = int(event_parameter[0, 1])
        minCell = event_parameter[0, 2]
        recoveredTime = event_parameter[0, 3]
        originVaccine = int(event_parameter[1, 0]) # vaccination
        currentVaccine= int(event_parameter[1, 1])
        minCellV = event_parameter[1, 2]
        recoveredTimeV = event_parameter[1, 3]
        # keep initial value at the moment of tn
        currentOut_Value[:, :] = np.copy(gOutIn_array[:-inWay, :, tn])
        currentIn_T_Value = np.copy(gOutIn_array[-inWay, 0, tn])
        # first try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                ## infection
                event_recovered = 0.0
                event_OAS_boost = 0.0
                event_OAS_press = 0.0
                # cutoff --- set virus = 0 if viral population < minCell  
                if gOutIn_array[eqV, xn, tn] < minCell:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVirus and tn > int(event_infect[xn, timeIn]*tn_unit)                                      and event_infect[xn, numberIn] > minCell                                      and gOutIn_array[eqB, xn, tn] < minCell:
                    gOutIn_array[eqB, xn, tn] = minCell 
                # viral loading for fresh-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_infect[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_infect[xn, numberIn] 
                # viral loading for repeated-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recovered = event_parameter[0, 4]
                # OAS+immunity --- # boosting origin-virus (memory B-cell from origin-virus is still existing) 
                if xn == originVirus and event_infect[xn + 1, numberIn] > minCell                                      and tn > int(event_infect[xn + 1, timeIn]*tn_unit):
                    event_OAS_boost = event_parameter[0, 5] # boosting in-rate of antibody-IgG from origin-virus
                # OAS-immunity --- depress non-origin-virus
                if xn > originVirus and event_infect[xn, numberIn] > minCell                                     and tn > int(event_infect[xn, timeIn]*tn_unit):
                    event_OAS_press = event_parameter[0, 6] # depress act-rate of memory B-cell from non-origin-virus
                ## vaccination
                event_recoveredV = 0.0
                event_OAS_boostV = 0.0
                event_OAS_pressV = 0.0
                event_OAS_slowV = 0.0
                # cutoff --- set vaccine = 0 if vaccine population < 1  
                if gOutIn_array[eqV, xn, tn] < minCellV:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVaccine and tn > int(event_vaccine[xn, timeIn]*tn_unit)                                        and event_vaccine[xn, numberIn] > minCellV                                        and gOutIn_array[eqB, xn, tn] < minCellV:
                    gOutIn_array[eqB, xn, tn] = minCellV 
                # viral loading for fresh-vaccination --- set vaccine if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_vaccine[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_vaccine[xn, numberIn] 
                # viral loading for repeated-vaccination --- set viral vaccination if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recoveredV = event_parameter[1, 4]
                # OAS+immunity --- # boosting origin-vaccine (memory B-cell from origin-vaccine is still existing) 
                if xn == originVaccine and event_vaccine[xn + 1, numberIn] > minCellV                                        and tn > int(event_vaccine[xn + 1, timeIn]*tn_unit):
                    event_OAS_boostV = event_parameter[1, 5] # boosting in-rate of antibody-IgG from origin-vaccine
                    event_OAS_slowV = event_parameter[1, 7]
                # OAS-immunity --- depress non-origin-vaccine
                if xn > originVaccine and event_vaccine[xn, numberIn] > minCellV                                       and tn > int(event_vaccine[xn, timeIn]*tn_unit):
                    event_OAS_pressV = event_parameter[1, 6] # depress act-rate of memory B-cell from non-origin
                ###
                dydt1_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt1_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # second half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                ## infection
                event_recovered = 0.0
                event_OAS_boost = 0.0
                event_OAS_press = 0.0
                # cutoff --- set virus = 0 if viral population < minCell  
                if gOutIn_array[eqV, xn, tn] < minCell:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVirus and tn > int(event_infect[xn, timeIn]*tn_unit)                                      and event_infect[xn, numberIn] > minCell                                      and gOutIn_array[eqB, xn, tn] < minCell:
                    gOutIn_array[eqB, xn, tn] = minCell 
                # viral loading for fresh-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_infect[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_infect[xn, numberIn] 
                # viral loading for repeated-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recovered = event_parameter[0, 4]
                # OAS+immunity --- # boosting origin-virus (memory B-cell from origin-virus is still existing) 
                if xn == originVirus and event_infect[xn + 1, numberIn] > minCell                                      and tn > int(event_infect[xn + 1, timeIn]*tn_unit):
                    event_OAS_boost = event_parameter[0, 5] # boosting in-rate of antibody-IgG from origin-virus
                # OAS-immunity --- depress non-origin-virus
                if xn > originVirus and event_infect[xn, numberIn] > minCell                                     and tn > int(event_infect[xn, timeIn]*tn_unit):
                    event_OAS_press = event_parameter[0, 6] # depress act-rate of memory B-cell from non-origin-virus
                ## vaccination
                event_recoveredV = 0.0
                event_OAS_boostV = 0.0
                event_OAS_pressV = 0.0
                event_OAS_slowV = 0.0
                # cutoff --- set vaccine = 0 if vaccine population < 1  
                if gOutIn_array[eqV, xn, tn] < minCellV:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVaccine and tn > int(event_vaccine[xn, timeIn]*tn_unit)                                        and event_vaccine[xn, numberIn] > minCellV                                        and gOutIn_array[eqB, xn, tn] < minCellV:
                    gOutIn_array[eqB, xn, tn] = minCellV 
                # viral loading for fresh-vaccination --- set vaccine if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_vaccine[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_vaccine[xn, numberIn] 
                # viral loading for repeated-vaccination --- set viral vaccination if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recoveredV = event_parameter[1, 4]
                # OAS+immunity --- # boosting origin-vaccine (memory B-cell from origin-vaccine is still existing) 
                if xn == originVaccine and event_vaccine[xn + 1, numberIn] > minCellV                                        and tn > int(event_vaccine[xn + 1, timeIn]*tn_unit):
                    event_OAS_boostV = event_parameter[1, 5] # boosting in-rate of antibody-IgG from origin-vaccine
                    event_OAS_slowV = event_parameter[1, 7]
                # OAS-immunity --- depress non-origin-vaccine
                if xn > originVaccine and event_vaccine[xn, numberIn] > minCellV                                       and tn > int(event_vaccine[xn, timeIn]*tn_unit):
                    event_OAS_pressV = event_parameter[1, 6] # depress act-rate of memory B-cell from non-origin
                ###
                dydt2_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt2_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # third half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                ## infection
                event_recovered = 0.0
                event_OAS_boost = 0.0
                event_OAS_press = 0.0
                # cutoff --- set virus = 0 if viral population < minCell  
                if gOutIn_array[eqV, xn, tn] < minCell:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVirus and tn > int(event_infect[xn, timeIn]*tn_unit)                                      and event_infect[xn, numberIn] > minCell                                      and gOutIn_array[eqB, xn, tn] < minCell:
                    gOutIn_array[eqB, xn, tn] = minCell 
                # viral loading for fresh-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_infect[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_infect[xn, numberIn] 
                # viral loading for repeated-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recovered = event_parameter[0, 4]
                # OAS+immunity --- # boosting origin-virus (memory B-cell from origin-virus is still existing) 
                if xn == originVirus and event_infect[xn + 1, numberIn] > minCell                                      and tn > int(event_infect[xn + 1, timeIn]*tn_unit):
                    event_OAS_boost = event_parameter[0, 5] # boosting in-rate of antibody-IgG from origin-virus
                # OAS-immunity --- depress non-origin-virus
                if xn > originVirus and event_infect[xn, numberIn] > minCell                                     and tn > int(event_infect[xn, timeIn]*tn_unit):
                    event_OAS_press = event_parameter[0, 6] # depress act-rate of memory B-cell from non-origin-virus
                ## vaccination
                event_recoveredV = 0.0
                event_OAS_boostV = 0.0
                event_OAS_pressV = 0.0
                event_OAS_slowV = 0.0
                # cutoff --- set vaccine = 0 if vaccine population < 1  
                if gOutIn_array[eqV, xn, tn] < minCellV:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVaccine and tn > int(event_vaccine[xn, timeIn]*tn_unit)                                        and event_vaccine[xn, numberIn] > minCellV                                        and gOutIn_array[eqB, xn, tn] < minCellV:
                    gOutIn_array[eqB, xn, tn] = minCellV 
                # viral loading for fresh-vaccination --- set vaccine if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_vaccine[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_vaccine[xn, numberIn] 
                # viral loading for repeated-vaccination --- set viral vaccination if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recoveredV = event_parameter[1, 4]
                # OAS+immunity --- # boosting origin-vaccine (memory B-cell from origin-vaccine is still existing) 
                if xn == originVaccine and event_vaccine[xn + 1, numberIn] > minCellV                                        and tn > int(event_vaccine[xn + 1, timeIn]*tn_unit):
                    event_OAS_boostV = event_parameter[1, 5] # boosting in-rate of antibody-IgG from origin-vaccine
                    event_OAS_slowV = event_parameter[1, 7]
                # OAS-immunity --- depress non-origin-vaccine
                if xn > originVaccine and event_vaccine[xn, numberIn] > minCellV                                       and tn > int(event_vaccine[xn, timeIn]*tn_unit):
                    event_OAS_pressV = event_parameter[1, 6] # depress act-rate of memory B-cell from non-origin
                ###
                dydt3_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt3_array[:, :]*dt # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt # update input
        # fourth try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                ## infection
                event_recovered = 0.0
                event_OAS_boost = 0.0
                event_OAS_press = 0.0
                # cutoff --- set virus = 0 if viral population < minCell  
                if gOutIn_array[eqV, xn, tn] < minCell:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVirus and tn > int(event_infect[xn, timeIn]*tn_unit)                                      and event_infect[xn, numberIn] > minCell                                      and gOutIn_array[eqB, xn, tn] < minCell:
                    gOutIn_array[eqB, xn, tn] = minCell 
                # viral loading for fresh-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_infect[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_infect[xn, numberIn] 
                # viral loading for repeated-infection --- set viral infection if tn == specific time 
                if event_infect[xn, numberIn] > minCell and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recovered = event_parameter[0, 4]
                # OAS+immunity --- # boosting origin-virus (memory B-cell from origin-virus is still existing) 
                if xn == originVirus and event_infect[xn + 1, numberIn] > minCell                                      and tn > int(event_infect[xn + 1, timeIn]*tn_unit):
                    event_OAS_boost = event_parameter[0, 5] # boosting in-rate of antibody-IgG from origin-virus
                # OAS-immunity --- depress non-origin-virus
                if xn > originVirus and event_infect[xn, numberIn] > minCell                                     and tn > int(event_infect[xn, timeIn]*tn_unit):
                    event_OAS_press = event_parameter[0, 6] # depress act-rate of memory B-cell from non-origin-virus
                ## vaccination
                event_recoveredV = 0.0
                event_OAS_boostV = 0.0
                event_OAS_pressV = 0.0
                event_OAS_slowV = 0.0
                # cutoff --- set vaccine = 0 if vaccine population < 1  
                if gOutIn_array[eqV, xn, tn] < minCellV:
                    gOutIn_array[eqV, xn, tn] = 0.0 
                # bottom line --- set Bcell = 1 if Bcell population < 1  
                if xn >= originVaccine and tn > int(event_vaccine[xn, timeIn]*tn_unit)                                        and event_vaccine[xn, numberIn] > minCellV                                        and gOutIn_array[eqB, xn, tn] < minCellV:
                    gOutIn_array[eqB, xn, tn] = minCellV 
                # viral loading for fresh-vaccination --- set vaccine if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_vaccine[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_vaccine[xn, numberIn] 
                # viral loading for repeated-vaccination --- set viral vaccination if tn == specific time 
                if event_vaccine[xn, numberIn] > minCellV and tn == int(event_repeat[xn, timeIn]*tn_unit):
                    gOutIn_array[eqV, xn, tn] = event_repeat[xn, numberIn] 
                    event_recoveredV = event_parameter[1, 4]
                # OAS+immunity --- # boosting origin-vaccine (memory B-cell from origin-vaccine is still existing) 
                if xn == originVaccine and event_vaccine[xn + 1, numberIn] > minCellV                                        and tn > int(event_vaccine[xn + 1, timeIn]*tn_unit):
                    event_OAS_boostV = event_parameter[1, 5] # boosting in-rate of antibody-IgG from origin-vaccine
                    event_OAS_slowV = event_parameter[1, 7]
                # OAS-immunity --- depress non-origin-vaccine
                if xn > originVaccine and event_vaccine[xn, numberIn] > minCellV                                       and tn > int(event_vaccine[xn, timeIn]*tn_unit):
                    event_OAS_pressV = event_parameter[1, 6] # depress act-rate of memory B-cell from non-origin
                ###
                dydt4_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio 
        # solid step (update the next output) by accumulate all the try-steps with proper adjustment
        gOutIn_array[:-inWay, :, tn + 1] = currentOut_Value[:, :] + dt*(dydt1_array[:, :]/6 
                                                                                      + dydt2_array[:, :]/3 
                                                                                      + dydt3_array[:, :]/3 
                                                                                      + dydt4_array[:, :]/6)
        # restore to initial value
        gOutIn_array[:-inWay, :, tn] = np.copy(currentOut_Value[:, :])
        gOutIn_array[-inWay, 0, tn] = np.copy(currentIn_T_Value) 
        # end of loop
    return (gOutIn_array[:-inWay, :])

