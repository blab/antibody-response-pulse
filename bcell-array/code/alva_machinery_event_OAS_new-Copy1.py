
# coding: utf-8

# # Infectious Pulse
# https://github.com/blab/antibody-response-pulse/
# 
# ## Homemade machinery for solving partial differential equations
# ### Runge-kutta algorithm for a array of coupled partial differential equation 

# In[1]:

'''
author: Alvason Zhenhua Li
date:   03/23/2015

Home-made machinery for solving partial differential equations
'''
import numpy as np

# define RK4 for an array (3, n) of coupled differential equations
def AlvaRungeKutta4XT(pde_array, initial_Out, minX_In, maxX_In, totalPoint_X, minT_In, maxT_In, totalPoint_T, event_table):
    global event_active
    event_active = 0.0
    global event_OAS
    event_OAS = 0.0
    global event_OAS_B
    event_OAS_B = 0.0
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
        event_parameter = event_table[0][0]
        event_1st = event_table[1]
        event_2nd = event_table[2]
        tn_unit = totalPoint_T/(maxT_In - minT_In)
        activeTime = event_parameter[2]
        # keep initial value at the moment of tn
        currentOut_Value[:, :] = np.copy(gOutIn_array[:-inWay, :, tn])
        currentIn_T_Value = np.copy(gOutIn_array[-inWay, 0, tn])
        # first try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                # cutoff --- set virus = 0 if viral population < 1  
                if gOutIn_array[0, xn, tn] < 1.0:
                    gOutIn_array[0, xn, tn] = 0.0 
                # post-infection --- replace pre-parameter by post-parameter
                event_active = event_parameter[0]
                event_OAS = 0.0
                event_OAS_B = 0.0
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int((event_1st[xn, 1] + activeTime)*tn_unit):
                    event_active = event_parameter[1]
                # 1st-infection --- set viral infection if tn == specific time 
                if tn == int(event_1st[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_1st[xn, 0] 
                # 2nd-infection --- set viral infection if tn == specific time 
                if tn == int(event_2nd[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_2nd[xn, 0] 
                # OAS+immunity --- set viral infection if tn == specific time 
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn + 1, 1]*tn_unit):
                    event_OAS = event_parameter[3] # in-rate of antibody-IgG from memory B-cell
                # OAS-immunity --- set viral infection if tn == specific time 
                if xn > 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn, 1]*tn_unit):
                    event_OAS_B = event_parameter[4] # in-rate of antibody-IgG from memory B-cell
                ###
                dydt1_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt1_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # second half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                # cutoff --- set virus = 0 if viral population < 1  
                if gOutIn_array[0, xn, tn] < 1.0:
                    gOutIn_array[0, xn, tn] = 0.0 
                # post-infection --- replace pre-parameter by post-parameter
                event_active = event_parameter[0]
                event_OAS = 0.0
                event_OAS_B = 0.0
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int((event_1st[xn, 1] + activeTime)*tn_unit):
                    event_active = event_parameter[1]
                # 1st-infection --- set viral infection if tn == specific time 
                if tn == int(event_1st[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_1st[xn, 0] 
                # 2nd-infection --- set viral infection if tn == specific time 
                if tn == int(event_2nd[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_2nd[xn, 0] 
                # OAS+immunity --- set viral infection if tn == specific time 
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn + 1, 1]*tn_unit):
                    event_OAS = event_parameter[3] # in-rate of antibody-IgG from memory B-cell
                # OAS-immunity --- set viral infection if tn == specific time 
                if xn > 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn, 1]*tn_unit):
                    event_OAS_B = event_parameter[4] # in-rate of antibody-IgG from memory B-cell
                ###
                dydt2_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt2_array[:, :]*dt/2 # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt/2 # update input
        # third half try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                # cutoff --- set virus = 0 if viral population < 1  
                if gOutIn_array[0, xn, tn] < 1.0:
                    gOutIn_array[0, xn, tn] = 0.0 
                # post-infection --- replace pre-parameter by post-parameter
                event_active = event_parameter[0]
                event_OAS = 0.0
                event_OAS_B = 0.0
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int((event_1st[xn, 1] + activeTime)*tn_unit):
                    event_active = event_parameter[1]
                # 1st-infection --- set viral infection if tn == specific time 
                if tn == int(event_1st[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_1st[xn, 0] 
                # 2nd-infection --- set viral infection if tn == specific time 
                if tn == int(event_2nd[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_2nd[xn, 0] 
                # OAS+immunity --- set viral infection if tn == specific time 
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn + 1, 1]*tn_unit):
                    event_OAS = event_parameter[3] # in-rate of antibody-IgG from memory B-cell
                # OAS-immunity --- set viral infection if tn == specific time 
                if xn > 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn, 1]*tn_unit):
                    event_OAS_B = event_parameter[4] # in-rate of antibody-IgG from memory B-cell
                ###
                dydt3_array[i, xn] = pde_array[i](gOutIn_array[:, :, tn])[xn] # computing ratio   
        gOutIn_array[:-inWay, :, tn] = currentOut_Value[:, :] + dydt3_array[:, :]*dt # update output
        gOutIn_array[-inWay, 0, tn] = currentIn_T_Value + dt # update input
        # fourth try-step
        for i in range(outWay):
            for xn in range(totalPoint_X):
                ###
                # cutoff --- set virus = 0 if viral population < 1  
                if gOutIn_array[0, xn, tn] < 1.0:
                    gOutIn_array[0, xn, tn] = 0.0 
                # post-infection --- replace pre-parameter by post-parameter
                event_active = event_parameter[0]
                event_OAS = 0.0
                event_OAS_B = 0.0
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int((event_1st[xn, 1] + activeTime)*tn_unit):
                    event_active = event_parameter[1]
                # 1st-infection --- set viral infection if tn == specific time 
                if tn == int(event_1st[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_1st[xn, 0] 
                # 2nd-infection --- set viral infection if tn == specific time 
                if tn == int(event_2nd[xn, 1]*tn_unit):
                    gOutIn_array[0, xn, tn] = event_2nd[xn, 0] 
                # OAS+immunity --- set viral infection if tn == specific time 
                if xn == 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn + 1, 1]*tn_unit):
                    event_OAS = event_parameter[3] # in-rate of antibody-IgG from memory B-cell
                # OAS-immunity --- set viral infection if tn == specific time 
                if xn > 1 and event_1st[xn, 0] > 1.0 and tn > int(event_1st[xn, 1]*tn_unit):
                    event_OAS_B = event_parameter[4] # in-rate of antibody-IgG from memory B-cell
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


# In[2]:

# min-max sorting
def AlvaMinMax(data):
    totalDataPoint = np.size(data)
    minMaxListing = np.zeros(totalDataPoint)   
    for i in range(totalDataPoint):
        # searching the minimum in current array
        jj = 0 
        minMaxListing[i] = data[jj] # suppose the 1st element [0] of current data-list is the minimum
        for j in range(totalDataPoint - i):
            if data[j] < minMaxListing[i]: 
                minMaxListing[i] = data[j]
                jj = j # recording the position of selected element
        # reducing the size of searching zone (removing the minmum from current array)
        data = np.delete(data, jj)
    return (minMaxListing)

