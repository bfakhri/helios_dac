# -*- coding: utf-8 -*-
"""
Example for using Helios DAC libraries in python (using C library with ctypes)

NB: If you haven't set up udev rules you need to use sudo to run the program for it to detect the DAC.
"""

import ctypes
import numpy as np
import math

#Define point structure
class HeliosPoint(ctypes.Structure):
    #_pack_=1
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]

#Load and initialize library
HeliosLib = ctypes.cdll.LoadLibrary("./libHeliosDacAPI.so")
numDevices = HeliosLib.OpenDevices()
print("Found ", numDevices, "Helios DACs")

xy_max = int(2**12-1)
num_points = 1000

#frameType = HeliosPoint * num_points
frames = []
frameType = HeliosPoint * num_points
#lum = 255

# Build the frames we want to display
frames.append(frameType())
for idx,angle in enumerate(np.linspace(0, 2*np.pi, num_points)):
    x_coord = xy_max*((math.cos(angle)+1))/2
    y_coord = xy_max*((math.sin(angle)+1))/2
    red = 255
    green = 255
    blue = 255
    #if(angle > np.pi and angle < 1.5*np.pi):
    if(idx%10 < 8):
        lum = 0
    else:
        lum = 1
    frames[0][idx] = HeliosPoint(int(x_coord),int(y_coord), lum*red, lum*green, lum*blue, 255)

dac_idx = 0

step_num = 0
#Play frames on DAC
try:
    HeliosLib.WriteFrame(dac_idx, 65000, 0, ctypes.pointer(frames[0]), num_points) #Send the frame
    while(True):
        continue
except KeyboardInterrupt:
    print('Closing the program')
    HeliosLib.CloseDevices()

print('Closed')
