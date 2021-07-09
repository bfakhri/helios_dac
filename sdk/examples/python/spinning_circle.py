# -*- coding: utf-8 -*-
"""
Example for using Helios DAC libraries in python (using C library with ctypes)

NB: If you haven't set up udev rules you need to use sudo to run the program for it to detect the DAC.
"""

import ctypes
import numpy as np
import math
import time

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
def build_frames(space, margin, start_ang=0):
    frames.append(frameType())
    for idx,angle in enumerate(np.linspace(start_ang, start_ang+2*np.pi, num_points)):
        x_coord = xy_max*((math.cos(angle)+1))/2
        y_coord = xy_max*((math.sin(angle)+1))/2
        red = 255
        green = 255
        blue = 255
        #if(angle > np.pi and angle < 1.5*np.pi):
        if(idx%space < margin):
            lum = 0
        else:
            lum = 1
        frames[0][idx] = HeliosPoint(int(x_coord),int(y_coord), lum*red, lum*green, lum*blue, 255)
    return frames

frames = build_frames(space=100, margin=3)
dac_idx = 0

step_num = 0
#Play frames on DAC
dlen = 3
space = 100
margin = space - dlen
delta = -1
ang = 0
try:
    while(True):
        statusAttempts = 0
        # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
        while (statusAttempts < 512 and HeliosLib.GetStatus(dac_idx) != 1):
            statusAttempts += 1
        HeliosLib.WriteFrame(dac_idx, 65000, 0, ctypes.pointer(frames[0]), num_points) #Send the frame
        frames = build_frames(space=space, margin=margin, start_ang=ang)
        #frames = build_frames(space=100, margin=90, start_ang=ang)
        space += delta
        ang -= 0.05
        #if(space <= 0 or margin <= 0 or space > 100 or margin > 100 - dlen):
        if(space <= 0 or space > 100):
            delta *= -1
            space += delta
        print(ang, margin, space, delta)
        margin = space - dlen
        time.sleep(0.01)
except KeyboardInterrupt:
    print('Closing the program')
    HeliosLib.CloseDevices()

print('Closed')
