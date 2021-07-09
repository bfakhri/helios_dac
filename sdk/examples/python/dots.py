# -*- coding: utf-8 -*-
"""
Example for using Helios DAC libraries in python (using C library with ctypes)

NB: If you haven't set up udev rules you need to use sudo to run the program for it to detect the DAC.
"""

import ctypes
import numpy as np

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

num_x_points = 2
num_y_points = 2
num_points = int(num_x_points*num_y_points)
points = np.linspace(0, 1, num_x_points)

#frameType = HeliosPoint * num_points
frames = []
frameType = HeliosPoint * num_x_points
lum = 255
x_points = [x for x in range(num_x_points)]
y_points = [y for y in range(num_y_points)]

# Build the frames we want to display
# Go horizontal line by line
for y in y_points:
    frames.append(frameType())
    if(y % 2 == 0):
        x_points_list = x_points
    else:
        x_points_list = reversed(x_points)
    for idx,x in enumerate(x_points_list):
        x_coord = int(xy_max*points[x])
        y_coord = int(xy_max*points[y])
        #frames.append(HeliosPoint(int(x_coord),int(y_coord), lum, lum, lum, 255))
        frames[-1][idx] = HeliosPoint(int(x_coord),int(y_coord), lum, lum, lum, 255)
        print(x_coord, y_coord)

# Now go backwards
frames += reversed(frames)

#frames.append(HeliosPoint(int(100),int(0), lum, lum, lum, 255))
dac_idx = 0

#Play frames on DAC
try:
    while(True):
        for line in range(num_y_points):
            statusAttempts = 0
            # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
            while (statusAttempts < 512 and HeliosLib.GetStatus(dac_idx) != 1):
                statusAttempts += 1
            HeliosLib.WriteFrame(dac_idx, 30, 0, ctypes.pointer(frames[line]), num_x_points) #Send the frame
except KeyboardInterrupt:
    print('Closing the program')
    HeliosLib.CloseDevices()
