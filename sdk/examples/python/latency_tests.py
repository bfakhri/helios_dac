# -*- coding: utf-8 -*-
"""
Example for using Helios DAC libraries in python (using C library with ctypes)

NB: If you haven't set up udev rules you need to use sudo to run the program for it to detect the DAC.
"""

import ctypes
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

#Play frames on DAC
dac_idx = 0
pps = 65000
flags = 0
num_points = 1
x = 0

# Max integer position for the X/Y coordinates
dac_maxpos = 2**12 - 1

delta = 1

#frame = HeliosPoint(int(x),int(x),255,255,255,255)
frame = HeliosPoint(int(0),int(0),255,255,255,255)

try:
    start_time = time.perf_counter_ns()
    while(True):
        statusAttempts = 0
        # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
        while (statusAttempts < 512 and HeliosLib.GetStatus(dac_idx) != 1):
            statusAttempts += 1
        HeliosLib.WriteFrame(dac_idx, pps, flags, ctypes.pointer(frame), num_points) #Send the frame
        frame = HeliosPoint(int(x),int(x),255,255,255,255)
        x += delta
        if(x > dac_maxpos or x < 0):
            end_time = time.perf_counter_ns()
            break
except KeyboardInterrupt:
    print('Closing the program')
    HeliosLib.CloseDevices()


duration_ns = end_time - start_time
duration_s = duration_ns/(10**9)
pps_effective = float(dac_maxpos)/duration_s
latency = 1.0/pps_effective*1000

print(f'Nanoseconds: {duration_ns}\tSeconds: {duration_s}\tNumPoints: {dac_maxpos}\tPPS_eff: {pps_effective}\tMeanLatency: {latency}ms')
HeliosLib.CloseDevices()
