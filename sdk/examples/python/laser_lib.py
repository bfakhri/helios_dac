import ctypes
import numpy as np
import math
import time
import cv2

#Define point structure
class HeliosPoint(ctypes.Structure):
    #_pack_=1
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]
class Dac:
    def __init__(self):
        #Load and initialize library
        self.HeliosLib = ctypes.cdll.LoadLibrary("./libHeliosDacAPI.so")
        self.num_devices = self.HeliosLib.OpenDevices()
        print("Found ", self.num_devices, "Helios DACs")
        # Define limits
        self.xy_max = int(2**12-1)

class DacQueue:
    '''
    A queue for patterns sent to the dac. 
    Performs smart stitching between patterns.
    '''
    def __init__(self):
        # Dac object for this queue
        self.dac = Dac()
        # The last position of the last pattern (x,y)
        self.last_pos = (0,0)
        # Sample rate of the DAC
        self.dac_rate = 30000
        # Debugging dac rate
        self.debug_rate = 150
        # Color shift constants (this should change w/ dac_rate)
        #self.color_shifts = [11, 10, 9]
        self.color_shifts = [5, 5, 4]
        
    
    def submit(self, pat_pos, pat_col, angular_density=100, debug=False):
        '''
        Submits a new pattern to the dac, transitioning smoothly from the last one.
        Angular density describes how many points per radian should be used to transition
        '''
        # Make the transition
        dist = np.sqrt(np.sum(np.power(self.last_pos-pat_pos[0,:], 2)))
        num_gap_points = int(dist*angular_density)

        gap_pos = np.linspace(self.last_pos, pat_pos[0,:], num_gap_points, dtype=np.float)
        gap_col = np.zeros((num_gap_points, 3), dtype=np.float)
        if(debug):
            gap_col = np.ones_like(gap_col)/4

        # Prep the transition pattern
        gap_points, num_gap_points = self.prep_pattern(gap_pos, gap_col, gap=True)
        
        # Prep the new pattern
        pat_points, num_pat_points = self.prep_pattern(pat_pos, pat_col)
        
        # Send the transition pattern to the dac
        self.write_frames(gap_points, num_gap_points, do_not_loop=True, debug=debug)
        # Send the new pattern to the dac
        self.write_frames(pat_points, num_pat_points, do_not_loop=True, debug=debug)
        # Set the last_position
        self.last_pos = pat_pos[-1, :].copy()
        
    def prep_pattern(self, arr_pos, arr_col, gap=False):
        '''
        1) Scales and color corrects 
        2) Converts from unit space to DAC coordinates
        3) Produces a DAC compatible frame and displays it
        '''
        # Force pass-by-value
        arr_pos = arr_pos.copy()
        arr_col = arr_col.copy()
        # Scale the pattern 
        arr_pos = position_corection(arr_pos)
        # Format the position array (-1.0-1.0) -> int(0-4095)
        arr_pos = (((arr_pos+1)/2.0)*self.dac.xy_max).astype(np.int32)

        # Perform color correction if not gap points
        if(not gap):
            arr_col = color_correction(arr_col, color_shifts=self.color_shifts)
        # Format the color array (0-1.0) -> int(0-255)
        arr_col = (arr_col*255).astype(np.int32)
        # Fill a heliospoint arr with these values
        num_points = len(arr_pos)

        frameType = HeliosPoint * num_points
        points = frameType()
        # Fill the frame
        for idx in range(num_points):
            points[idx] =     HeliosPoint(int(arr_pos[idx, 0]), 
                                         int(arr_pos[idx, 1]), 
                                         int(arr_col[idx, 0]), 
                                         int(arr_col[idx, 1]),
                                         int(arr_col[idx, 2]),
                                         int(255))
        
        return points, num_points
            
    def write_frames(self, points, num_points, do_not_loop=False, start_immediately=False, debug=False):
        if(num_points < 1):
            return
        # Write the values out 
        status_attempts = 0
        max_attempts = np.inf
        # Make 512 attempts for DAC status to be ready. After that, just give up and try to write the frame anyway
        while(status_attempts < max_attempts and self.dac.HeliosLib.GetStatus(0) != 1):
            status_attempts += 1
            
        # Create flags 
        flags = do_not_loop << 1 | start_immediately << 0
        # Send to DAC
        frame_rate = self.debug_rate if debug else self.dac_rate
        self.dac.HeliosLib.WriteFrame(0, frame_rate, flags, ctypes.pointer(points), num_points)

def color_correction(arr_col, color_shifts=[]):
    '''
    Corrects the nonlinearities in the color curve 
    Color shifts correspond to [Red, Green, Blue]
    '''
    non_zero = arr_col > 0
    # Scale to the lower cutoff (R, G, B)
    lower_cutoff = np.array([0.25, 0.09, 0.09])
    arr_col = arr_col*(1-lower_cutoff) + lower_cutoff
    
    # Cutoff any zeros to ensure real black
    arr_col *= non_zero
    
    # Perform time shifting
    # Corresponds to Red, Green, Blue @ dac_rate 55k
    for col_idx, shift in enumerate(color_shifts):
        arr_col[:, col_idx] = np.roll(arr_col[:, col_idx], shift, axis=0)
    
    return arr_col

def position_corection(arr_pos):
    '''
    Corrects x-y directions so that (x=-1,y=-1) is bottom left
    Scales the entire pattern so that it does not clip (amps are weird)
    '''
    # Reverse direction of x and y
    arr_pos *= -1
    
    max_scale = 0.75
    return arr_pos*max_scale
    
def make_circular(arr_pos, arr_color):
    return np.concatenate([arr_pos, arr_pos[::-1, :]]), np.concatenate([arr_color, arr_color[::-1, :]])

def connect_in_space(arr_pos_list, angular_density=100, wait_per=None):
    '''
    Given a list of position arrays, produce a list of array positions that are connected
    '''
    new_arr_pos_list = []
    for i in range(len(arr_pos_list)-1):
        # Get start/end positions for this gap
        pos_start = arr_pos_list[i][-1, :].copy()
        pos_end = arr_pos_list[i+1][ 0, :].copy()
        
        # Make the transition
        dist = np.sqrt(np.sum(np.power(pos_start-pos_end, 2)))
        num_gap_points = int(dist*angular_density)
        gap_pos = np.linspace(pos_start, pos_end, num_gap_points, dtype=np.float)
        
        # If we have a wait period, add it to the end of the gap positions
        if(wait_per):
            rep_bef = np.repeat(pos_start[np.newaxis, :], wait_per, axis=0)
            rep_aft = np.repeat(pos_end[np.newaxis, :], wait_per, axis=0)
            gap_pos = np.concatenate([rep_bef, gap_pos, rep_aft])
            
        
        new_arr_pos_list.append(arr_pos_list[i])
        new_arr_pos_list.append(gap_pos.copy())
    new_arr_pos_list.append(arr_pos_list[-1])
    
    return new_arr_pos_list

def fixed_interp(points, n_interp_points):
    '''
    Expands a list of points to a position array where those 
    points are the anchors in a smooth trajectory. 
    There is a fixed number of interp points between 
    the given points. 
    Inputs: 
        points - (N,2) - list of anchor points
        n_interp_points - number of points between anchors
    '''
    interped_pos_arr = []
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        gap_pos = np.linspace(p1, p2, n_interp_points)
        interped_pos_arr += [p1]
        interped_pos_arr += [gap_pos]
    interped_pos_arr += [p2]
    
    # Convert them into one array 
    interped_pos_arr = np.concatenate(interped_pos_arr, axis=0)
    return interped_pos_arr
