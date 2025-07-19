#!/usr/bin/env python
# coding: utf-8

# # Camera Gain Tester
# This companion script strobes a laser at a fixed point while plotting
# the webcam's average brightness in real-time. This is useful for
# adjusting camera settings (gain, exposure, etc.) to ensure a clear
# signal is detected when the laser is on.

import laser_lib
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import datetime as dt

# --- Configuration ---
# Number of data points to display on the plot
MAX_POINTS = 100
# Define how long the laser should be on and off
LASER_ON_DURATION_MS = 1000
LASER_OFF_DURATION_MS = 1000
# Update interval for the plot. This should be fast for a smooth graph.
PLOT_UPDATE_INTERVAL_MS = 50

# --- Global State ---
laser_is_on = False
frame_counter = 0

# --- Data Acquisition Function ---
def get_camera_brightness(cap):
    """
    Captures a single frame and returns its average brightness.
    """
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Could not read frame from camera.")
        return None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_frame)

# --- Plot Initialization ---
brightness_data = deque(maxlen=MAX_POINTS)
time_data = deque(maxlen=MAX_POINTS)
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], 'r-')

def setup_plot():
    """Configures the appearance of the plot."""
    ax.set_ylim(0, 255)
    ax.set_xlim(0, MAX_POINTS)
    ax.set_title('Real-time Camera Brightness', fontsize=16)
    ax.set_ylabel('Average Brightness (0-255)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.20)

# --- Animation Update Function ---
def update(frame_number, cap, queue, pos, col_on, col_off, on_frames, off_frames):
    """
    This function is called for each new frame of the animation.
    It uses a state machine to control the laser's on/off duration
    and updates the brightness plot on every frame.
    """
    global laser_is_on, frame_counter
    print(frame_number, frame_counter, laser_is_on)

    # --- State Machine for Laser Control ---
    if laser_is_on:
        if frame_counter >= on_frames:
            laser_is_on = False
            frame_counter = 0
            queue.submit(pos, col_off, loop=True) # Turn laser off
    else: # laser is off
        if frame_counter >= off_frames:
            laser_is_on = True
            frame_counter = 0
            queue.submit(pos, col_on, loop=True) # Turn laser on
    
    frame_counter += 1

    # --- Data Acquisition and Plotting (runs every frame) ---
    brightness = get_camera_brightness(cap)
    if brightness is None:
        return line,

    # Update data for plotting
    brightness_data.append(brightness)
    time_data.append(dt.datetime.now().strftime('%H:%M:%S'))

    # Update plot
    line.set_data(np.arange(len(brightness_data)), brightness_data)
    ax.set_xticks(np.arange(len(time_data)))
    ax.set_xticklabels(list(time_data), rotation=45, ha='right')
    ax.set_xlim(0, len(brightness_data) - 1 if len(brightness_data) > 1 else 1)
    
    return line,

# --- Main Execution ---
if __name__ == "__main__":
    cap = None
    queue = None
    
    try:
        # --- Initialize Laser ---
        print("Initializing laser queue...")
        queue = laser_lib.DacQueue()
        queue.dac_rate = 1000
        print("✅ Laser queue initialized.")

        # --- Initialize Camera ---
        print("\nInitializing camera...")
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam at index {camera_index}. Is it connected?")
        print(f"✅ Camera {camera_index} initialized.")

        # --- Laser Strobe Setup ---

        # Calculate how many plot frames correspond to the on/off durations
        on_frames = LASER_ON_DURATION_MS // PLOT_UPDATE_INTERVAL_MS
        off_frames = LASER_OFF_DURATION_MS // PLOT_UPDATE_INTERVAL_MS
        print(f"Laser will be ON for {on_frames} frames and OFF for {off_frames} frames.")
        # A single point for the laser to target
        laser_point = np.zeros((off_frames * PLOT_UPDATE_INTERVAL_MS * 1000 // queue.dac_rate, 2))
        
        # Frame for LASER ON at the target point
        #arr_col_on = np.array([[1.0, 1.0, 1.0]]) # Full power
        arr_col_on = np.ones((on_frames * PLOT_UPDATE_INTERVAL_MS * 1000 // queue.dac_rate, 3))
        print(arr_col_on.shape)
        
        # Frame for LASER OFF
        #arr_col_off = np.array([[0.0, 0.0, 0.0]]) # Off
        arr_col_off = np.zeros((off_frames * PLOT_UPDATE_INTERVAL_MS * 1000 // queue.dac_rate, 3))
        


        # --- Plotting and Animation ---
        setup_plot()

        # Create the animation, passing required objects to the update function
        ani = animation.FuncAnimation(
            fig,
            update,
            fargs=(cap, queue, laser_point, arr_col_on, arr_col_off, on_frames, off_frames),
            interval=PLOT_UPDATE_INTERVAL_MS,
            blit=True,
            cache_frame_data=False
        )

        print("\nStrobing laser and plotting brightness. Close the plot window to exit.")
        plt.show()

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    finally:
        # --- Cleanup ---
        if queue:
            # Ensure laser is off
            off_point = np.array([[0.0, 0.0]])
            off_color = np.array([[0.0, 0.0, 0.0]])
            queue.submit(off_point, off_color, loop=True)
            print("Laser turned off.")
        if cap:
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("Script finished.")

