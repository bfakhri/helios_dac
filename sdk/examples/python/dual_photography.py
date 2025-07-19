#!/usr/bin/env python
# coding: utf-8

# # Dual Photography with Webcam Input
# This script combines laser scanning with a webcam to capture scene data.
# It replaces the Arduino + photoresistor setup with a standard webcam,
# which measures the light from the laser as it scans across a scene.

import laser_lib
import numpy as np
import math
import time
import cv2
import matplotlib.pyplot as plt

# --- Data Acquisition Function ---
def get_camera_line_data(cap, num_points=100):
    """
    Captures a frame from the webcam and processes it to get a line of brightness data.

    Args:
        cap (cv2.VideoCapture): The active camera capture object.
        num_points (int): The number of data points to generate for the line.

    Returns:
        np.array: A 1D numpy array of brightness values, or None on error.
    """
    # Let the camera stabilize
    time.sleep(0.1) 
    
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Could not read frame from camera.")
        return None

    # Convert to grayscale to work with brightness
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Average the image vertically to get a 1D horizontal brightness profile.
    # This collapses the 2D image into a 1D array representing brightness across the width.
    horizontal_profile = np.mean(gray_frame, axis=0)

    # Resample the profile to the desired number of points (e.g., 100)
    # This ensures our data has the same dimensions as the original Arduino data.
    x_original = np.linspace(0, 1, len(horizontal_profile))
    x_new = np.linspace(0, 1, num_points)
    resampled_profile = np.interp(x_new, x_original, horizontal_profile)
    
    return resampled_profile

def sample_camera(cap, throwaway_frames=1, avg_frames=1):
    """
    Captures a frame from the webcam and processes it to get a point of brightness data.

    Args:
        cap (cv2.VideoCapture): The active camera capture object.
        throwaway_frames: how many frames to throw away before we capture the point.
    """
    #print('sampling...')
    for _ in range(throwaway_frames):
        cap.read()
    frames = []
    for _ in range(avg_frames):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Could not read frame from camera.")
            return None
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(np.mean(gray_frame))

    # Convert to grayscale to work with brightness

    return np.mean(frames)


# --- Main Execution ---
if __name__ == "__main__":
    cap = None
    queue = None
    
    try:
        # --- Initialize Laser ---
        print("Initializing laser queue...")
        queue = laser_lib.DacQueue()
        print("✅ Laser queue initialized.")

        # --- Initialize Camera ---
        print("\nInitializing camera...")
        camera_index = 0  # Default to the first available camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam at index {camera_index}. Is it connected?")
        print(f"✅ Camera {camera_index} initialized.")

        # --- Scanning Parameters ---
        xy_min = -0.86
        xy_max = 0.86
        queue.dac_rate = 10000
        T = 100  # Points in a single scan line
        T_rew = T*10 # Points for the rewind path
        
        # Define the laser path for a single line scan (left to right)
        arr_pos = np.zeros((T, 2))
        arr_pos[:, 0] = np.linspace(xy_max, xy_min, T)
        arr_col = np.ones((T, 3)) # Laser ON

        # Define the laser path for the rewind (right to left)
        arr_pos_rew = np.zeros((T_rew, 2))
        arr_pos_rew[:, 0] = np.linspace(xy_min, xy_max, T_rew)
        arr_col_rew = np.zeros((T_rew, 3)) # Laser OFF

        num_lines = 100  # Vertical and horizontal resolution of the final image
        
        lines_y = np.linspace(xy_max, xy_min, num_lines)
        lines_x = np.linspace(xy_max, xy_min, num_lines)
        raw_img_arr = []

        print("\nStarting scene scan...")
        # --- Main Scanning Loop ---
        for idx_y, y in enumerate(lines_y):
            line_arr = []
            
            for idx_x, x in enumerate(lines_x):
                print(f"Scanning {idx_x + 1}/{num_lines}   {idx_y + 1}/{num_lines}", end='\r')
                arr_pos[:, 0] = x
                arr_pos[:, 1] = y
                arr_pos_rew[:, 1] = y

                # Move to the next point
                queue.submit(arr_pos, arr_col, loop=True)
                # Get the brightness
                brightness = sample_camera(cap)
                line_arr.append(brightness)

            # Rewind the galvos
            queue.submit(arr_pos_rew, arr_col_rew)
            raw_img_arr.append(line_arr)
            print('\n\n')
                

        print("\n✅ Scan complete. Processing image...")
        
        img = np.stack(raw_img_arr, axis=0)
        print(f"Image processed. Final shape: {img.shape}")

        # --- Display Result ---
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='viridis', extent=[xy_min, xy_max, xy_min, xy_max])
        plt.title('Dual Photography Scan (Webcam)')
        plt.xlabel('Horizontal Scan')
        plt.ylabel('Vertical Scan')
        plt.colorbar(label='Measured Brightness')
        plt.show()

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    finally:
        # --- Cleanup ---
        if cap:
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("Script finished.")


