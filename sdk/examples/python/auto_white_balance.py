import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import datetime as dt
import laser_lib
import math
import time

queue = laser_lib.DacQueue()
queue.dac_rate = 50000


# --- Assume set_color is provided ---
# (Remove or replace this block with your actual function import/definition)
def set_color(rgb, queue, T):
    t = np.linspace(0, 1, T)
    arr_pos = np.zeros((T, 2))
    arr_col = rgb[None, :] > t[:, None]
    queue.submit(arr_pos, arr_col, loop=True)

# --- Configuration ---
MAX_POINTS = 100
UPDATE_INTERVAL_MS = 100

# --- Control Loop Configuration ---
TARGET_BRIGHTNESS = 128.0  # Target exposure value for all channels (0-255)
KP = 0.005                 # Proportional gain (Adjust if the adjustment is too slow or too jumpy)
current_light = np.array([0.5, 0.5, 0.5]) # Initial light brightness (R, G, B)

# --- Webcam Initialization ---
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Error opening webcam: {e}")
    print("Using random data for demonstration purposes.")
    cap = None

# --- Data Storage ---
r_data = deque(maxlen=MAX_POINTS)
g_data = deque(maxlen=MAX_POINTS)
b_data = deque(maxlen=MAX_POINTS)
time_data = deque(maxlen=MAX_POINTS)

# --- Plot Initialization ---
fig, ax = plt.subplots(figsize=(10, 5))
line_r, = ax.plot([], [], 'r-', label='Red Channel')
line_g, = ax.plot([], [], 'g-', label='Green Channel')
line_b, = ax.plot([], [], 'b-', label='Blue Channel')

# --- Plot Styling ---
def setup_plot():
    ax.set_ylim(0, 255)  
    ax.set_xlim(0, MAX_POINTS)
    ax.set_title('Real-time RGB Channel Brightness', fontsize=16)
    ax.set_ylabel('Average Value (0-255)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.grid(True)
    ax.legend(loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.20)

# --- Animation Update Function ---
def update(frame_number):
    global current_light # Required to update the persistent lighting state
    
    if cap:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            ani.event_source.stop()
            return line_r, line_g, line_b

        # OpenCV reads frames in BGR format
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        
        # --- Lighting Control Loop ---
        # 1. Arrange webcam means into standard RGB order
        cam_rgb = np.array([r_mean, g_mean, b_mean])
        
        # 2. Calculate error (Target - Current Camera Value)
        # If the camera is darker than 128, the error is positive.
        error = TARGET_BRIGHTNESS - cam_rgb
        
        # 3. Update the light state based on the error
        current_light = current_light + (KP * error)
        
        # 4. Clamp values to ensure they stay strictly between 0.0 and 1.0
        current_light = np.clip(current_light, 0.0, 1.0)
        
        # 5. Push the new colors to your hardware
        global queue
        set_color(current_light, queue, T=100)
        
    else:
        # Random Data Generation
        r_mean = np.random.randint(50, 255)
        g_mean = np.random.randint(50, 255)
        b_mean = np.random.randint(50, 255)

    # --- Data Handling ---
    r_data.append(r_mean)
    g_data.append(g_mean)
    b_data.append(b_mean)
    time_data.append(dt.datetime.now().strftime('%H:%M:%S'))

    # --- Plot Update ---
    x_data = np.arange(len(r_data))
    line_r.set_data(x_data, r_data)
    line_g.set_data(x_data, g_data)
    line_b.set_data(x_data, b_data)

    ax.set_xticks(np.arange(len(time_data)))
    ax.set_xticklabels(time_data, rotation=45, ha='right')
    ax.set_xlim(0, len(r_data) - 1 if len(r_data) > 1 else 1)

    fig.canvas.draw()
    fig.canvas.flush_events()

    return line_r, line_g, line_b

# --- Main Execution ---
if __name__ == '__main__':
    setup_plot()

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=UPDATE_INTERVAL_MS,
        blit=True,
        cache_frame_data=False 
    )

    plt.show()

    # --- Cleanup ---
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Script finished and resources released.")
