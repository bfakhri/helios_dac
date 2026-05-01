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
def set_color(rgb, queue, T):
    t = np.linspace(0, 1, T)
    arr_pos = np.zeros((T, 2))
    arr_col = (rgb[None, :] > t[:, None]).astype(float)
    queue.submit(arr_pos, arr_col, loop=True)

# --- Configuration ---
MAX_POINTS = 100
UPDATE_INTERVAL_MS = 100
TARGET_BRIGHTNESS = 128.0  
KP = 0.005                 
current_light = np.array([0.5, 0.5, 0.5]) 

# --- Webcam Initialization ---
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Error opening webcam: {e}")
    cap = None

# --- Data Storage ---
r_cam, g_cam, b_cam = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)
r_light, g_light, b_light = deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS), deque(maxlen=MAX_POINTS)
time_data = deque(maxlen=MAX_POINTS)

# --- Plot Initialization ---
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()  # Create secondary Y-axis

# Camera Lines (Solid)
line_r_cam, = ax1.plot([], [], 'r-', label='Cam Red', linewidth=1)
line_g_cam, = ax1.plot([], [], 'g-', label='Cam Green', linewidth=1)
line_b_cam, = ax1.plot([], [], 'b-', label='Cam Blue', linewidth=1)

# Light Control Lines (Dashed/Thicker)
line_r_lit, = ax2.plot([], [], 'r--', label='Light Red (Cmd)', alpha=0.6)
line_g_lit, = ax2.plot([], [], 'g--', label='Light Green (Cmd)', alpha=0.6)
line_b_lit, = ax2.plot([], [], 'b--', label='Light Blue (Cmd)', alpha=0.6)

def setup_plot():
    ax1.set_ylim(0, 255)  
    ax2.set_ylim(-0.1, 1.1) # Pad slightly to see 0 and 1 clearly
    
    ax1.set_title('Webcam Feedback vs. Laser Light Command', fontsize=16)
    ax1.set_ylabel('Camera Brightness (0-255)', color='black', fontsize=12)
    ax2.set_ylabel('Light Control Signal (0.0-1.0)', color='blue', fontsize=12)
    ax1.set_xlabel('Time', fontsize=12)
    
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2)
    
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.20, right=0.85)

# --- Additional Control Configuration ---
MAX_CHANGE_RATE = 0.04   # Max change allowed per update (0.0 to 1.0 scale)
SMOOTHING_FACTOR = 0.1  # EMA filter (1.0 = no smoothing, 0.1 = very heavy smoothing)

# --- Updated Animation Update Function ---
def update(frame_number):
    global current_light 
    
    if cap:
        ret, frame = cap.read()
        if not ret:
            return line_r_cam, line_g_cam, line_b_cam, line_r_lit, line_g_lit, line_b_lit

        # 1. Get raw camera means
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        cam_rgb = np.array([r_mean, g_mean, b_mean])
        
        # 2. Calculate the "Ideal" next step based on KP
        error = TARGET_BRIGHTNESS - cam_rgb
        target_step = KP * error
        
        # 3. Apply Slew Rate Limiting (Damping the "jump")
        # This ensures the light doesn't change more than MAX_CHANGE_RATE in one frame
        damped_step = np.clip(target_step, -MAX_CHANGE_RATE, MAX_CHANGE_RATE)
        
        # 4. Apply Exponential Moving Average (EMA) Filtering
        # This smooths the transition: NewValue = (α * Target) + ((1-α) * OldValue)
        next_light_raw = current_light + damped_step
        current_light = (SMOOTHING_FACTOR * next_light_raw) + ((1 - SMOOTHING_FACTOR) * current_light)
        
        # 5. Final Clamp and Hardware Push
        current_light = np.clip(current_light, 0.0, 1.0)
        
        global queue
        set_color(current_light, queue, T=100)
        
    else:
        # Mock data for demonstration
        r_mean, g_mean, b_mean = np.random.randint(120, 140, 3)
        current_light = np.clip(current_light + np.random.uniform(-0.01, 0.01, 3), 0, 1)

    # --- Data Handling & Plotting remain the same ---
    r_cam.append(r_mean); g_cam.append(g_mean); b_cam.append(b_mean)
    r_light.append(current_light[0]); g_light.append(current_light[1]); b_light.append(current_light[2])
    time_data.append(dt.datetime.now().strftime('%H:%M:%S'))

    x_data = np.arange(len(r_cam))
    line_r_cam.set_data(x_data, r_cam)
    line_g_cam.set_data(x_data, g_cam)
    line_b_cam.set_data(x_data, b_cam)
    line_r_lit.set_data(x_data, r_light)
    line_g_lit.set_data(x_data, g_light)
    line_b_lit.set_data(x_data, b_light)

    ax1.set_xticks(np.arange(len(time_data)))
    ax1.set_xticklabels(time_data, rotation=45, ha='right')
    ax1.set_xlim(0, max(1, len(r_cam) - 1))

    return line_r_cam, line_g_cam, line_b_cam, line_r_lit, line_g_lit, line_b_lit

if __name__ == '__main__':
    setup_plot()
    ani = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL_MS, blit=True, cache_frame_data=False 
    )
    plt.show()

    if cap: cap.release()
    cv2.destroyAllWindows()
