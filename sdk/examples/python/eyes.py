#!/usr/bin/env python
# coding: utf-8

# In[1]:


import laser_lib
import numpy as np
import math
import time


# In[2]:


queue = laser_lib.DacQueue()


# In[4]:


import laser_lib
import numpy as np
import math
import time
import random
import os
import pygame

# --- Pygame & Controller Setup ---
# Prevent Pygame from opening a window
os.environ["SDL_VIDEODRIVER"] = "dummy" 
pygame.init()
pygame.joystick.init()

joystick = None
if pygame.joystick.get_count() == 0:
    print("No Xbox controller found! Defaulting to auto-darting.")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Connected to: {joystick.get_name()}")

# --- Settings ---
queue.dac_rate = 30000
points_per_unit = 400
r_outer, r_inner = 0.4, 0.1
brow_w, brow_h = 0.4, 0.15
transition_points = 20
laser_offset = -1
overshoot_iris = 0.0275
overshoot_pupil = 0.05
offset_distance = 0.6
pupil_height = r_outer * 0.45

# Blink Settings
blink_interval = 4.0    
blink_duration = 0.2    
brow_drop = 0.15        

# Saccade & Mode States
max_angle = np.deg2rad(45) 
target_x_l, target_y_l = 0.0, 0.0
current_x_l, current_y_l = 0.0, 0.0
target_x_r, target_y_r = 0.0, 0.0
current_x_r, current_y_r = 0.0, 0.0
lerp_speed = 0.35          

# Auto-mode synchronous darting variables
auto_target_x, auto_target_y = 0.0, 0.0
next_auto_dart_time = time.time() + 2.0

# Mode Toggles & Edge Detection State
independent_gaze = True
auto_mode = False
prev_lb = False
prev_rb = False
prev_back = False
prev_start = False

# Random Mad State for Auto-Mode
next_random_mad_check = time.time() + 3.0
random_mad_active_until = 0.0

# Independent Manual Blink Trigger State Tracking & Auto-Blink Scheduling
manual_blink_start_l = -999.0
manual_blink_start_r = -999.0
auto_blink_start = -999.0
next_auto_blink_time = time.time() + blink_interval

# Color State
user_selected_color = np.array([1.0, 1.0, 1.0]) # Default to White

# --- 1. Generate Local Geometry ---
t_out = int(points_per_unit * r_outer)
t_in = int(points_per_unit * r_inner)

theta_out = np.linspace(-overshoot_iris * 2*np.pi, 2*np.pi + 2*np.pi * overshoot_iris, t_out)
pts_out = np.column_stack([r_outer * np.cos(theta_out), r_outer * np.sin(theta_out), np.zeros(t_out)])

theta_in = np.linspace(-overshoot_pupil * 2*np.pi, 2*np.pi + 2*np.pi * overshoot_pupil, t_in)
pts_in = np.column_stack([r_inner * np.cos(theta_in), r_inner * np.sin(theta_in), pupil_height * np.ones(t_in)])

t_brow = int(points_per_unit * brow_w)
theta_brow = np.linspace(0.2, 0.8 * np.pi, t_brow)
pts_brow = np.column_stack([brow_w * np.cos(theta_brow), 0.6 + brow_h * np.sin(theta_brow), np.zeros(t_brow)])

# --- 2. Main Loop ---
while True:
    t = time.time()

    # --- 2a. Controller Event Pump & Mode Toggles ---
    pygame.event.pump() 
    
    if joystick is not None:
        try:
            back_pressed = joystick.get_button(6)
            start_pressed = joystick.get_button(7)
            
            # Toggle Gaze Coupling (Back Button)
            if back_pressed and not prev_back:
                independent_gaze = not independent_gaze
                print(f"Independent Gaze Mode: {independent_gaze}")
            prev_back = back_pressed
            
            # Toggle Auto-Mode (Start Button)
            if start_pressed and not prev_start:
                auto_mode = not auto_mode
                print(f"Auto-Mode: {auto_mode}")
            prev_start = start_pressed
        except Exception:
            pass

    # --- 2b. Gaze & Expression Logic (Auto vs Manual) ---
    deadzone = 0.15
    mad_l = 0.0
    mad_r = 0.0

    if auto_mode or joystick is None:
        # Synchronous Automatic Darting Gaze
        if t > next_auto_dart_time:
            auto_target_x = random.uniform(-max_angle, max_angle)
            auto_target_y = random.uniform(-max_angle, max_angle)
            next_auto_dart_time = t + random.uniform(0.5, 3.0)
            
        target_x_l = auto_target_x
        target_y_l = auto_target_y
        target_x_r = auto_target_x
        target_y_r = auto_target_y
            
        # Random Mad Generator in Auto-Mode (In-sync)
        if t > next_random_mad_check:
            if random.random() < 0.4: # 40% chance to get mad
                random_mad_active_until = t + random.uniform(1.0, 3.0)
            next_random_mad_check = t + random.uniform(4.0, 9.0)
            
        if t < random_mad_active_until:
            mad_l = 1.0
            mad_r = 1.0
            
    else:
        # Manual Joystick Control
        l_sx = joystick.get_axis(1)
        l_sy = joystick.get_axis(0)
        if abs(l_sx) < deadzone: l_sx = 0.0
        if abs(l_sy) < deadzone: l_sy = 0.0
        
        if independent_gaze:
            target_x_l = l_sx * max_angle
            target_y_l = l_sy * max_angle
            
            r_sx = joystick.get_axis(4)
            r_sy = joystick.get_axis(3)
            if abs(r_sx) < deadzone: r_sx = 0.0
            if abs(r_sy) < deadzone: r_sy = 0.0
            target_x_r = r_sx * max_angle
            target_y_r = r_sy * max_angle
        else:
            # Coupled: Left stick controls both eyes simultaneously
            target_x_l = l_sx * max_angle
            target_y_l = l_sy * max_angle
            target_x_r = l_sx * max_angle
            target_y_r = l_sy * max_angle

    current_x_l += (target_x_l - current_x_l) * lerp_speed
    current_y_l += (target_y_l - current_y_l) * lerp_speed
    current_x_r += (target_x_r - current_x_r) * lerp_speed
    current_y_r += (target_y_r - current_y_r) * lerp_speed

    # Create separate rotation matrices for each eye
    cx_l, sx_l = np.cos(current_x_l), np.sin(current_x_l)
    cy_l, sy_l = np.cos(current_y_l), np.sin(current_y_l)
    rx_l = np.array([[1, 0, 0], [0, cx_l, -sx_l], [0, sx_l, cx_l]])
    ry_l = np.array([[cy_l, 0, sy_l], [0, 1, 0], [-sy_l, 0, cy_l]])
    R_l = (ry_l @ rx_l).T

    cx_r, sx_r = np.cos(current_x_r), np.sin(current_x_r)
    cy_r, sy_r = np.cos(current_y_r), np.sin(current_y_r)
    rx_r = np.array([[1, 0, 0], [0, cx_r, -sx_r], [0, sx_r, cx_r]])
    ry_r = np.array([[cy_r, 0, sy_r], [0, 1, 0], [-sy_r, 0, cy_r]])
    R_r = (ry_r @ rx_r).T

    # --- 3. Controller Inputs (Buttons & Triggers - Manual Mode Only) ---
    if joystick is not None:
        try:
            lb_pressed = joystick.get_button(4)
            rb_pressed = joystick.get_button(5)
            
            # Face Buttons for User Color Selection with Mixing Support
            mixed_color = np.array([0.0, 0.0, 0.0])
            buttons_pressed = 0

            if joystick.get_button(0):  # A Button -> Green
                mixed_color += np.array([0.0, 1.0, 0.0])
                buttons_pressed += 1
            if joystick.get_button(1): # B Button -> Red
                mixed_color += np.array([1.0, 0.0, 0.0])
                buttons_pressed += 1
            if joystick.get_button(2): # X Button -> Blue
                mixed_color += np.array([0.0, 0.0, 1.0])
                buttons_pressed += 1
            if joystick.get_button(3): # Y Button -> Yellow
                mixed_color += np.array([1.0, 1.0, 0.0])
                buttons_pressed += 1

            # Update color if at least one color button is pressed
            if buttons_pressed > 0:
                user_selected_color = np.clip(mixed_color, 0.0, 1.0)

            # Triggers
            if not auto_mode:
                lt_raw = joystick.get_axis(2)
                rt_raw = joystick.get_axis(5)
                mad_l = np.clip((lt_raw + 1.0) / 2.0, 0.0, 1.0)
                mad_r = np.clip((rt_raw + 1.0) / 2.0, 0.0, 1.0)
            
            # Left/Right Bumper Blink Triggers
            if lb_pressed and not prev_lb:
                if (t - manual_blink_start_l) > blink_duration:
                    manual_blink_start_l = t
                    next_auto_blink_time = max(next_auto_blink_time, t + blink_interval)
                    
            if rb_pressed and not prev_rb:
                if (t - manual_blink_start_r) > blink_duration:
                    manual_blink_start_r = t
                    next_auto_blink_time = max(next_auto_blink_time, t + blink_interval)
                    
            prev_lb = lb_pressed
            prev_rb = rb_pressed
        except Exception:
            pass 

    # Determine active color
    if mad_l > 0.95 and mad_r > 0.95:
        active_color = np.array([1.0, 0.0, 0.0])
    else:
        active_color = user_selected_color

    # Scheduled Auto-Blink Logic
    if t >= next_auto_blink_time:
        auto_blink_start = t
        next_auto_blink_time = t + blink_interval

    auto_elapsed = t - auto_blink_start
    auto_blink = np.sin((auto_elapsed / blink_duration) * np.pi) if auto_elapsed < blink_duration else 0.0

    manual_elapsed_l = t - manual_blink_start_l
    manual_blink_l = np.sin((manual_elapsed_l / blink_duration) * np.pi) if manual_elapsed_l < blink_duration else 0.0

    manual_elapsed_r = t - manual_blink_start_r
    manual_blink_r = np.sin((manual_elapsed_r / blink_duration) * np.pi) if manual_elapsed_r < blink_duration else 0.0

    base_blink_l = max(auto_blink, manual_blink_l)
    base_blink_r = max(auto_blink, manual_blink_r)

    # --- 4. Assembly (Local modification -> Rotate -> World Translation) ---
    
    # -- LEFT EYE --
    # Apply rotation ONLY to the pupil (inner circle)
    pupil_l = pts_in.copy() @ R_l
    lid_l = pts_out.copy() # Lids stay completely stationary 
    
    # Build transition from stationary lid to rotated pupil dynamically
    trans_l = np.linspace(lid_l[-1], pupil_l[0], transition_points)
    eye_l_local = np.vstack([lid_l, trans_l, pupil_l])
    
    brow_l_local = pts_brow.copy()
    
    # 1. Standard Uniform Dual-Direction Blink
    top_lid_y_l = r_outer * (1.0 - base_blink_l)
    bot_lid_y_l = -r_outer * (1.0 - base_blink_l)
    eye_l_local[:, 1] = np.clip(eye_l_local[:, 1], bot_lid_y_l, top_lid_y_l)
    brow_l_local[:, 1] -= (brow_drop * base_blink_l)
    
    # 2. Mad Expression (Sloped Squint towards middle + Eyebrow Tilt)
    if mad_l > 0.01:
        x_rel_l = eye_l_local[:, 0] / r_outer
        weight_l = 0.2 + 0.8 * np.clip((x_rel_l + 1.0) / 2.0, 0.0, 1.0)
        mad_squint_l = mad_l * 0.3 * weight_l
        
        mad_top_l = r_outer * (1.0 - mad_squint_l)
        mad_bot_l = -r_outer * (1.0 - mad_squint_l)
        eye_l_local[:, 1] = np.clip(eye_l_local[:, 1], mad_bot_l, mad_top_l)
        
        brow_l_local[:, 1] -= (brow_drop * mad_l)
        b_center_l = np.mean(brow_l_local, axis=0)
        brow_l_shifted = brow_l_local - b_center_l
        theta_mad_l = mad_l * np.deg2rad(22) 
        c, s = np.cos(theta_mad_l), np.sin(theta_mad_l)
        rot_z_l = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        brow_l_local = (brow_l_shifted @ rot_z_l) + b_center_l

    # Translation to world space (no rotation here)
    eye_l_world = eye_l_local - [offset_distance, 0, 0]
    brow_l_world = brow_l_local - [offset_distance, 0, 0]

    # -- RIGHT EYE --
    # Apply rotation ONLY to the pupil (inner circle)
    pupil_r = pts_in.copy() @ R_r
    lid_r = pts_out.copy() # Lids stay completely stationary 
    
    # Build transition from stationary lid to rotated pupil dynamically
    trans_r = np.linspace(lid_r[-1], pupil_r[0], transition_points)
    eye_r_local = np.vstack([lid_r, trans_r, pupil_r])
    
    brow_r_local = pts_brow.copy()
    
    # Mirror the right eyebrow in local X space
    brow_r_local[:, 0] *= -1 
    
    # 1. Standard Uniform Dual-Direction Blink
    top_lid_y_r = r_outer * (1.0 - base_blink_r)
    bot_lid_y_r = -r_outer * (1.0 - base_blink_r)
    eye_r_local[:, 1] = np.clip(eye_r_local[:, 1], bot_lid_y_r, top_lid_y_r)
    brow_r_local[:, 1] -= (brow_drop * base_blink_r)
    
    # 2. Mad Expression (Sloped Squint towards middle + Eyebrow Tilt)
    if mad_r > 0.01:
        x_rel_r = eye_r_local[:, 0] / r_outer
        weight_r = 0.2 + 0.8 * (1.0 - np.clip((x_rel_r + 1.0) / 2.0, 0.0, 1.0))
        mad_squint_r = mad_r * 0.3 * weight_r
        
        mad_top_r = r_outer * (1.0 - mad_squint_r)
        mad_bot_r = -r_outer * (1.0 - mad_squint_r)
        eye_r_local[:, 1] = np.clip(eye_r_local[:, 1], mad_bot_r, mad_top_r)
        
        brow_r_local[:, 1] -= (brow_drop * mad_r)
        b_center_r = np.mean(brow_r_local, axis=0)
        brow_r_shifted = brow_r_local - b_center_r
        theta_mad_r = -mad_r * np.deg2rad(22) 
        c, s = np.cos(theta_mad_r), np.sin(theta_mad_r)
        rot_z_r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        brow_r_local = (brow_r_shifted @ rot_z_r) + b_center_r

    # Translation to world space (no rotation here)
    eye_r_world = eye_r_local + [offset_distance, 0, 0]
    brow_r_world = brow_r_local + [offset_distance, 0, 0]

    # Stitch the vectors together
    t_l_to_brow = np.linspace(eye_l_world[-1], brow_l_world[0], transition_points)
    t_brow_to_r = np.linspace(brow_l_world[-1], eye_r_world[0], transition_points)
    t_r_to_brow = np.linspace(eye_r_world[-1], brow_r_world[0], transition_points)
    t_wrap = np.linspace(brow_r_world[-1], eye_l_world[0], transition_points)

    full_frame_3d = np.vstack([
        eye_l_world, t_l_to_brow, brow_l_world, t_brow_to_r,
        eye_r_world, t_r_to_brow, brow_r_world, t_wrap
    ])

    # --- 5. Color Mapping ---
    arr_col = np.zeros((len(full_frame_3d), 3))
    curr = 0
    for _ in range(2): 
        arr_col[curr + 3 : curr + t_out - 3, :] = active_color
        curr += t_out + transition_points 
        arr_col[curr + 3 : curr + t_in - 3, :] = active_color
        curr += t_in + transition_points
        arr_col[curr + 3 : curr + t_brow - 3, :] = active_color
        if _ == 0: curr += t_brow + transition_points

    arr_col = np.roll(arr_col, -laser_offset, axis=0)
    queue.submit(full_frame_3d[:, :2], 0.1*arr_col, loop=False, angular_density=1000)


# In[ ]:





# In[ ]:




