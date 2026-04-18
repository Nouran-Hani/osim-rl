# import pandas as pd
# import numpy as np
# from osim.env import L2M2019Env

# csv_path = "D:\\Projects\\osim-rl\\muscle_activations.csv"
# data = pd.read_csv(csv_path)

# # This is the CORRECT order — matches what L2M2019Env.step() expects BEFORE act2mus remapping
# # Source: osim.py dict_muscle and the act2mus comment block
# L2M_API_ORDER = [
#     "abd_r",        # HAB
#     "add_r",        # HAD
#     "iliopsoas_r",  # HFL  ← index 2 in API, but index 5 in .osim file
#     "glut_max_r",   # GLU  ← index 3 in API, but index 4 in .osim file
#     "hamstrings_r", # HAM  ← index 4 in API, but index 2 in .osim file
#     "rect_fem_r",   # RF
#     "vasti_r",      # VAS
#     "bifemsh_r",    # BFSH
#     "gastroc_r",    # GAS
#     "soleus_r",     # SOL
#     "tib_ant_r",    # TA
#     "abd_l",
#     "add_l",
#     "iliopsoas_l",
#     "glut_max_l",
#     "hamstrings_l",
#     "rect_fem_l",
#     "vasti_l",
#     "bifemsh_l",
#     "gastroc_l",
#     "soleus_l",
#     "tib_ant_l"
# ]

# # Build activation matrix using the API order
# activation_matrix = np.zeros((len(data), len(L2M_API_ORDER)))
# for idx, muscle in enumerate(L2M_API_ORDER):
#     if muscle in data.columns:
#         activation_matrix[:, idx] = data[muscle].values
#     else:
#         print(f"WARNING: {muscle} not found in CSV!")

# n_frames = len(activation_matrix)

# env = L2M2019Env(visualize=True, difficulty=0)
# # env.reset()

# print("Starting Continuous Playback...")
# for i in range(len(data)):
#     current_row = i % n_frames

#     blend_window = 5
#     if current_row < blend_window and i >= n_frames:
#         alpha = current_row / blend_window
#         action = ((1 - alpha) * activation_matrix[-1] +
#                    alpha      * activation_matrix[current_row])
#     else:
#         action = activation_matrix[current_row].copy()

#     observation, reward, done, info = env.step(action)

#     if done:
#         print(f"Model fell at step {i}. Resetting...")
#         # env.reset()
#         break

# print("Playback finished.")

import pandas as pd
import numpy as np
from osim.env import L2M2019Env

csv_path = "D:\\Projects\\osim-rl\\muscle_activations.csv"
data = pd.read_csv(csv_path)

deg = np.pi / 180  # conversion factor

init_pose = np.array([
    1.25,                    # [0] forward speed (keep as is)
    0.0,                     # [1] rightward speed (keep as is)
    0.960,                   # [2] pelvis_ty — height (already in meters, use directly)

    -1.870 * deg,            # [3] trunk lean  ← pelvis_tilt
    
    # RIGHT LEG
     1.090 * deg,            # [4] right hip adduct  ← hip_adduction_r
     24.610 * deg,           # [5] right hip flex    ← hip_flexion_r
    -3.940 * deg,            # [6] right knee extend ← knee_angle_r
    -1.700 * deg,            # [7] right ankle flex  ← ankle_angle_r
    
    # LEFT LEG
     2.680 * deg,            # [8] left hip adduct   ← hip_adduction_l
    -16.560 * deg,           # [9] left hip flex     ← hip_flexion_l
    -8.200 * deg,            # [10] left knee extend ← knee_angle_l
     9.810 * deg,            # [11] left ankle flex  ← ankle_angle_l
])

L2M_API_ORDER = [
    "abd_r", "add_r", "iliopsoas_r", "glut_max_r", "hamstrings_r",
    "rect_fem_r", "vasti_r", "bifemsh_r", "gastroc_r", "soleus_r", "tib_ant_r",
    "abd_l", "add_l", "iliopsoas_l", "glut_max_l", "hamstrings_l",
    "rect_fem_l", "vasti_l", "bifemsh_l", "gastroc_l", "soleus_l", "tib_ant_l"
]

activation_matrix = np.zeros((len(data), len(L2M_API_ORDER)))
for idx, muscle in enumerate(L2M_API_ORDER):
    if muscle in data.columns:
        activation_matrix[:, idx] = data[muscle].values
    else:
        print(f"WARNING: {muscle} not found in CSV!")

n_frames = len(activation_matrix)

env = L2M2019Env(visualize=True, difficulty=0)
env.reset(init_pose=init_pose)  # ← pass the matched initial pose here

print("Starting Continuous Playback...")
for i in range(500):
    current_row = i % n_frames

    blend_window = 5
    if current_row < blend_window and i >= n_frames:
        alpha = current_row / blend_window
        action = ((1 - alpha) * activation_matrix[-1] +
                   alpha      * activation_matrix[current_row])
    else:
        action = activation_matrix[current_row].copy()

    observation, reward, done, info = env.step(action)

    if done:
        print(f"Model fell at step {i}. Resetting...")
        env.reset(init_pose=init_pose)  # ← also here on reset

print("Playback finished.")