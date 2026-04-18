from osim.env import L2M2019Env
import numpy as np

print("1. Initializing Environment...")
env = L2M2019Env(visualize=True, difficulty=0)

print("2. Resetting to Static Pose...")
# Get initial observation
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
observation = env.reset(init_pose=init_pose)

# Create an action vector of all zeros (no muscle activation)
zeros_action = np.zeros(env.action_space.shape)

print("3. Displaying Static Model (Close the window or press Ctrl+C to stop)...")
for i in range(500):

    observation, reward, done, info = env.step(zeros_action)
    break