from osim.env import ProstheticsEnv
import numpy as np

print("1. Initializing Environment...")
env = ProstheticsEnv(visualize=True)

print("2. Resetting to Static Pose...")
# Get initial observation
observation = env.reset()

# Create an action vector of all zeros (no muscle activation)
zeros_action = np.zeros(env.action_space.shape)

print("3. Displaying Static Model (Close the window or press Ctrl+C to stop)...")
try:
    while True:
        # 1. Take a tiny step to refresh the visualizer
        # We use the zeros_action so no muscles pull
        env.step(zeros_action)
        
        # 2. IMMEDIATELY reset the internal state back to the start
        # This "freezes" the model in the reset position
        env.reset()
        
except KeyboardInterrupt:
    print("Stopping display.")