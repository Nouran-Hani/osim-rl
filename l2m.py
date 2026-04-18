from osim.env import L2M2019Env
import math
import numpy as np

print("Initializing OpenSim Environment...")

# Create the environment and turn on the 3D visualizer
env = L2M2019Env(visualize=True)

# Reset the environment to the starting position
observation = env.reset()

print(f"Success! The AI receives {len(observation)} sensor readings.")
print(f"The AI controls {env.action_space.shape[0]} muscles/motors.")
print("Starting simulation loop...")

for step in range(1000):
    # Create a rhythmic signal between 0 and 1
    # frequency controls speed, math.sin creates the oscillation
    rhythm = (math.sin(step * 0.05) + 1) / 2 
    
    action = np.zeros(env.action_space.shape)
    # action = env.action_space.sample()  # Start with random actions (can be replaced with a policy later)
    # Apply the rhythm to hip flexors and knee extensors
    action[2] = rhythm      # Right Hip Flexion
    action[9] = 1 - rhythm  # Left Hip Flexion (alternating)
    
    observation, reward, done, info = env.step(action)
    
    # If the model falls over, the environment flags 'done=True'
    if done:
        print(f"Model fell over at step {step}! Resetting...")
        observation = env.reset()

print("Simulation finished successfully!")