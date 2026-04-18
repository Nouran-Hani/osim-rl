from osim.env import ProstheticsEnv

print("1. Initializing Environment...")
# Start with visualization OFF to save memory while debugging
env = ProstheticsEnv(visualize=True)

print("2. Attempting Reset...")
observation = env.reset()

# The reward loop
print("3. Running simulation steps...")
total_reward = 0.0
for i in range(1):
    # Use a random action instead of my_controller to test stability
    action = env.action_space.sample() 
    print(action)
    observation, reward, done, info = env.step(action)
    print(f"Step {i}: Reward={reward:.4f}, Done={done}")
    total_reward += reward

    if done:
        break

print("4. Finished!")
print("Total reward %f" % total_reward)

