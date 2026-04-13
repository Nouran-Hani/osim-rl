from osim.env import ProstheticsEnv

print("1. Initializing Environment...")
# Start with visualization OFF to save memory while debugging
env = ProstheticsEnv(visualize=True)

print("2. Attempting Reset...")
observation = env.reset()

# try:
#     # We call reset() - this is where the LAPACK crash usually happens
#     observation = env.reset()
#     print("Success! Environment is ready.")
# except Exception as e:
#     print(f"Failed at reset. Error: {e}")

# The reward loop
print("3. Running simulation steps...")
total_reward = 0.0
for i in range(200):
    # Use a random action instead of my_controller to test stability
    action = env.action_space.sample() 
    observation, reward, done, info = env.step(action)
    total_reward += reward
    if i % 50 == 0:
        print(f"   ...at step {i}")
    if done:
        break

print("4. Finished!")
print("Total reward %f" % total_reward)