from stable_baselines3 import PPO
from osim.env import ProstheticsEnv
from Wrapper import OsimWrapper

# 1. Initialize the raw environment
print("1. Initializing Environment...")
# 1. Raw OpenSim
raw_env = ProstheticsEnv(visualize=False)

raw_env.change_model(model="3D", prosthetic=False)  # Switch to the prosthetic model

# 2. Your Smart Wrapper
wrapped_env = OsimWrapper(raw_env)

# 3. PPO initialization
print("2. Setting up PPO Algorithm...")
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# 4. Start Training
print("3. Starting Training...")
model.learn(total_timesteps=1000)

# 4. Save the trained brain
model_name = "ppo_prosthetic_agent"
model.save(model_name)
print(f"Model saved as {model_name}")

# 5. Testing the trained agent (The Loop)
print("4. Testing the trained agent...")
env_test = ProstheticsEnv(visualize=True) # Turn on visualization for the demo
obs = env_test.reset()
total_reward = 0.0

for i in range(1000):
    # Instead of action_space.sample(), we ask the model for the best move
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env_test.step(action)
    total_reward += reward
    
    if done:
        break

print(f"Test Finished! Total reward: {total_reward}")