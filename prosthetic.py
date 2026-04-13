import gym
import numpy as np

class OsimWrapper(gym.Wrapper):
    def __init__(self, env):
        super(OsimWrapper, self).__init__(env)
        self.env = env
        
        # 1. Run a dummy reset to see how long the flat vector actually is
        raw_obs = self.env.reset()
        flat_obs = self.flatten_obs(raw_obs)
        self.output_dim = len(flat_obs)
        
        # 2. Update the observation space so SB3 knows what to expect
        # We define a new Box space with the correct shape (412,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.output_dim,), 
            dtype=np.float32
        )
        print(f"Observation space adjusted: New shape is ({self.output_dim},)")

    def flatten_obs(self, item):
        """Recursively turns nested dicts/lists into a single flat list of floats."""
        res = []
        if isinstance(item, dict):
            for v in item.values():
                res.extend(self.flatten_obs(v))
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            for v in item:
                res.extend(self.flatten_obs(v))
        elif isinstance(item, (int, float, np.float32, np.float64)):
            res.append(item)
        return res

    def reset(self, **kwargs):
        result = self.env.reset()
        
        # Handle the (obs, info) tuple if it exists
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs = result
            info = {}

        # Flatten the dictionary mess into a simple list of numbers
        flat_obs = self.flatten_obs(raw_obs)
        return np.array(flat_obs, dtype=np.float32), info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        
        # Flatten here too!
        flat_obs = self.flatten_obs(raw_obs)
        
        terminated = done
        truncated = False
        return np.array(flat_obs, dtype=np.float32), reward, terminated, truncated, info

from stable_baselines3 import PPO
from osim.env import ProstheticsEnv

# 1. Initialize the raw environment
print("1. Initializing Environment...")
# 1. Raw OpenSim
raw_env = ProstheticsEnv(visualize=False)

# 2. Your Smart Wrapper
wrapped_env = OsimWrapper(raw_env)

# 3. PPO initialization
print("2. Setting up PPO Algorithm...")
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# 4. Start Training
print("3. Starting Training...")
model.learn(total_timesteps=100000)

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