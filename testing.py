import gym  # Changed from gymnasium!
import numpy as np
from stable_baselines3 import PPO
from osim.env import ProstheticsEnv

# ==========================================
# 1. THE WRAPPER (Old Gym Format)
# ==========================================
class OsimWrapper(gym.Wrapper):
    def __init__(self, env):
        super(OsimWrapper, self).__init__(env)
        self.env = env
        
        # The shape the model expects (from your error message)
        self.target_shape = 468 
        
        # Update the observation space to match what the model expects
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.target_shape,), dtype=np.float32
        )

    def flatten_obs(self, item):
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

    def fix_shape(self, flat_obs):
        """Pads or clips the observation to match the model's expected 468."""
        flat_obs = np.array(flat_obs, dtype=np.float32)
        current_shape = flat_obs.shape[0]
        
        if current_shape < self.target_shape:
            # Add zeros if too small
            return np.pad(flat_obs, (0, self.target_shape - current_shape), 'constant')
        elif current_shape > self.target_shape:
            # Cut off extra values if too big
            return flat_obs[:self.target_shape]
        return flat_obs

    def reset(self, **kwargs):
        raw_obs = self.env.reset()
        flat_obs = self.flatten_obs(raw_obs)
        return self.fix_shape(flat_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        flat_obs = self.flatten_obs(raw_obs)
        return self.fix_shape(flat_obs), reward, done, info
# ==========================================
# 2. TESTING PHASE
# ==========================================
if __name__ == "__main__":
    print("1. Initializing Environment for Testing...")
    
    # Keeping visualize=False for a quick blind test
    raw_test_env = ProstheticsEnv(visualize=True)
    
    # Wrap the environment
    test_env = OsimWrapper(raw_test_env)

    print("2. Loading the trained agent...")
    model = PPO.load("ppo_prosthetic_agent") 

    print("3. Starting the Test Loop...")
    # Old gym format: reset only returns ONE variable
    obs = test_env.reset()
    total_reward = 0.0

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        
        # Old gym format: step returns FOUR variables
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        
        if done:
            obs = test_env.reset()

    print(f"Test Finished Successfully! Total reward: {total_reward}")