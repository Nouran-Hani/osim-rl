import gym
import numpy as np

class OsimWrapper(gym.Wrapper):
    def __init__(self, env, target_muscle_data):
        super(OsimWrapper, self).__init__(env)
        self.env = env
        self.target_data = target_muscle_data  # This should be your reference data
        self.current_step = 0
        
        # Calculate new dimension: 412 + however many muscles you are tracking
        raw_obs = self.env.reset()
        self.output_dim = len(self.flatten_obs(raw_obs)) + len(self.target_data[0])
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.output_dim,), dtype=np.float32
        )

    def get_combined_obs(self, raw_obs):
        # 1. Flatten the current physical state
        flat_obs = self.flatten_obs(raw_obs)
        
        # 2. Get the target muscle data for the current time step
        # We use modulo to loop the data if the simulation is longer than the file
        target = self.target_data[self.current_step % len(self.target_data)]
        
        # 3. Combine them
        return np.concatenate([flat_obs, target]).astype(np.float32)

    def reset(self, **kwargs):
        self.current_step = 0
        result = self.env.reset()
        
        # Handle tuple vs single return
        raw_obs = result[0] if isinstance(result, tuple) else result
        info = result[1] if isinstance(result, tuple) else {}
        
        return self.get_combined_obs(raw_obs), info

    def step(self, action):
        self.current_step += 1
        raw_obs, reward, done, info = self.env.step(action)
        
        combined_obs = self.get_combined_obs(raw_obs)
        
        # Optional: Enhance the reward if the action matches the target!
        # target = self.target_data[(self.current_step-1) % len(self.target_data)]
        # reward -= np.mean(np.square(action - target)) # Penalize deviation
        
        return combined_obs, reward, done, False, info