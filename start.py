import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# 1. SETUP PATHS
base_dir = r"D:\Projects\osim-rl"
repo_root = os.path.join(base_dir, "parent", "neurips18")
sys.path.insert(0, repo_root)

from osim.env import ProstheticsEnv
from parent.neurips18.envs.prosthetics_preprocess import preprocess_obs_round2 as preprocess

# 2. THE START MODEL ARCHITECTURE (FC-0990)
class FCActor(nn.Module):
    def __init__(self):
        super().__init__()
        # Mismatch fix: layer_0 expects 344, not 1376
        self.feature_net = nn.ModuleDict({
            "net": nn.ModuleDict({
                "layer_0": nn.Linear(344, 512, bias=False),
                "norm_0": nn.LayerNorm(512),
                "layer_1": nn.Linear(512, 512, bias=False),
                "norm_1": nn.LayerNorm(512),
                "layer_2": nn.Linear(512, 256, bias=False),
                "norm_2": nn.LayerNorm(256),
            })
        })
        
        self.policy_net = nn.ModuleDict({
            "net": nn.ModuleDict({
                "layer_0": nn.Linear(256, 19)
            })
        })

    def forward(self, x):
        # x shape: (batch, 4, 344)
        # Based on the 344-input weight, we use the most recent frame (index -1)
        x = x[:, -1, :] 
        
        # Pass through the feature net layers
        for i in range(3):
            x = self.feature_net["net"][f"layer_{i}"](x)
            x = self.feature_net["net"][f"norm_{i}"](x)
            x = torch.relu(x)
            
        action = torch.tanh(self.policy_net["net"]["layer_0"](x))
        return action

# 3. LOAD THE START MODEL
def load_start_model():
    model = FCActor()
    # Point to the 'fc-0990' checkpoint
    ckpt_path = os.path.join(repo_root, "submit", "checkpoints", "start", "181029-l2r-ecritic-qua-fc-0990", "checkpoint.6000.pth.tar")
    
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Use the key 'actor_state_dict' as identified in your previous debug logs
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    return model

def main():
    model = load_start_model()
    print("SUCCESS: Start (FC) weights loaded with perfect 344-input match!")
    
    env = ProstheticsEnv(visualize=True)
    env.change_model(model="3D", prosthetic=True, difficulty=1)
    
    history = deque(maxlen=4)
    
    for ep in range(3):
        state_desc = env.reset(project=False)
        done = False
        s = preprocess(state_desc)
        s.append(-1.0) # Time feature
        for _ in range(4): history.append(s)
            
        step = 0
        while not done:
            obs_tensor = torch.FloatTensor(np.array(history)).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_tensor).numpy()[0]
            
            # Use the standard scaling from the challenge
            final_action = action * 0.5 + 0.5
            state_desc, reward, done, _ = env.step(final_action, project=False)
            
            s = preprocess(state_desc)
            time_feature = (step / 1000.0 - 0.5) * 2.0
            s.append(time_feature)
            history.append(s)
            
            step += 1
            if step > 1000: break
        print(f"Episode {ep+1} Finished.")

if __name__ == "__main__":
    main()