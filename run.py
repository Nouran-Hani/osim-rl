import os
import sys
from osim import env
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# 1. SETUP PATHS
base_dir = r"D:\Projects\osim-rl\parent"
repo_root = os.path.join(base_dir, "neurips18")
sys.path.insert(0, repo_root)

from osim.env import ProstheticsEnv
from parent.neurips18.envs.prosthetics_preprocess import preprocess_obs_round2 as preprocess

# 2. UPDATED ARCHITECTURE TO MATCH CHECKPOINT DIMENSIONS
class SequentialNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleDict()
        for i, (in_f, out_f) in enumerate(layers):
            self.net[f"layer_{i}"] = nn.Linear(in_f, out_f, bias=False)
            self.net[f"norm_{i}"] = nn.LayerNorm(out_f)
            
    def forward(self, x, layer_idx):
        x = self.net[f"layer_{layer_idx}"](x)
        x = self.net[f"norm_{layer_idx}"](x)
        return torch.relu(x)

class LamaActor(nn.Module):
    def __init__(self):
        super().__init__()
        # feature_net sequence: 344 -> 512 -> 512 -> 256 -> 256
        self.feature_net = SequentialNet([(344, 512), (512, 512), (512, 256), (256, 256)])
        
        # Attention pooling expects a 3D weight [1, 256, 1]
        self.attn = nn.Sequential(nn.Conv1d(256, 1, kernel_size=1))
        
        # feature_net2 handles the 1024-sized concatenated LAMA vector
        self.feature_net2 = SequentialNet([(1024, 256)])
        
        self.policy_net = nn.ModuleDict({
            "net": nn.ModuleDict({
                "layer_0": nn.Linear(256, 19)
            })
        })

    def forward(self, x):
        # x shape: (batch, 4, 344)
        batch_size = x.size(0)
        
        # 1. CORRECTED FEATURE EXTRACTION
        h_list = []
        for i in range(4): # Loop over the 4 frames in history
            h_step = x[:, i, :] # Get frame i (size 344)
            
            # Each individual frame MUST go through all 4 layers of feature_net
            for j in range(4): 
                h_step = self.feature_net(h_step, j)
            
            h_list.append(h_step.unsqueeze(1))
        
        h = torch.cat(h_list, dim=1) # Now h is (batch, 4, 256)
        
        # 2. LAMA POOLING
        h_last = h[:, -1, :] # The most recent observation
        h_avg = torch.mean(h, dim=1) # Average pool
        h_max, _ = torch.max(h, dim=1) # Max pool
        
        # Attention pool: Needs (batch, channels, length) for Conv1d
        h_for_attn = h.transpose(1, 2) # (batch, 256, 4)
        attn_weights = torch.softmax(self.attn(h_for_attn), dim=2) # (batch, 1, 4)
        h_attn = torch.sum(h * attn_weights.transpose(1, 2), dim=1)
        
        # Concatenate: [last, avg, max, attn] -> Total size 1024
        h_lama = torch.cat([h_last, h_avg, h_max, h_attn], dim=-1)
        
        # 3. FINAL LAYERS
        h_final = self.feature_net2(h_lama, 0)
        action = torch.tanh(self.policy_net["net"]["layer_0"](h_final))
        return action

# 3. LOADING THE WEIGHTS
def load_model():
    model = LamaActor() # For side walk
    ckpt_path = os.path.join(repo_root, "submit", "checkpoints", "run", "181018-l2r-ecritic-qua-lama04", "checkpoint.7300.pth.tar")
    # ckpt_path = os.path.join(repo_root, "submit", "checkpoints", "side", "lama12-side", "checkpoint.2850.pth.tar")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint['actor_state_dict'])
    model.eval()
    return model

# 4. MAIN SIMULATION LOOP
def main():
    model = load_model()
    print("Model loaded successfully. Starting environment...")
    env = ProstheticsEnv(visualize=True, difficulty=1)
    # Target is usually [velocity_x, velocity_z]
    history = deque(maxlen=4)
    
    print("SUCCESS: Weights loaded with perfect size match!")

    for ep in range(3):
        # env.request_velocity(1.5, 0) # Attempt to set 1.5 m/s forward

        state_desc = env.reset(project=False)
        done = False

        # try:
        #     s = preprocess(state_desc)
        # except KeyError as e:
        #     print(f"KeyError: {e}. Check if the environment forces match.")
        #     # Debug: Print available force keys if it fails again
        #     print(f"Available forces: {state_desc['forces'].keys()}")
        #     return

        s = preprocess(state_desc)
        s.append(-1.0)
        for _ in range(4): history.append(s)
            
        step = 0
        while not done:
            obs_tensor = torch.FloatTensor(np.array(history)).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_tensor).numpy()[0]
            
            # Post-process: Scitator's team uses a specific scaling
            final_action = action * 0.5 + 0.5
            state_desc, reward, done, _ = env.step(final_action, project=False)
            print(f"Step: {step}, Reward: {reward:.4f}, Done: {done}")
            # print(f"Activations: {state_desc['muscles']}")
            s = preprocess(state_desc)
            # Time trick: moves from -1.0 to 1.0 over 1000 steps
            time_feature = (step / 1000.0 - 0.5) * 2.0
            s.append(time_feature)
            history.append(s)
            step += 1
            if step > 1000: break
        print(f"Episode {ep+1} Finished.")

if __name__ == "__main__":
    main()