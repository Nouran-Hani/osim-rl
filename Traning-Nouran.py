from stable_baselines3 import PPO
from osim.env import ProstheticsEnv
from InitNouran import OsimWrapper

MOT_FILE = r"D:\Projects\osim-rl\Data\normal.mot"   # ← path to your gait reference file

# ── 1. Raw OpenSim environment ────────────────────────────────────────────────
print("1. Initialising environment...")
raw_env = ProstheticsEnv(visualize=False)
# raw_env.change_model(model="3D", prosthetic=True)  # uncomment for prosthetic model

# ── 2. Wrap with joint-angle imitation wrapper ────────────────────────────────
print("2. Wrapping environment with gait reference...")
wrapped_env = OsimWrapper(
    env=raw_env,
    mot_filepath=MOT_FILE,
    imitation_weight=0.1,       # weight on the joint-angle imitation penalty
    obs_joint_indices=None,     # set once you know which flat-obs indices map to joints
)

# ── 3. PPO setup ──────────────────────────────────────────────────────────────
print("3. Setting up PPO...")
model = PPO(
    "MlpPolicy",
    wrapped_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    ent_coef=0.01,
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("4. Starting training...")
model.learn(total_timesteps=500000)

model_name = "ppo_gait_agent"
model.save(model_name)
print(f"Model saved as '{model_name}'")

# ── 5. (Optional) evaluate ───────────────────────────────────────────────────
# print("5. Evaluating agent...")
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=5)
# print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")