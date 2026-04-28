import gym
import numpy as np
import pandas as pd
import os


def load_mot_file(filepath):
    """
    Parse an OpenSim .mot file and return:
      - joint_angles: np.ndarray of shape (n_frames, n_joints), values in RADIANS
      - column_names: list of joint angle column names (excludes 'time')
    """
    # Find the line after 'endheader'
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end = next(i for i, l in enumerate(lines) if l.strip() == "endheader")
    data_lines = lines[header_end + 1 :]  # first data line is the column-name row

    # Parse with pandas (tab-separated)
    from io import StringIO
    df = pd.read_csv(StringIO("".join(data_lines)), sep="\t")

    # Drop the time column – we only want joint angle values
    column_names = [c for c in df.columns if c != "time"]
    joint_angles_deg = df[column_names].values.astype(np.float32)

    # Convert degrees → radians (standard for physics / RL environments)
    joint_angles_rad = np.deg2rad(joint_angles_deg)

    return joint_angles_rad, column_names


class OsimWrapper(gym.Wrapper):
    """
    Gym wrapper that appends reference joint-angle targets (from a .mot file)
    to every observation so the policy can imitate normal gait.

    The wrapper also adds an imitation reward term that penalises the agent
    for deviating from the reference joint angles when those angles are
    observable in the environment's own state dict.

    Args:
        env               : raw OsimEnv (or any gym-compatible env)
        mot_filepath      : path to the OpenSim .mot file
        imitation_weight  : scalar weight on the imitation penalty (set 0 to disable)
        obs_joint_indices : optional list of indices inside the *flat* observation
                            that correspond to the same joints as in the .mot file.
                            If provided, the imitation reward is computed from those.
    """

    # Joints that appear in ProstheticsEnv's state dict (body_pos / joint_pos).
    # Adjust this mapping if your env exposes joints differently.
    MOT_JOINT_NAMES = [
        "pelvis_list", "pelvis_rotation", "pelvis_tilt",
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "knee_angle_r", "ankle_angle_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
        "knee_angle_l", "ankle_angle_l",
        "pelvis_ty",
        "lumbar_bending", "lumbar_rotation", "lumbar_extension",
    ]

    def __init__(self, env, mot_filepath, imitation_weight=0.1, obs_joint_indices=None):
        super(OsimWrapper, self).__init__(env)
        self.env = env
        self.imitation_weight = imitation_weight
        self.obs_joint_indices = obs_joint_indices
        self.current_step = 0

        # ── Load reference gait data ──────────────────────────────────────────
        if not os.path.exists(mot_filepath):
            raise FileNotFoundError(f".mot file not found: {mot_filepath}")

        self.reference_angles, self.joint_names = load_mot_file(mot_filepath)
        self.n_frames, self.n_joints = self.reference_angles.shape
        print(f"[OsimWrapper] Loaded {self.n_frames} frames × {self.n_joints} joints "
              f"from '{mot_filepath}'")
        print(f"[OsimWrapper] Joint columns: {self.joint_names}")

        # ── Redefine observation space ────────────────────────────────────────
        raw_obs = self.env.reset()
        raw_obs = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs

        # Debug: show top-level obs structure so empty sub-dicts are visible
        if isinstance(raw_obs, dict):
            print("[OsimWrapper] Obs keys:", list(raw_obs.keys()))
            for k, v in raw_obs.items():
                if isinstance(v, dict):
                    print(f"  '{k}' -> dict keys: {list(v.keys())}, "
                          f"non-empty: {[kk for kk, vv in v.items() if len(vv) > 0] if v else '(empty)'}")
                elif isinstance(v, (list, tuple)):
                    print(f"  '{k}' -> list/tuple len={len(v)}")
                else:
                    print(f"  '{k}' -> {type(v).__name__} shape={np.array(v).shape}")

        flat_obs = self.flatten_obs(raw_obs)

        # New obs = original flat obs + reference joint angles for this timestep
        self.output_dim = len(flat_obs) + self.n_joints
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.output_dim,), dtype=np.float32
        )
        print(f"[OsimWrapper] Observation space: {len(flat_obs)} (env) "
              f"+ {self.n_joints} (reference) = {self.output_dim}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def flatten_obs(self, obs):
        """Recursively flatten a dict/list/ndarray observation into a 1-D array.
        Empty containers are skipped so np.concatenate never receives an empty list.
        """
        if isinstance(obs, dict):
            parts = [self.flatten_obs(v) for v in obs.values()]
            parts = [p for p in parts if p.size > 0]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        elif isinstance(obs, (list, tuple)):
            parts = [self.flatten_obs(v) for v in obs]
            parts = [p for p in parts if p.size > 0]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        else:
            arr = np.atleast_1d(np.array(obs, dtype=np.float32)).flatten()
            return arr

    def _reference_frame(self):
        """Return the reference joint angles for the current timestep (looped)."""
        return self.reference_angles[self.current_step % self.n_frames]

    def _build_obs(self, raw_obs):
        flat_obs = self.flatten_obs(raw_obs)
        target = self._reference_frame()
        return np.concatenate([flat_obs, target]).astype(np.float32)

    def _imitation_penalty(self, raw_obs):
        """
        Compute an imitation penalty.  If obs_joint_indices was supplied we
        compare the env's own joint values against the reference; otherwise
        we return 0 (you can extend this once you know the index mapping).
        """
        if self.imitation_weight == 0 or self.obs_joint_indices is None:
            return 0.0

        flat_obs = self.flatten_obs(raw_obs)
        current_angles = flat_obs[self.obs_joint_indices]
        target_angles = self._reference_frame()

        # MSE penalty (already in radians on both sides)
        penalty = float(np.mean(np.square(current_angles - target_angles)))
        return penalty

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self.current_step = 0
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        result = self.env.reset(**kwargs)

        raw_obs = result[0] if isinstance(result, tuple) else result
        info    = result[1] if isinstance(result, tuple) else {}

        return self._build_obs(raw_obs), info

    def step(self, action):
        result = self.env.step(action)

        # Support both 4-tuple (old gym) and 5-tuple (new gym) step returns
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
        else:
            raw_obs, reward, done, info = result
            terminated, truncated = done, False

        self.current_step += 1

        combined_obs = self._build_obs(raw_obs)

        # ── Imitation reward shaping ──────────────────────────────────────────
        penalty = self._imitation_penalty(raw_obs)
        shaped_reward = reward - self.imitation_weight * penalty

        return combined_obs, shaped_reward, terminated, truncated, info