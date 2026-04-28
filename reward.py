import numpy as np
import pandas as pd

def load_mot_file(path):
    df = pd.read_csv(path, sep='\t', skiprows=6)
    print("motion file is loaded")


    joints = [
        "hip_flexion_l",
        "knee_angle_l",
        "ankle_angle_l",
        "hip_flexion_r",
        "knee_angle_r"
    ]

    theta = df[joints].values

    # convert to radians
    theta = np.deg2rad(theta)

    return theta
#--------------------------------------------------------

from scipy.interpolate import interp1d

def build_interpolator(theta):
    t = np.linspace(0, 1, len(theta))
    return interp1d(t, theta, axis=0, kind='cubic')

#--------------------------------------------------------

from osim.env import ProstheticsEnv
class ProstheticsImitationEnv(ProstheticsEnv):

    def __init__(self, theta_target, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolator = build_interpolator(theta_target)
        self.sim_steps = self.time_limit

        self.current_step = 0

        # reward weights
        self.w_track = 1.0
        self.w_effort = 0.001
        self.w_vel = 1.0
        self.w_balance = 5.0

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.current_step = 0
        return self.get_observation()

    def get_joint_angles(self):
        state = self.get_state_desc()

        theta = [
            state["joint_pos"]["hip_l"][0],
            state["joint_pos"]["knee_l"][0],
            state["joint_pos"]["ankle_l"][0],
            state["joint_pos"]["hip_r"][0],
            state["joint_pos"]["knee_r"][0],
        ]

        return np.array(theta)

    def get_muscle_effort(self):
        return np.sum(np.array(self.osim_model.get_activations())**2)

    def get_reward(self):
        # phase
        phase = self.current_step / self.sim_steps
        theta_ref = self.interpolator(phase)

        theta_current = self.get_joint_angles()

        # tracking
        tracking = np.mean((theta_current - theta_ref)**2)

        # effort
        effort = self.get_muscle_effort()

        # velocity
        state = self.get_state_desc()
        vx = state["body_vel"]["pelvis"][0]

        # balance
        pelvis_height = state["body_pos"]["pelvis"][1]

        reward = (
            - self.w_track * tracking
            - self.w_effort * effort
            + self.w_vel * vx
            - self.w_balance * (pelvis_height - 1.0)**2
        )

        if self.is_done():
            reward -= 100

        self.current_step += 1

        return reward
    
    def step(self, action):
        _, _, done, info = super().step(action)

        obs = self.get_observation()  
        reward = self.get_reward()

        return obs, reward, done, info