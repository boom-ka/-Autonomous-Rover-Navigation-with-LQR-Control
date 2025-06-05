import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# LQR gains adjusted for rover stability
LQR_K = [-1.5, -0.025, 5.9748026764525894e-18, 1.5]
WHEEL_RADIUS = 0.04
MAX_MOTOR_VEL = 500.0 # rad/s


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class RobotLqr:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

    def set_velocity_linear_set_point(self, vel: float) -> None:
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw: float) -> None:
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        angles = rotation.as_euler('xyz', degrees=False)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        return angular[0]

    def get_wheel_velocity(self) -> float:
        vel_fl = self.data.joint('fl_wheel_joint').qvel[0]
        vel_fr = self.data.joint('fr_wheel_joint').qvel[0]
        vel_rl = self.data.joint('rl_wheel_joint').qvel[0]
        vel_rr = self.data.joint('rr_wheel_joint').qvel[0]

        # Average of all wheels
        return (vel_fl + vel_fr + vel_rl + vel_rr) / 4.0

    def calculate_lqr_velocity(self) -> float:
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        # Apply filters for smoother control
        self.pitch_dot_filtered = (self.pitch_dot_filtered * .975) + (pitch_dot * .025)
        self.velocity_angular_filtered = (self.velocity_angular_filtered * .975) + (self.get_wheel_velocity() * .025)

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS

        lqr_v = LQR_K[0] * (0 - pitch) + LQR_K[1] * self.pitch_dot_filtered + LQR_K[2] * 0 + LQR_K[3] * velocity_linear_error 
        return lqr_v / WHEEL_RADIUS
    
    def update_motor_speed(self) -> None:
        vel = self.calculate_lqr_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # Enhanced steering factor that increases at lower speeds
        base_steering_factor = 1.5  # Increased from original values
        min_speed_for_full_steering = 0.5  # m/s
        
        # Calculate dynamic steering intensity - stronger at lower speeds
        steering_intensity = min(1.0, abs(self.velocity_linear_set_point) / min_speed_for_full_steering)
        # Inverse relationship - stronger steering at lower speeds
        effective_steering_factor = base_steering_factor * (2.0 - steering_intensity)
        
        # Calculate wheel speed differentials based on yaw
        yaw_abs = abs(self.yaw)
        steering_diff_front = vel * effective_steering_factor * yaw_abs
        steering_diff_rear = vel * (effective_steering_factor * 0.6) * yaw_abs  # Rear wheels turn less
        
        # Minimum effect to ensure turning happens even at very low speeds
        min_steering_effect = 10.0  # Minimum differential for wheels when steering
        if yaw_abs > 0.1 and abs(vel) > 0:
            steering_diff_front = max(steering_diff_front, min_steering_effect)
            steering_diff_rear = max(steering_diff_rear, min_steering_effect * 0.6)
        
        # Apply differential steering based on direction
        if self.yaw > 0:  # turning right
            fl_vel = vel + steering_diff_front  # outside wheel goes faster
            fr_vel = vel - steering_diff_front  # inside wheel goes slower
            rl_vel = vel + steering_diff_rear   # less effect on rear wheels
            rr_vel = vel - steering_diff_rear   # less effect on rear wheels
        elif self.yaw < 0:  # turning left
            fl_vel = vel - steering_diff_front  # inside wheel goes slower
            fr_vel = vel + steering_diff_front  # outside wheel goes faster
            rl_vel = vel - steering_diff_rear   # less effect on rear wheels
            rr_vel = vel + steering_diff_rear   # less effect on rear wheels
        else:  # going straight
            fl_vel = vel
            fr_vel = vel
            rl_vel = vel
            rr_vel = vel

        # Optional: For very sharp turns, we can even apply opposite rotation to inner wheels
        sharp_turn_threshold = 0.8
        if yaw_abs > sharp_turn_threshold and abs(vel) < 1.0:
            # Make inner wheels rotate in opposite direction for sharp turns at low speed
            if self.yaw > 0:  # sharp right turn
                fr_vel = -abs(fr_vel) * 0.3  # inner front wheel reverses slightly
                rr_vel = -abs(rr_vel) * 0.1  # inner rear wheel reverses slightly
            else:  # sharp left turn
                fl_vel = -abs(fl_vel) * 0.3  # inner front wheel reverses slightly
                rl_vel = -abs(rl_vel) * 0.1  # inner rear wheel reverses slightly

        # Clamp individual wheel velocities
        fl_vel = clamp(fl_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        fr_vel = clamp(fr_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        rl_vel = clamp(rl_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
        rr_vel = clamp(rr_vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        # Set motor control values
        self.data.actuator('motor_fl_wheel').ctrl = [fl_vel]
        self.data.actuator('motor_fr_wheel').ctrl = [fr_vel]
        self.data.actuator('motor_rl_wheel').ctrl = [rl_vel]
        self.data.actuator('motor_rr_wheel').ctrl = [rr_vel]

    def reset(self):
        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

        # Set position slightly above ground to account for suspension
        self.data.qpos[2] = 0.08

        # Smaller random initial orientation for stability
        x_rot = (np.random.random() - 0.5) * 0.05
        y_rot = (np.random.random() - 0.5) * 0.05
        z_rot = (np.random.random() - 0.5) * 0.05
        euler_angles = [x_rot, y_rot, z_rot]
        rotation = Rotation.from_euler('xyz', euler_angles)
        self.data.qpos[3:7] = rotation.as_quat()

        # Reset all motors
        self.data.actuator('motor_fl_wheel').ctrl = [0]
        self.data.actuator('motor_fr_wheel').ctrl = [0]
        self.data.actuator('motor_rl_wheel').ctrl = [0]
        self.data.actuator('motor_rr_wheel').ctrl = [0]