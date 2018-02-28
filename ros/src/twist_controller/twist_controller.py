'''
###############################################################################
# twist_controller.py                                                         #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the command controller for the throttle #
# brakes and steering for the Self-Driving Car System.                        #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 2/24/2018          | Henry Savage  | Initial pass on the code           | #
# +--------------------+---------------+------------------------------------+ #
# | 2/27/2018          | Henry Savage  | Integrated a velocity controller   | #
# |                    |               | that works better than a PID       | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# For steering control
from yaw_controller import YawController

# For throttle/brake control
from velocity_controller import VelocityController

# For timestamping - time()
# from time import time

class Controller(object):
    def __init__(self, wheel_base=0.0, steer_ratio=0.0, min_speed=0.0,
                 max_lat_accel=0.0, max_steer_angle=0.0, vehicle_mass=1e-6,
                 max_accel=0.0, max_decel=0.0, max_input_accel=0.0,
                 max_input_decel=0.0, deadband=0.0, fuel_capacity=0.0,
                 wheel_radius=0.0):
        '''
        '''

        # Steering controller
        self.steering_controller = YawController(wheel_base=wheel_base,
                                                 steer_ratio=steer_ratio,
                                                 min_speed=min_speed,
                                                 max_lat_accel=max_lat_accel,
                                                 max_steer_angle=max_steer_angle)

        # Throttle/Brake Controller
        # PID(kp=0.2, ki=0.0004, kd=5.0, mn=-1.0, mx=1.0)
        self.throttle_controller = VelocityController(
                                        vehicle_mass=vehicle_mass,
                                        max_accel=max_accel,
                                        max_decel=max_decel,
                                        max_input_accel=max_input_accel,
                                        max_input_decel=max_input_decel,
                                        wheel_radius=wheel_radius,
                                        deadband=deadband,
                                        fuel_capacity=fuel_capacity)

        # Vehicle Status variables
        self.cur_linear_velocity = 0
        self.cur_angular_velocity = 0

        # Desired state variables
        self.target_linear_velocity = 0
        self.target_angular_velocity = 0

    def set_current_linear_velocity(self, vel=0):
        '''
        '''
        self.cur_linear_velocity = vel

    def set_current_angular_velocity(self, vel=0):
        '''
        '''
        self.cur_angular_velocity = vel

    def set_linear_velocity_cmd(self, vel=0):
        '''
        '''
        self.target_linear_velocity = vel

    def set_angular_velocity_cmd(self, vel=0):
        '''
        '''
        self.target_angular_velocity = vel

    def control(self):
        '''
        Returns a list of the desired throttle, brake and steering values

        Returns:
            list<float>: [throttle, brake, steering]

        Complexity: O(1)
        '''

        # Values to return
        throttle = 0.0
        brake = 0.0
        steer = 0.0

        # Run steering controller
        steer = self.steering_controller.get_steering(
                                            self.target_linear_velocity,
                                            self.target_angular_velocity,
                                            self.cur_linear_velocity
                                         )

        # Run throttle controller
        t_err = self.target_linear_velocity - self.cur_linear_velocity
        throttle, brake = self.throttle_controller.get_throttle_brake(
                                                      self.target_linear_velocity,
                                                      self.target_angular_velocity,
                                                      self.cur_linear_velocity
                                                   )

        # Hand back values
        return throttle, brake, steer
