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
# | 2/28/2018          | Henry Savage  | Remove a few irrelevant lines of   | #
# |                    |               | code and added comments            | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Changed twist cmd update interface | #
# |                    |               | to "set_target_*" for clarity      | #
# +--------------------+---------------+------------------------------------+ #
# | 3/29/2018          | Xiao He       | Updated the velocity_controller    | #
# +--------------------+---------------+------------------------------------+ #
# | 4/12/2018          | Henry Savage  | Reverted some changes to carry max | #
# |                    |               | accel values for thresholding in   | #
# |                    |               | the velocity controller            | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# Debug prints - to be removed
import rospy

# For steering control
from yaw_controller import YawController

# For throttle/brake control
from velocity_controller import VelocityController

class Controller(object):
    def __init__(self, wheel_base=0.0, steer_ratio=0.0, min_speed=0.0,
                 max_lat_accel=0.0, max_steer_angle=0.0, vehicle_mass=1e-6,
                 max_accel=0.0, max_decel=0.0, max_input_accel=0.0,
                 max_input_decel=0.0, deadband=0.0, fuel_capacity=0.0,
                 wheel_radius=0.0):
        '''
        Initializes the controller object
        '''

        # Steering controller
        self.steering_controller = YawController(wheel_base=wheel_base,
                                                 steer_ratio=steer_ratio,
                                                 min_speed=min_speed,
                                                 max_lat_accel=max_lat_accel,
                                                 max_steer_angle=max_steer_angle)

        # Throttle/Brake Controller
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
        Sets the current linear velocity of the vehicle for the controller
        to use

        Returns:
            float: vel - the current linear velocity (m/s)

        Complexity: O(1)
        '''
        self.cur_linear_velocity = vel

    def set_current_angular_velocity(self, vel=0):
        '''
        Sets the current angular velocity of the vehicle for the controller
        to use

        Returns:
            float: vel - the current angular velocity (m/s)

        Complexity: O(1)
        '''
        self.cur_angular_velocity = vel

    def set_target_linear_velocity(self, vel=0):
        '''
        Sets the target linear velocity of the vehicle for the controller
        to use

        Returns:
            float: vel - the target linear velocity (m/s)

        Complexity: O(1)
        '''
        self.target_linear_velocity = vel

    def set_target_angular_velocity(self, vel=0):
        '''
        Sets the target angular velocity of the vehicle for the controller
        to use

        Returns:
            float: vel - the target angular velocity (m/s)

        Complexity: O(1)
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
        throttle, brake = self.throttle_controller.get_throttle_brake(
                                                      self.target_linear_velocity,
                                                      self.target_angular_velocity,
                                                      self.cur_linear_velocity
                                                   )

        # Hand back values
        return throttle, brake, steer
