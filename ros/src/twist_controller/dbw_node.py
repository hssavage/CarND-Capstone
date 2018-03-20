#!/usr/bin/env python

'''
###############################################################################
# dbw_node.py                                                                 #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the Drive by Wire Node as part of the   #
# Self-Driving Car System. This node is responsible for controlling the car's #
# throttle, brakes and steering relative to the final waypoints that the      #
# waypoint_updater node is putting out.                                       #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Inputs                                                                  | #
# +----------------------+----------------+---------------------------------+ #
# | Topic Path           | Source         | Description                     | #
# +----------------------+----------------+---------------------------------+ #
# | /current_velocity    | Simulator /    | Contains the velocity info of   | #
# |                      | Localization   | the vehicle in the sim          | #
# +----------------------+----------------+---------------------------------+ #
# | /vehicle/dbw_enabled | Simulator /    | Contains the drive by wire      | #
# |                      | Localization   | status of the vehicle           | #
# +----------------------+----------------+---------------------------------+ #
# | /twist_cmd           | Waypoint       | Contains the position info of   | #
# |                      | Follower       | the vehicle in the sim          | #
# +----------------------+----------------+---------------------------------+ #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Outputs                                                                 | #
# +-----------------------+---------------+---------------------------------+ #
# | Topic Path            | Update Rate + | Description                     | #
# |                       | Queue Size    |                                 | #
# +-----------------------+---------------+---------------------------------+ #
# | /vehicle/brake_cmd    | 50hz / ? item | The command to the vehicles     | #
# |                       |               | brake controller to carry out   | #
# +-----------------------+---------------+---------------------------------+ #
# | /vehicle/throttle_cmd | 50hz / ? item | The command to the vehicles     | #
# |                       |               | throttle controller to carry    | #
# |                       |               | out                             | #
# +-----------------------+---------------+---------------------------------+ #
# | /vehicle/steering_cmd | 50hz / ? item | The command to the vehicles     | #
# |                       |               | steering controller to carry    | #
# |                       |               | out                             | #
# +-----------------------+---------------+---------------------------------+ #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 2/24/2018          | Henry Savage  | Initial pass on the code based on  | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Fixed the sign on max deceleration | #
# |                    |               | and updated target velocity        | #
# |                    |               | interface.                         | #
# +--------------------+---------------+------------------------------------+ #
# | 3/19/2018          | Henry Savage  | Added comments on presumed units   | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        '''
        '''

        # Initialize and register our node
        rospy.init_node('dbw_node')

        # Grab our constant parameters
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35) # Kg, probably
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5) # Gallons? Litters?
        brake_deadband = rospy.get_param('~brake_deadband', .1) # Torque value
        decel_input_limit = rospy.get_param('~decel_limit', -5) # Torque value
        accel_input_limit = rospy.get_param('~accel_limit', 1.) # Percent engaged value
        accel_limit = 10.0
        decel_limit = -10.0
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        min_speed = 0.0

        # Register our subscribes
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Register our publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # Is this node enabled?
        self.enabled = False

        # Create the Controller object
        self.controller = Controller(
            wheel_base=wheel_base, steer_ratio=steer_ratio,
            min_speed=min_speed, max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle, vehicle_mass=vehicle_mass,
            max_accel=accel_limit, max_decel=decel_limit,
            max_input_accel=accel_input_limit, max_input_decel=decel_input_limit,
            deadband=brake_deadband, fuel_capacity=fuel_capacity,
            wheel_radius=wheel_radius
        )

        # Keep track of header meta data
        self.frame_id = None
        self.seq = 0

        # Run the control loop
        self.run()

    def twist_cmd_cb(self, cmd):
        '''
        Reports the status of the vehicle's velocity.

        /twist_cmd returns the follow values:

        cmd.header.seq
        cmd.header.stamp.secs
        cmd.header.stamp.n_secs
        cmd.header.frame_id
        cmd.twist.linear.x  -- velocity in forward direction of the car
        cmd.twist.linear.y  -- ?
        cmd.twist.linear.z  -- probably zero?
        cmd.twist.angular.x -- roll
        cmd.twist.angular.y -- pitch
        cmd.twist.angular.z -- yaw
        '''
        # rospy.loginfo("Command -- lin: " + str(cmd.twist.linear.x) + ", ang: " + str(cmd.twist.angular.z))
        self.controller.set_target_linear_velocity(vel=cmd.twist.linear.x)
        self.controller.set_target_angular_velocity(vel=cmd.twist.angular.z)

    def current_velocity_cb(self, vel):
        '''
        Reports the status of the vehicle's velocity.

        /current_velocity returns the follow values:

        vel.header.seq
        vel.header.stamp.secs
        vel.header.stamp.n_secs
        vel.header.frame_id
        vel.twist.linear.x
        vel.twist.linear.y
        vel.twist.linear.z
        vel.twist.angular.x
        vel.twist.angular.y
        vel.twist.angular.z
        '''
        self.controller.set_current_linear_velocity(vel.twist.linear.x)
        self.controller.set_current_angular_velocity(vel.twist.angular.z)

    def dbw_enabled_cb(self, enabled):
        '''
        Reports the status of the drive by wire system.

        /vehicle/dbw_enabled returns the following values:

        enabled
        '''
        self.enabled = enabled

    def publish(self, throttle, brake, steer):
        '''
        Publishes the desired throttle, brake and steering values as commands
        '''

        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

    def run(self):
        '''
        '''

        # Set rate of loop
        rate = rospy.Rate(50) # 50Hz

        while not rospy.is_shutdown():
            if(self.enabled):
                throttle, brake, steer = self.controller.control()
                self.publish(throttle, brake, steer)
            rate.sleep()

if __name__ == '__main__':
    try:
        DBWNode()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start drive-by-wire node.')
