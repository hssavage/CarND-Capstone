#!/usr/bin/env python

'''
###############################################################################
# waypoint_updater.py                                                         #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the Waypoint Updater Node as part of    #
# the Self-Driving Car System. This node is to update the target velocity of  #
# each waypoint based on the traffic light and obstacle detection data. This  #
# node will publish to '/final_waypoints' a fixed number of waypoints         #
# currently ahead of the vehicle.                                             #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Inputs                                                                  | #
# +--------------------+--------------------+-------------------------------+ #
# | Topic Path         | Source             | Description                   | #
# +--------------------+--------------------+-------------------------------+ #
# | /current_pose      | Simulator /        | Contains the position info of | #
# |                    | Localization       | the vehicle in the sim        | #
# +--------------------+--------------------+-------------------------------+ #
# | /base_waypoints    | Waypoint Loader    | The list of waypoints (Note   | #
# |                    |                    | these are sent once as a CSV) | #
# +--------------------+--------------------+-------------------------------+ #
# | /obstacle_waypoint | Obstacle Detection | A waypoint of an obstacle     | #
# +--------------------+--------------------+-------------------------------+ #
# | /traffic_waypoint  | Traffic Light      | A waypoint of a known traffic | #
# |                    | Detection          | light and its status          | #
# +--------------------+--------------------+-------------------------------+ #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Outputs                                                                 | #
# +--------------------+---------------+------------------------------------+ #
# | Topic Path         | Update Rate + | Description                        | #
# |                    | Queue Size    |                                    | #
# +--------------------+---------------+------------------------------------+ #
# | /final_waypoints   | ??hz / ? item | The updated subset of waypoints    | #
# |                    |               | with velocities adjusted for       | #
# |                    |               | traffic lights and obstacles       | #
# +--------------------+---------------+------------------------------------+ #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 2/22/2018          | Henry Savage  | Initial pass on the code based on  | #
# |                    |               | pointers in online and diagrams    | #
# +--------------------+---------------+------------------------------------+ #
# | 3/09/2018          | Henry Savage  | Added traffic_waypoint handling.   | #
# |                    |               | Added max velocity watching.       | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Updated waypoint publish rate      | #
# |                    |               |                                    | #
# +--------------------+---------------+------------------------------------+ #
# | 3/15/2018          | Henry Savage  | Added debug printing, which can be | #
# |                    |               | enabled by adding the output attr  | #
# |                    |               | to the launch file                 | #
# +--------------------+---------------+------------------------------------+ #
# | 3/19/2018          | Henry Savage  | Updated the debug interface to use | #
# |                    |               | the new next_waypoint value. Also  | #
# |                    |               | changed debug interface to output  | #
# |                    |               | update timestamp and latencies     | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# ROS core stuff
import rospy

# Data types
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

# Our waypoint planner
from planning import PathPlanner, PathPlannerException

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

###############################################################################
# DEBUG FUNCTIONS                                                             #
###############################################################################

from tf.transformations import euler_from_quaternion

def get_euler(quat_orient):
    '''
    Returns the yaw, pitch and roll values from a quaternion orientation

    Returns:
        list<float>: [pitch, roll, yaw]

    Complexity: O(1)
    '''
    return euler_from_quaternion([quat_orient.x, quat_orient.y,
                                  quat_orient.z, quat_orient.w])

def red(msg):
    '''
    Format text as red
    '''
    return "\033[0;31m" + msg + "\033[0m"

def green(msg):
    '''
    Format text as green
    '''
    return "\033[0;32m" + msg + "\033[0m"

def purple(msg):
    '''
    Format text as purple
    '''
    return "\033[0;35m" + msg + "\033[0m"

def term_map(cur_wp, tl_wp, tl_status, wps, x_size=100, y_size=60):
    '''
    '''

    # Error check
    if(wps is None):
        return ""

    # Bounds of map
    X_MAX = 2400.0
    Y_MAX = 3200.0

    # Place map
    frame = [[" "]*x_size for i in range(y_size)]
    for wp in wps:
        wp_x = wp.pose.pose.position.x
        wp_y = wp.pose.pose.position.y
        m_x = int((wp_x / X_MAX) * x_size)
        m_y = int((wp_y / Y_MAX) * y_size)
        # print("\t- (" + str(m_x) + ", " + str(m_y) + ")")
        try:
            frame[y_size - m_y][m_x] = '.'
        except Exception:
            pass

    # Add next traffic light
    if(tl_wp in range(0, len(wps))):
        wp_x = wps[tl_wp].pose.pose.position.x
        wp_y = wps[tl_wp].pose.pose.position.y
        m_x = int((wp_x / X_MAX) * x_size)
        m_y = int((wp_y / Y_MAX) * y_size)
        if(tl_status == "red"):
            frame[y_size - m_y][m_x] = red('o')
        else:
            frame[y_size - m_y][m_x] = green('o')

    # Add current position
    if(cur_wp in range(0, len(wps))):
        wp_x = wps[cur_wp].pose.pose.position.x
        wp_y = wps[cur_wp].pose.pose.position.y
        m_x = int((wp_x / X_MAX) * x_size)
        m_y = int((wp_y / Y_MAX) * y_size)
        frame[y_size - m_y][m_x] = purple('x')

    # Remove blank lines and return
    frame_s = '\n'.join(["".join(s) if '.' in s else "" for s in frame]).strip("\n")
    return frame_s

last_pos_ts = None
last_tl_ts = None
last_wp_ts = None

def update_frame(planner, ts):
    '''
    Create a debug frame
    '''

    global last_pos_ts
    global last_tl_ts
    global last_wp_ts

    # Grab status all at once
    s_limit = planner.speed_limit
    cur_x = planner.vehicle_pose.position.x
    cur_y = planner.vehicle_pose.position.y
    cur_z = planner.vehicle_pose.position.z
    _, _, s_angle = get_euler(planner.vehicle_pose.orientation)
    cur_wp = planner.next_waypoint
    tl_wp = planner.traffic_light_ind
    wps = planner.waypoints
    total_wps = len(wps) if wps != None else 0

    # Times stamps
    pos_ts = planner.vehicle_pose_ts if planner.vehicle_pose_ts != None else 0.0
    tl_ts = planner.traffic_light_ind_ts if planner.traffic_light_ind_ts != None else 0.0
    wp_ts = planner.next_waypoint_ts if planner.next_waypoint_ts != None else 0.0

    # Edge case for last known times
    if(last_pos_ts is None):
        last_pos_ts = pos_ts
    if(last_wp_ts is None):
        last_wp_ts = wp_ts
    if(last_tl_ts is None):
        last_tl_ts = tl_ts

    # Determine light color string
    tl_status = "red"
    if(tl_wp < 0):
        tl_status = "green"
        tl_wp *= -1

    # Build frame
    frame = ""
    frame += "Timestamp: " + str(ts) + "\n"
    frame += "Current Position: (" + str(cur_x) + ", " + str(cur_y) + ", " + str(cur_z) + ")\n"
    frame += "(Updated: " + str(pos_ts) + ", Delta: " + str(pos_ts - last_pos_ts) + ")\n"
    frame += "Current Waypoint: " + str(cur_wp) + "/" + str(total_wps) + "\n"
    frame += "(Updated: " + str(wp_ts) + ", Delta: " + str(wp_ts - last_wp_ts) + ")\n"
    frame += "Next Traffic Light: " + str(tl_wp) + "\n"
    frame += "Next Traffic Status: " + str(tl_status) + "\n"
    frame += "(Updated: " + str(tl_ts) + ", Delta: " + str(tl_ts - last_tl_ts) + ")\n"
    frame += term_map(cur_wp, tl_wp, tl_status, wps)

    # Update timestamps
    last_pos_ts = pos_ts
    last_wp_ts = wp_ts
    last_tl_ts = tl_ts

    return frame

def set_text_frame(msg):
    '''
    Clear the screen and replace it with the given message text frame
    '''
    s = "\033[?25l" + "\033[1J" + "\033[1;1H" + msg + "\033[?25h"
    print(s)

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
KPH_TO_MPS = (1000.0 / 3600.0)
MPS_TO_MPH = 2.236940

###############################################################################
# Waypoint Updater                                                            #
###############################################################################

class WaypointUpdater(object):
    def __init__(self):

        # Initialize our node
        rospy.init_node('waypoint_updater')

        # Get parameters we need
        # NOTE: Sim default looks to be ~25 MPH
        max_velocity = rospy.get_param('/waypoint_loader/velocity', 0.0) * KPH_TO_MPS

        # Define the data that will come into this node
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_light_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.waypoints_cb)

        # Define the data that is exiting this node
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Define our PathPlanner object that will do all the heavy lifting
        self.planner = PathPlanner(speed_limit=max_velocity)

        # Keep track of header meta data
        self.frame_id = None
        self.seq = 0

        # Keep track of update timing
        self.last_update_ts = None

        # Begin publishing
        self.run()

    def pose_cb(self, msg):
        '''
        A callback function for the vehicle current pose subscriber

        /current_pose returns the following values as 'msg'

        msg.header.seq
        msg.header.stamp.secs
        msg.header.stamp.nsecs
        msg.header.frame_id
        msg.pose.position.x
        msg.pose.position.y
        msg.pose.position.z
        msg.pose.orientation.x
        msg.pose.orientation.y
        msg.pose.orientation.z
        msg.pose.orientation.w
        '''
        self.planner.set_vehicle_pose(msg.pose)
        self.frame_id = msg.header.frame_id

    def waypoints_cb(self, waypoints):
        '''
        A callback function for the base waypoints subscriber

        Waypoints are assumed to be in the order the car should follow them.

        /base_waypoints returns the following values as 'waypoints'

        waypoints.header.seq
        waypoints.header.stamp.secs
        waypoints.header.stamp.nsecs
        waypoints.header.frame_id
        waypoints.waypoints[].pose.header.seq
        waypoints.waypoints[].pose.header.stamp.secs
        waypoints.waypoints[].pose.header.stamp.nsecs
        waypoints.waypoints[].pose.header.frame_id
        waypoints.waypoints[].pose.pose.position.x
        waypoints.waypoints[].pose.pose.position.y
        waypoints.waypoints[].pose.pose.position.z
        waypoints.waypoints[].pose.pose.orientation.x
        waypoints.waypoints[].pose.pose.orientation.y
        waypoints.waypoints[].pose.pose.orientation.z
        waypoints.waypoints[].pose.pose.orientation.w
        waypoints.waypoints[].twist.header.seq
        waypoints.waypoints[].twist.header.stamp.secs
        waypoints.waypoints[].twist.header.stamp.nsecs
        waypoints.waypoints[].twist.header.frame_id
        waypoints.waypoints[].twist.twist.linear.x
        waypoints.waypoints[].twist.twist.linear.y
        waypoints.waypoints[].twist.twist.linear.z
        waypoints.waypoints[].twist.twist.angular.x
        waypoints.waypoints[].twist.twist.angular.y
        waypoints.waypoints[].twist.twist.angular.z
        '''
        self.planner.set_waypoints(waypoints.waypoints)
        self.frame_id = waypoints.header.frame_id

    def traffic_light_cb(self, msg):
        '''
        Callback for /traffic_waypoint message.
        '''
        self.planner.set_traffic_light_wp(msg.data)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it
        # later
        pass

    def pub_waypoints(self, waypoints):
        '''
        Publishes the next set of waypoints with proper headers
        '''
        lane = Lane()
        lane.header.frame_id = self.frame_id
        lane.header.seq = self.seq
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)
        self.seq = self.seq + 1

    def run(self):
        '''
        The control loop for the final_waypoint publisher
        '''

        # Set the refresh rate
        rate = rospy.Rate(2)

        # Use the start time sentinel to know when to start processing
        start_time = 0

        # Poll until we're ready to start
        while not start_time:
            start_time = rospy.Time.now().to_sec()

        # Control loop
        while not rospy.is_shutdown():

            # Get the next waypoints if possible
            try:
                next_waypoints = self.planner.get_next_waypoints(LOOKAHEAD_WPS)
            except PathPlannerException as e:
                rate.sleep()
                continue;

            # Publish
            self.pub_waypoints(next_waypoints)
            self.last_update_ts = rospy.get_time()

            # Debug output, gather status
            # frame = update_frame(self.planner, self.last_update_ts)
            # set_text_frame(frame)

            # Sleep
            rate.sleep()

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
