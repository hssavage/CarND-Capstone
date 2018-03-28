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
# | /final_waypoints   | 2hz / 1 item  | The updated subset of waypoints    | #
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
# | 3/25/2018          | Henry Savage  | Updated the debug interface to use | #
# |                    |               | the new /debug topic, which is a   | #
# |                    |               | system wide debug interface        | #
# |                    |               | defined by the debug_output node   | #
# +--------------------+---------------+------------------------------------+ #
# | 3/27/2018          | Henry Savage  | Added lines to debug output.       | #
# |                    |               | Decreased the lookahead points and | #
# |                    |               | increased the rate of publishing   | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# ROS core stuff
import rospy

# Data types
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from debug_msgs.msg import Debug

# Our waypoint planner
from planning import PathPlanner, PathPlannerException

#-----------------------------------------------------------------------------#
# Debug Output                                                                #
#-----------------------------------------------------------------------------#

last_pos_ts = None
last_tl_ts = None
last_wp_ts = None

def debug_frame(planner, ts):
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
        tl_wp = "?"

    # Build frame
    frame = ""
    frame += "Timestamp: " + str(ts) + "\n"
    frame += "Speed Limit: " + str(s_limit) + " m/s | " + str(s_limit * MPS_TO_MPH) + " mph\n"
    frame += "Current Position: (" + str(cur_x) + ", " + str(cur_y) + ", " + str(cur_z) + ") "
    frame += "(Updated: " + str(pos_ts) + ", Delta: " + str(pos_ts - last_pos_ts) + ")\n"
    frame += "Current Waypoint: " + str(cur_wp) + "/" + str(total_wps) + " "
    frame += "(Updated: " + str(wp_ts) + ", Delta: " + str(wp_ts - last_wp_ts) + ")\n"
    frame += "Next Traffic Light: " + str(tl_wp) + "\n"
    frame += "Next Traffic Status: " + str(tl_status) + " "
    frame += "(Updated: " + str(tl_ts) + ", Delta: " + str(tl_ts - last_tl_ts) + ")"

    # Update timestamps
    last_pos_ts = pos_ts
    last_wp_ts = wp_ts
    last_tl_ts = tl_ts

    return frame

#-----------------------------------------------------------------------------#
# Waypoint Updater                                                            #
#-----------------------------------------------------------------------------#

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
KPH_TO_MPS = (1000.0 / 3600.0)
MPS_TO_MPH = 2.236940

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

        # Debug message publisher
        self.debug_output = rospy.Publisher('/debug', Debug, queue_size=2)

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

    def debug(self, line, s):
        '''
        '''
        try:
            d = Debug()
            d.line_start = line
            d.text = str(s)
            self.debug_output.publish(d)
        except Exception as e:
            print("WHAT THE EFF: " + str(e))
            return

    def run(self):
        '''
        The control loop for the final_waypoint publisher
        '''

        # Set the refresh rate
        rate = rospy.Rate(8)

        # Use the start time sentinel to know when to start processing
        start_time = 0

        # Poll until we're ready to start
        while not start_time:
            start_time = rospy.Time.now().to_sec()

        # Control loop
        while not rospy.is_shutdown():

            # For latency tracking
            start = rospy.get_time()

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
            latency = (rospy.get_time() - start) * 1000.0
            frame = debug_frame(self.planner, self.last_update_ts)
            lines = len(frame.split("\n"))
            self.debug(0, "----------------------------- WAYPOINT UPDATER -----------------------------")
            self.debug(1, frame)
            self.debug(1 + lines, "Node Latency: " + str(latency) + " ms")

            # Sleep
            rate.sleep()

#-----------------------------------------------------------------------------#
# Run                                                                         #
#-----------------------------------------------------------------------------#

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
