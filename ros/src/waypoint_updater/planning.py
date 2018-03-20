'''
###############################################################################
# planning.py                                                                 #
# -----------                                                                 #
# This module contains the source for the planning objects required to handle #
# waypoints, localization and perception info and return a set of final       #
# waypoints for the vehicle to follow with paired velocities. These waypoints #
# are eventually handed to the control module to carry out.                   #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 2/22/2018          | Henry Savage  | Initial pass on the code           | #
# +--------------------+---------------+------------------------------------+ #
# | 2/22/2018          | Henry Savage  | Added acceleration values to help  | #
# |                    |               | comfortably reduce speed at stop   | #
# |                    |               | lights. Added a hook to update the | #
# |                    |               | planners traffic light status. Now | #
# |                    |               | handle slowing down and stopping   | #
# |                    |               | a given stop index.                | #
# +--------------------+---------------+------------------------------------+ #
# | 3/15/2018          | Henry Savage  | Changed red light logic from       | #
# |                    |               | '!= -1' to '> 0'                   | #
# +--------------------+---------------+------------------------------------+ #
# | 3/19/2018          | Henry Savage  | Removed extra next waypoint code.  | #
# |                    |               | Updated stop line logic to use the | #
# |                    |               | length of the vehicle in the stop  | #
# |                    |               | velocity calculations and lowered  | #
# |                    |               | the max decel value. The vehicle   | #
# |                    |               | should now stop behind the line.   | #
# +--------------------+---------------+------------------------------------+ #
# | 3/19/2018 (2)      | Henry Savage  | Added updated timestamps so we can | #
# |                    |               | begin tracking and monitoring any  | #
# |                    |               | unacceptable or worrisome delays   | #
# |                    |               | in data updates and handle stale   | #
# |                    |               | values.                            | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# TODO:
#  - Pick adequate deceleration values (or fix how we do it) so we can
#    consistently stop at traffic lights on time

# Might not need this in the end, currently using it for logging
# We could provide a transparent logging interface instead?
import rospy

# Used for the sqrt, sin, cos functions
from math import sin, cos, sqrt

# Used to deep copy waypoint objects so we don't edit the reference points
from copy import deepcopy

# Transform library, used to make quaternion transforms from the pose data type
from tf.transformations import euler_from_quaternion

# Understand the data types we're using. Not sure if we need this since we're
# never creating any actual objects
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

#-----------------------------------------------------------------------------#
# Path Planner Exceptions                                                     #
#-----------------------------------------------------------------------------#

# Generic PathPlannerException
class PathPlannerException(Exception):
    pass

#-----------------------------------------------------------------------------#
# Path Planner Class                                                          #
#-----------------------------------------------------------------------------#

class PathPlanner():
    '''
    '''

    def __init__(self, vehicle_pose=None, waypoints=None, speed_limit=None,
                 max_decel=-1.3, max_accel=10.0):
        '''
        Initializes the PathPlanner object with optional current vehicle
        position and list of waypoints
        '''

        # Map waypoints
        self.waypoints = waypoints

        # Max allowed speed
        self.speed_limit = speed_limit

        # Max allowed acceleration and deceleration values, enforce signs
        self.max_decel = max_decel if max_decel <= 0 else -1.0*max_decel
        self.max_accel = max_accel if max_accel >= 0 else -1.0*max_accel

        # Vehicle dimensions info - honestly should be a topic that the
        # runtime platform itself knows to put out so our code can not
        # care.
        self.vehicle_length = 4.47

        # The current up to date position (location + orientation) of the
        # vehicle
        self.vehicle_pose = vehicle_pose
        self.vehicle_pose_ts = None

        # Keep track of closest waypoints to our pose
        self.next_waypoint = -1
        self.next_waypoint_ts = None

        # Keep track of a target stop point from the traffic light node or
        # obstacle node
        self.traffic_light_ind = -1
        self.traffic_light_ind_ts = None

    def set_vehicle_pose(self, pose):
        '''
        Sets the planner's known vehicle pose and updates the next and previous
        waypoints if we have a waypoint set to use.
        '''

        # Update the pose
        self.vehicle_pose = pose
        self.vehicle_pose_ts = rospy.get_time()

        # If we have waypoints, update where we are relative to them
        if(self.waypoints != None):
            self.__set_closest_waypoint()

    def set_waypoints(self, waypoints):
        '''
        Sets the planner's waypoint set and updates the next and previous
        waypoints if we have a vehicle position to use.
        '''

        # Update the waypoints
        self.waypoints = waypoints

        # If we have a pose, update where we are relative to the new waypoints
        # and the pose
        if(self.vehicle_pose != None):
            self.__set_closest_waypoint()

    def set_traffic_light_wp(self, wp_ind):
        '''
        Handle a red traffic light waypoint
        '''
        self.traffic_light_ind = wp_ind
        self.traffic_light_ind_ts = rospy.get_time()

    def set_speed_limit(self, vel):
        '''
        Update the max speed we should assign to waypoints
        '''
        self.speed_limit = vel

    def __set_closest_waypoint(self):
        '''
        Sets the closest waypoint value to the closests waypoint in front of the vehicle.

        Waypoint setting is done by finding the overall closest point to the vehicle
        and determining if that point is in front of or behind the vehicle. If it
        is behind, its assumed that the next waypoint would be in front of the
        vehicle and that point is used.

        Complexity: O(n), n = number of waypoints
        '''

        # Make sure we can even do this thing
        if(self.vehicle_pose is None or self.waypoints is None or len(self.waypoints) == 0):
            raise PathPlannerException("Not enough information to find closest waypoint")

        closest = -1
        closest_dist = float('inf')
        current_position = self.vehicle_pose.position
        behind = False

        # Find the closest waypoint to our vehicle's position
        for i in range(len(self.waypoints) - 1):
            wp_position = self.waypoints[i].pose.pose.position
            dist = self.__get_distance2d(current_position, wp_position)
            if(dist < closest_dist):
                closest = i
                closest_dist = dist

        # Error state
        if(closest == -1):
            raise PathPlannerException("Failed to find closest waypoint")

        # Figure out if the closest waypoint is behind us so we can assign
        # in the points in the correct order
        behind = self.__wp_is_behind(self.waypoints[closest].pose.pose.position)
        if(behind):
            closest = closest + 1 if closest + 1 < len(self.waypoints) else len(self.waypoints)

        self.next_waypoint = closest
        self.next_waypoint_ts = rospy.get_time()

    def __get_distance2d(self, p1, p2):
        '''
        Determines the two dimensional distance between two positions

        Returns:
            float: distance between p1 and p2

        Complexity: O(1)
        '''
        return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def __get_euler(self, quat_orient):
        '''
        Returns the yaw, pitch and roll values from a quaternion orientation

        Returns:
            list<float>: [pitch, roll, yaw]

        Complexity: O(1)
        '''
        return euler_from_quaternion([quat_orient.x, quat_orient.y,
                                      quat_orient.z, quat_orient.w])

    def __wp_is_behind(self, wp_pos):
        '''
        Determines if a waypoint is behind the current vehicle position.

        Determination is made by doing a partial perspective transformation on
        the waypoint value given the vehicle position. In this case, all we
        care about is the new transformed X value. Behind means that the new
        transformed X value is <= to the origin.

        Returns:
            bool: True if the waypoint is behind, False otherwise.

        Complexity: O(1)
        '''

        if(self.vehicle_pose == None):
            raise(PathPlannerException("No current vehicle pose information available"))

        o_x = self.vehicle_pose.position.x
        o_y = self.vehicle_pose.position.y
        _, _, o_theta = self.__get_euler(self.vehicle_pose.orientation)

        # Origin shift
        n_x = wp_pos.x - o_x
        n_y = wp_pos.y - o_y

        # Rotation shift
        n_x = n_x * cos(0 - o_theta) - n_y * sin(0 - o_theta)

        return (n_x <= 0)

    def __stop_at_ind(self, stop_ind, waypoints):
        '''
        Bring the vehicle to a stop by the given stop index.

        This function works backwards from "stop_ind", using kinematics to set
        the speed at each waypoint. It assumes 0 at 'stop_ind' and will stop
        when the desired speed is greater than or equal to the existing speed
        of a waypoint. All points after stop ind are set to zero

        Args:
            stop_ind: <int> Index in the following set of waypoints to stop at
            waypoints: list<Waypoint> List of waypoints we're following, 0th is
                       our next waypoint to arrive at
        Returns:
            list<Waypoint> list of modified waypoints where the target velocity
            values have all been adjusted such that the vehicle will stop

        Complexity: O(n) where n is the number of waypoints
        '''

        # Check inputs
        if(waypoints is None):
            raise PathPlannerException("Given set of waypoints is None")
        if(stop_ind < 0 or stop_ind >= len(waypoints)):
            raise PathPlannerException("Stop index is out of bounds given the set of waypoints")

        # Book keeping variables for managing state and working backwards
        vf = 0.0
        vi = 0.0
        d = 0.0
        a = self.max_decel
        total_dist = 0;

        # Iterate on the list in reverse order
        for i in reversed(range(0, len(waypoints))):

            # Target speed for the waypoint
            orig_speed = waypoints[i].twist.twist.linear.x

            # if the index is after the stop point, we should be at zero
            if(i >= stop_ind):
                waypoints[i].twist.twist.linear.x = 0.0

            # If its between us and the stop ind then we need to start
            # maintaining state and working backwards to change speeds
            else:
                d = self.__get_distance2d(waypoints[i].pose.pose.position,
                                          waypoints[i + 1].pose.pose.position)
                if(total_dist <= self.vehicle_length / 2.0):
                    total_dist += d
                    waypoints[i].twist.twist.linear.x = 0.0
                else:
                    vi = sqrt((vf*vf) - (2.0 * a * d))
                    if(vi >= orig_speed):
                        return waypoints
                    waypoints[i].twist.twist.linear.x = vi
                    vf = vi

        return waypoints

    def get_next_waypoints(self, count):
        '''
        Returns the next <count> waypoints given the current vehicle pose and
        the known set of waypoints

        Returns:
            list<waypoint>: The next <count> waypoints in front of the vehicle

        Complexity: O(n), where n = count
        '''

        # Make sure we can do this
        if(self.vehicle_pose is None or self.waypoints is None or len(self.waypoints) == 0):
            raise PathPlannerException("Not enough information to find closest waypoint")

        # Check count requested
        if(count > len(self.waypoints)):
            raise PathPlannerException("Count requested exceeds number of waypoints available")

        # Get the subset of way points to return
        tl_ind = self.traffic_light_ind
        s = self.next_waypoint
        e = min(self.next_waypoint + count, len(self.waypoints))
        waypoints = deepcopy(self.waypoints[s:e])

        #rospy.loginfo("s: " + str(s) + ", e: " + str(e) + ", count: " + str(count) + ", next_light (global): " + str(tl_ind))

        # Check wrap around and fix e to be relative in case we need it
        # if(len(waypoints) < count and count <= len(self.waypoints)):
        #     waypoints.extend(deepcopy(self.waypoints[0:(count - len(waypoints))]))
        #     e = s + count
        #
        # Adjust traffic index for wrap around
        # if(tl_ind != -1 and s > tl_ind):
        #     tl_ind += len(self.waypoints)
        #     rospy.loginfo("next_light (gloabl, adjusted for wrap): " + str(tl_ind))

        # Determine where in our set from [s:e] our wp is
        #  and tl_ind >= s
        if(tl_ind > 0):
            tl_ind = tl_ind - s
            #rospy.loginfo("next_light (local): " + str(tl_ind))

            # Clamp stop index to 0 if it goes negative
            # tl_ind = max(0, tl_ind)
            # rospy.loginfo("next_light (local with safe stop buffer): " + str(tl_ind))

            if(tl_ind <= len(waypoints)):
                waypoints = self.__stop_at_ind(tl_ind, waypoints)

        # return final adjusted waypoints
        return waypoints
