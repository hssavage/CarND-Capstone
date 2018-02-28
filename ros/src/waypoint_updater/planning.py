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
###############################################################################
'''

# Might not need this in the end, currently using it for logging
# We could provide a transparent logging interface instead?
import rospy

# Used for the sqrt, sin, cos functions
from math import sin, cos, sqrt

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

    def __init__(self, vehicle_pose=None, waypoints=None, speed_limit=None):
        '''
        Initializes the PathPlanner object with optional current vehicle
        position and list of waypoints
        '''

        # Map waypoints
        self.waypoints = waypoints

        # Max allowed speed
        self.speed_limit = speed_limit

        # The current up to date position (location + orientation) of the
        # vehicle
        self.vehicle_pose = vehicle_pose

        # Keep track of closest waypoints to our pose
        self.closest_behind = -1
        self.closest_in_front = -1

    def set_vehicle_pose(self, pose):
        '''
        Sets the planner's known vehicle pose and updates the next and previous
        waypoints if we have a waypoint set to use.
        '''

        # Update the pose
        self.vehicle_pose = pose

        # If we have waypoints, update where we are relative to them
        if(self.waypoints != None):
            self.__set_closest_waypoints()

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
            self.__set_closest_waypoints()

    def set_speed_limit(self, vel):
        '''
        Update the max speed we should assign to waypoints
        '''
        self.speed_limit = vel

    def __set_closest_waypoints(self):
        '''
        Sets the closest waypoint values for both in front and behind.

        Sets the closest point, determines if its in front or behind, and
        assigns the closest_behind and closest_in_front accordingly

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
        behind = self.__wp_is_behind(self.waypoints[i].pose.pose.position)
        if(behind):
            self.closest_behind = closest
            self.closest_in_front = closest + 1 if closest + 1 < len(self.waypoints) else 0
        else:
            self.closest_behind = closest - 1 if closest - 1 >= 0 else len(self.waypoints) - 1
            self.closest_in_front = closest

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

        # Return the subset of way points
        s = self.closest_in_front
        e = min(self.closest_in_front + count, len(self.waypoints))
        return self.waypoints[s:e]
