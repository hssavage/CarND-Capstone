'''
###############################################################################
# detector.py                                                                 #
# -----------                                                                 #
# This module contains the source for the detector objects required to handle #
# traffic light detection.                                                    #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 3/08/2018          | Henry Savage  | Initial pass on the code, adding a | #
# |                    |               | BaseDetector and a SimDetector.    | #
# |                    |               | Also moved some of the suggested   | #
# |                    |               | code over from tl_detector.py. The | #
# |                    |               | SimDetector can record images.     | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Added comments to some functions.  | #
# |                    |               | Also checked for null waypoints    | #
# |                    |               | list in the SimDetector.           | #
# +--------------------+---------------+------------------------------------+ #
# | 3/15/2018          | Henry Savage  | Small update to image output       | #
# |                    |               | location.                          | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# Might not need this in the end, currently using it for logging
# We could provide a transparent logging interface instead?
import rospy

# Data Types
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

# For our Legit Detector
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import cv2

# Transform library, used to make quaternion transforms from the pose data type
from tf.transformations import euler_from_quaternion

# For resolving file paths when gathering and saving data
from os.path import exists, isfile, isdir, abspath, expanduser
from distutils.dir_util import mkpath
import datetime

# Used for the sqrt, sin, cos functions
from math import sin, cos, sqrt

#-----------------------------------------------------------------------------#
# Base Detector                                                               #
#-----------------------------------------------------------------------------#

class BaseDetector(object):
    '''
    A base traffic light detection class that can be used to implement both
    the simulated detector and the "real" detector
    '''

    def __init__(self, vehicle_pose=None, waypoints=None, lights_config=None,
                 image=None, light_pose=-1, light_state=TrafficLight.UNKNOWN):
        '''
        The default constructor for the BaseDetector class. Defines all the
        high level objects that should be needed for a traffic light detector

        TODO: Can some of these maybe go away? Do we need to all of them?

            vehicle_pose: <Pose Msg Object> Optional argument to set current
                          vehicle position and orientation.
            waypoints: <Waypoints Object List> Optional argument to set the
                       list of base waypoints.
            lights_config: <Dictionary> A Configuration object containing
                           traffic light stop line info and camera info
                           TODO: We can probably break this up
            image: <Image Msg Object> Optional argument to set the image
            light_pose: <Pose Msg Object> Optional argument to set the light
                        position.
            light_status: <Int> Optional argument to set the initial light
                          state
        '''

        # The vehicle's most up to date pose information
        self.vehicle_pose = vehicle_pose

        # The most up to date list of the waypoints
        self.waypoints = waypoints

        # The map of traffic lights in our area, plus config data
        self.lights_config = lights_config
        self.lights = {}

        # The most up to date camera image
        self.image = image

        # To translate the image message to a cv2 image
        # TODO: Do we want this here, or do we want to assume we're being
        # GIVEN a cv2 compatible image?
        self.bridge = CvBridge()

        # The most up to date traffic light state
        self.light_pose = light_pose
        self.light_ind = -1
        self.light_state = light_state

    def set_vehicle_pose(self, pose):
        '''
        Set the current vehicle pose
        '''
        self.vehicle_pose = pose

    def set_waypoints(self, waypoints):
        '''
        Set the current waypoint list
        '''
        self.waypoints = waypoints

    def set_image(self, image):
        '''
        Set the current image
        '''
        self.image = image

    def traffic_light_status(self):
        '''
        Base traffic light detection status interface
        '''
        return self.light_ind, self.light_state

#-----------------------------------------------------------------------------#
# Simulated Training Detector                                                 #
#-----------------------------------------------------------------------------#

class SimDetector(BaseDetector):
    '''
    A Traffic light "Detector" that uses the ground truth data from thats
    provided by the simulator
    '''

    def __init__(self,vehicle_pose=None, waypoints=None, lights_config=None,
                 image=None, light_pose=-1, light_state=TrafficLight.UNKNOWN,
                 save_path=None):
        '''
        Detector constructor

        Takes all the normal parameters from the base class, plus:
            save_data - [True | False] True, if you want to save image data
            save_path - <string> Directory location to save data to
        '''

        # Call the super class init function
        BaseDetector.__init__(self,
                              vehicle_pose=vehicle_pose,
                              waypoints=waypoints,
                              lights_config=lights_config,
                              image=image,
                              light_pose=light_pose,
                              light_state=TrafficLight.UNKNOWN)

        # Do we want to gather training data?
        self.save_data = False
        self.save_path = save_path
        self.save_count = 0

        # Resolve the path
        if(self.save_path != None):
            self.save_path = abspath(expanduser(self.save_path))

            # See if the path exists or makes sense. If it does then make
            # the data collection directory to save images to
            if(exists(self.save_path) and isfile(self.save_path)):
                self.save_data = False
                rospy.logwarn("Given path is an existing file. Image capture has been turned off.")
            else:
                self.save_path = self.save_path + "/images_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                mkpath(self.save_path)
                self.save_data = True

        # Subscribe to the ground truth data secretly here
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                         self.vehicle_traffic_lights_cb, queue_size=1)

    def vehicle_traffic_lights_cb(self, msg):
        '''
        Callback for the ground truth traffic light data

        msg.header.seq
        msg.header.stamp.secs
        msg.header.stamp.nsecs
        msg.header.frame_id
        msg.lights[].state [0=RED, 1=YELLOW, 2=GREEN, 4=UNKNOWN]
        msg.lights[].pose.header.stamp.secs
        msg.lights[].pose.header.stamp.nsecs
        msg.lights[].pose.header.frame_id
        msg.lights[].pose.pose.position.x
        msg.lights[].pose.pose.position.y
        msg.lights[].pose.pose.position.z
        msg.lights[].pose.pose.orientation.x
        msg.lights[].pose.pose.orientation.y
        msg.lights[].pose.pose.orientation.z
        msg.lights[].pose.pose.orientation.w
        '''

        # Because of what I'm about to do, having a light list
        # without a a vehicle position is pointless. We can
        # wait until we have a vehicle position
        if(self.vehicle_pose is None):
            return

        # Same goes for waypoints
        if(self.waypoints is None):
            return

        # Grab the up to date light statuses
        self.lights = msg.lights

        # Transform light (X,Y) to vehicle coords
        o_x = self.vehicle_pose.position.x
        o_y = self.vehicle_pose.position.y
        o_z = self.vehicle_pose.position.z
        _, _, o_theta = euler_from_quaternion([self.vehicle_pose.orientation.x,
                                              self.vehicle_pose.orientation.y,
                                              self.vehicle_pose.orientation.z,
                                              self.vehicle_pose.orientation.w])

        # Do a coordinate shift on the light set so its relative to us
        # Find the closest light in front of us as well while we're doing it
        l_x = 0
        l_y = 0
        c_i = None
        c_dist = None
        for i in range(0, len(self.lights)):

            # Light World coords
            l_x = self.lights[i].pose.pose.position.x
            l_y = self.lights[i].pose.pose.position.y
            l_z = self.lights[i].pose.pose.position.z

            # Origin shift
            n_x = l_x - o_x
            n_y = l_y - o_y
            n_z = l_z - o_z

            # Rotation shift
            n_x = n_x * cos(0 - o_theta) - n_y * sin(0 - o_theta)
            n_y = n_y * cos(0 - o_theta) + n_x * sin(0 - o_theta)

            # No looking backwards
            if(n_x < 0):
                continue

            # Set value
            self.lights[i].pose.pose.position.x = n_x
            self.lights[i].pose.pose.position.y = n_y
            self.lights[i].pose.pose.position.z = n_z

            # Set closest traffic light
            dist = sqrt((0 - n_x)**2 + (0 - n_y)**2)
            if(c_dist is None or c_dist > dist):
                c_dist = dist
                c_i = i

        # Find the traffic light stop line waypoint (x, y)
        sl_x = self.lights_config['stop_line_positions'][c_i][0]
        sl_y = self.lights_config['stop_line_positions'][c_i][1]

        # Which base waypoint is closest?
        # TODO: We can cache this since the lights dont move
        wp = -1
        wp_dist = None
        for i in range(0, len(self.waypoints)):
            wp_x = self.waypoints[i].pose.pose.position.x
            wp_y = self.waypoints[i].pose.pose.position.y
            dist = sqrt((sl_x - wp_x)**2 + (sl_y - wp_y)**2)
            if(wp_dist is None or dist < wp_dist):
                wp = i
                wp_dist = dist

        # Set final status
        self.light_ind = wp
        self.light_state = self.lights[c_i].state

        # rospy.loginfo("Vehicle Position: (" + str(self.vehicle_pose.position.x) + ", " + str(self.vehicle_pose.position.y) + ")")
        # rospy.loginfo("Closest Light " + str(c_i) + ": (" + str(self.lights[c_i].pose.pose.position.x) + ", " + str(self.lights[c_i].pose.pose.position.y) + ") -- State: " + str(self.lights[c_i].state))
        # rospy.loginfo("Stop Line: (" + str(sl_x) + ", " + str(sl_y) + ")")
        # rospy.loginfo("Stop waypoint: IND " + str(wp))

    def set_image(self, image):
        '''
        Override the set_image() function so we can do data gather potentially
        when an image comes in
        '''

        # No need to do this if we're not saving any data
        if(not self.save_data):
            return

        # Grab the image - convert it to something we can use and save
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")

        # We can't do anything here if we don't have enough info
        if(self.vehicle_pose is None or self.waypoints is None or self.lights is None):
            return

        # TODO: Determine if what we're looking at is the next up light
        #       and grab the state if it is. If we don't, we're going to
        #       get mostly pictures of hills...

        # Finally, save the image
        file_name = self.save_path + "/" + "img_" + str(self.save_count) + "_s" + str(self.light_state) + ".jpg"
        cv2.imwrite(file_name, self.image)
        self.save_count += 1

    def traffic_light_status(self):
        '''
        '''
        return self.light_ind, self.light_state

#-----------------------------------------------------------------------------#
# "Real" Detector                                                             #
#-----------------------------------------------------------------------------#

#NOTE: Everything below here is pretty much garbage :/ Beware.

class Detector(BaseDetector):
    '''
    '''

    def __init__(self):
        '''
        '''

        # Pulled this from the old code for now
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.last_light_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.state_count_threshold = 3
        self.has_image = False
        self.camera_image = None

    def image_cb(self, image):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.image = image
        light_wp = -1
        state = None
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.vehicle_pose)

        # TODO find the closest visible traffic light (if one exists)
        # ... light_wp = ?

        if light:
            state = self.get_light_state(light)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.light_state != state:
            self.state_count = 0
            self.light_state = state
        elif self.state_count >= self.state_count_threshold:
            self.last_light_state = self.light_state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def traffic_light_status(self):
        '''
        '''
        return self.light_ind, self.light_state
