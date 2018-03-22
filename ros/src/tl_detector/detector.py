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
# | 3/17/2018          | Henry Savage  | Changed the fake TL reporting to   | #
# |                    |               | a new method where I precompute    | #
# |                    |               | the closest traffic light number,  | #
# |                    |               | waypoint for that light and the    | #
# |                    |               | waypoint for the associated stop   | #
# |                    |               | line.                              | #
# +--------------------+---------------+------------------------------------+ #
# | 3/18/2018          | Jason Clemons | Added inference engine to the      | #
# |                    |               | simulated traffic light detector.  | #
# |                    |               | This will run inference if the     | #
# |                    |               | flag is set.  This adds a publish  | #
# |                    |               | channel of the marked up image     | #
# |                    |               | result                             | #
# +--------------------+---------------+------------------------------------+ #
# | 3/21/2018          | Henry Savage  | Updated precomputed set of next    | #
# |                    |               | traffic light to assume there isnt | #
# |                    |               | any wrapping around occuring. All  | #
# |                    |               | "next lights" are set to waypoint  | #
# |                    |               | '-1' for the end of the set where  | #
# |                    |               | an index truly isn't known.        | #
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
import numpy as np
from tl_inference_engine import draw_bounding_boxes
from tl_inference_engine import DetectionInferenceEngine

#-----------------------------------------------------------------------------#
# Base Detector                                                               #
#-----------------------------------------------------------------------------#

class BaseDetector(object):
    '''
    A base traffic light detection class that can be used to implement both
    the simulated detector and the "real" detector
    '''

    def __init__(self, vehicle_pose=None, waypoints=None, lights_config=None, model_path=None):
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
                           TODO: We can probably break this up?
        '''

        # The vehicle's most up to date pose information
        self.vehicle_pose = vehicle_pose

        # The most up to date list of the waypoints
        self.waypoints = waypoints

        # The map of traffic lights in our area, plus config data
        self.lights_config = lights_config
        self.lights = {}

        #The path to the tf model to use
        self.model_path = model_path

        # The most up to date camera image
        self.image = None

        # The most up to date camera image
        self.detector_result_image_msg = None

        # To translate the image message to a cv2 image
        # TODO: Do we want this here, or do we want to assume we're being
        # GIVEN a cv2 compatible image?
        self.bridge = CvBridge()

        # The most up to date traffic light state
        self.light_ind = -1
        self.light_state = TrafficLight.UNKNOWN

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
                 save_path=None, model_path=None):
        '''
        Detector constructor

        Takes all the normal parameters from the base class, plus:
            save_path - <string> Directory location to save data to, None
                        if you dont want to save any data
        '''

        # Call the super class init function
        BaseDetector.__init__(self,
                              vehicle_pose=vehicle_pose,
                              waypoints=waypoints,
                              lights_config=lights_config)

        # Cache the next light/stop line for each waypoint
        self.next_light = None

        # Which waypoint are we closests to?
        self.closest_wp = -1

        # Do we want to gather training data?
        self.save_data = False
        self.save_path = save_path
        self.save_count = 0

        #Path to the model for inference
        self.model_path = model_path

        #Instantiate a inference engine
        # TODO Move thresh to parameter server
        self.run_inference_engine = False
        self.inference_engine = DetectionInferenceEngine(model_path=model_path, confidence_thr= 0.02)

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

    def set_vehicle_pose(self, pose):
        '''
        Set the current vehicle pose
        '''
        self.vehicle_pose = pose

        # We cant determine the closest waypoint if we dont have any
        if(self.waypoints is None):
            return

        # Set the vehicle's x, y, and yaw
        v_x = pose.position.x
        v_y = pose.position.y

        # Determine closest waypoint
        c_wp = -1
        c_dist = -1
        for i in range(0, len(self.waypoints)):
            w_x = self.waypoints[i].pose.pose.position.x
            w_y = self.waypoints[i].pose.pose.position.y
            dist = sqrt((v_x - w_x)**2 + (v_y - w_y)**2)
            if(dist < c_dist or c_dist == -1):
                c_dist = dist
                c_wp = i

        # Set the closest waypoint
        self.closest_wp = c_wp

    def set_waypoints(self, waypoints):
        '''
        Set the current waypoint list
        '''
        self.waypoints = waypoints

        # Once we have waypoints, try to calculate and cache the index of
        # the next traffic light for each waypoint.
        if(self.next_light is None):
            self.set_next_lights()

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

        # Set the up to date light statuses
        self.lights = msg.lights

        # Once we have lights, try to calculate and cache the index of
        # the next traffic light for each waypoint.
        if(self.next_light is None):
            self.set_next_lights()
            return

        # If we don't know where we are, we cant update the light status
        if(self.closest_wp == -1):
            return

        # Set final status:
        #  - light_ind -> where to stop for next light
        #  - light_state -> status of next light
        # NOTE: if there's no know light, the "next light" waypoint is -1
        # and we can set the status to unknown
        if(self.next_light[self.closest_wp][1] != -1):
            self.light_ind = self.next_light[self.closest_wp][2]
            self.light_state = self.lights[self.next_light[self.closest_wp][0]].state
        else:
            self.light_ind = -1
            self.light_state = TrafficLight.UNKNOWN

    def set_next_lights(self):
        '''
        '''

        # If we don't have waypoints or lights, we can't cache the next light
        # per waypoint
        if(self.waypoints is None or len(self.waypoints) == 0 or
           self.lights is None or len(self.lights) == 0):
            return

        # First, init the next_light structure
        self.next_light = [[-1, -1, -1] for i in range(0, len(self.waypoints))]

        # Next, we're going to find the closest waypoint to each stop light
        l_closest = [-1 for i in range(0, len(self.lights))]
        for i in range(0, len(self.lights)):
            l_x = self.lights[i].pose.pose.position.x
            l_y = self.lights[i].pose.pose.position.y
            c_wp = -1
            c_dist = -1
            for j in range(0, len(self.waypoints)):
                w_x = self.waypoints[j].pose.pose.position.x
                w_y = self.waypoints[j].pose.pose.position.y
                dist = sqrt((l_x - w_x) * (l_x - w_x) + (l_y - w_y) * (l_y - w_y))

                if(c_dist == -1 or c_dist > dist):
                    c_dist = dist
                    c_wp = j

                if(c_dist == 0):
                    break

            l_closest[i] = c_wp

        # Next, find the closest waypoint to each traffic light stop line
        sl_closest = [-1 for i in range(0, len(self.lights))]
        for i in range(0, len(self.lights)):
            sl_x = self.lights_config['stop_line_positions'][i][0]
            sl_y = self.lights_config['stop_line_positions'][i][1]

            wp = -1
            wp_dist = -1
            for j in range(0, len(self.waypoints)):
                w_x = self.waypoints[j].pose.pose.position.x
                w_y = self.waypoints[j].pose.pose.position.y
                dist = sqrt((sl_x - w_x) * (sl_x - w_x) + (sl_y - w_y) * (sl_y - w_y))
                if(wp_dist == -1 or dist < wp_dist):
                    wp = j
                    wp_dist = dist

                if(wp_dist == 0):
                    break

            sl_closest[i] = wp

        # Next, do a single pass on the waypoints to set traffic light
        # and stop line locations
        for i in range(0, len(l_closest)):
            self.next_light[l_closest[i]][0] = i             # light index
            self.next_light[l_closest[i]][1] = l_closest[i]  # light wp
            self.next_light[l_closest[i]][2] = sl_closest[i] # stop line wp

        # Next, fill in the spots in between by going backwards
        n_l = -1
        n_wp = -1
        n_sl = -1
        for i in reversed(range(0, len(self.next_light))):
            if(self.next_light[i][0] != -1):
                n_l  = self.next_light[i][0]
                n_wp = self.next_light[i][1]
                n_sl = self.next_light[i][2]
            else:
                self.next_light[i][0] = n_l
                self.next_light[i][1] = n_wp
                self.next_light[i][2] = n_sl

        # Handle the loop back case (though this may not matter)
        # i = len(self.next_light) - 1
        # while(self.next_light[i][0] == -1):
        #     self.next_light[i] = self.next_light[0]
        #     i -= 1

        # for i in range(0, len(self.next_light)):
        #     print(str(i) + ": " + str(self.next_light[i][0]) + ", " + str(self.next_light[i][1]) + ", " + str(self.next_light[i][2]))

    def set_image(self, image):
        '''
        Override the set_image() function so we can do data gather potentially
        when an image comes in
        '''

        # Grab the image - convert it to something we can use and save
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")

        # We can't do anything here if we don't have enough info
        if(self.vehicle_pose is None or self.waypoints is None or self.lights is None):
            return

        # TODO: Determine if what we're looking at is the next up light
        #       and grab the state if it is. If we don't, we're going to
        #       get mostly pictures of hills...


        if self.run_inference_engine:
            # Grab the detector result image
            # TODO: add in the marking of the image
            # I plan on using the simulated version so the vehicle can run
            detection_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # The model expects batches so ie 4 dimensional data so add in a single dimension to
            # get a bactch size of 1
            detection_image_ex = np.expand_dims(detection_image, axis=0)

            # Returns a dictionaru with bounding boxes and classes
            output_dict =self.inference_engine.run_inference(detection_image_ex)

            # Just some debug output of the live boxes
            print('box coordinates: ', output_dict['detection_boxes'])

            # draw the bounding boxes on the image
            detected_image = draw_bounding_boxes(detection_image, output_dict['detection_boxes'], output_dict['detection_classes'], self.inference_engine.color_list,thickness=4)

            # Convert the detection image to an image message and store for the publish loop
            self.detector_result_image_msg = self.bridge.cv2_to_imgmsg(detected_image, 'rgb8')#encoding="passthrough")

        # No need to do this if we're not saving any data
        if(not self.save_data):
            return

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


