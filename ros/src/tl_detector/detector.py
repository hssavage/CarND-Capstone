'''
###############################################################################
# detector.py                                                                 #
# -----------                                                                 #
# This module contains the source for the detector objects required to handle #
# traffic light detection.                                                    #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Outputs                                                                 | #
# +--------------------+---------------+------------------------------------+ #
# | Topic Path         | Update Rate + | Description                        | #
# |                    | Queue Size    |                                    | #
# +--------------------+---------------+------------------------------------+ #
# | /debug             | Camera Rate   | Contains the latencies for         | #
# |/tl_detector_latency|               | inference                          | #
# +--------------------+---------------+------------------------------------+ #
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
# | 3/21/2018          | Jason  Clemons| Added param for model path         | #
# |                    |               | and also changed image types to    | #
# |                    |               | png                                | #
# +--------------------+---------------+------------------------------------+ #
# | 3/24/2018          | Jason  Clemons| Added filters on the infernece     | #
# |                    |               | engine to help get to a point      | #
# |                    |               | where ready when model trained     | #
# +--------------------+---------------+------------------------------------+ #
# | 3/24/2018          | Jason  Clemons| Starting to add in real detector   | #
# |                    |               | greatly based on sim detector      | #
# |                    |               | and also using roslog outputs      | #
# +--------------------+---------------+------------------------------------+ #
# | 3/25/2018          | Jason  Clemons| Added in latency msgs              | #
# |                    |               |                                    | #
# |                    |               |                                    | #
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
from latency_msgs.msg import Traffic_light_latency


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

        # What was the last light wp and the state 
        self.last_closest_light_wp = -1
        self.last_closest_light_state = -1
        self.state_count_threshold = 2
        self.state_count = 0
        self.predicted_state = -1


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

        self.latency_pub = rospy.Publisher('/debug/tl_detector_latency', Traffic_light_latency, queue_size=1)



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
        latency = Traffic_light_latency()
        start_time = rospy.get_rostime()

        # Grab the image - convert it to something we can use and save
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")

        # We can't do anything here if we don't have enough info
        if(self.vehicle_pose is None or self.waypoints is None or self.lights is None):
            return

        # TODO: Determine if what we're looking at is the next up light
        #       and grab the state if it is. If we don't, we're going to
        #       get mostly pictures of hills...


        inference_start_time = None
        if self.run_inference_engine:
            inference_start_time = rospy.get_rostime()

            # Grab the detector result image
            # TODO: add in the marking of the image
            # I plan on using the simulated version so the vehicle can run
            detection_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # The model expects batches so ie 4 dimensional data so add in a single dimension to
            # get a bactch size of 1
            detection_image_ex = np.expand_dims(detection_image, axis=0)

            # Returns a dictionaru with bounding boxes and classes
            output_dict,predicted_light_state =self.inference_engine.run_inference(detection_image_ex)
            inference_end_time = rospy.get_rostime()
        


            if(self.last_closest_light_wp == self.light_ind):
                if(self.last_closest_light_state == predicted_light_state):

                    if(self.state_count >= self.state_count_threshold):
                        self.predicted_state = predicted_light_state
                else:
                    self.last_closest_light_state = predicted_light_state
                    self.state_count = 0

            else:
                self.last_closest_light_wp = self.light_ind    
                self.last_closest_light_state = predicted_light_state
                self.state_count = 0
            self.state_count +=1
            debug_start_time = rospy.get_rostime()

            


            rospy.loginfo('Detection Scores:  - %s',output_dict['detection_scores']) 

            # Just some debug output of the live boxes
            rospy.logdebug('Current predicted_light_state: %s', predicted_light_state)
            rospy.logdebug('Filtered predicted_light_state: %s vs %s', self.predicted_state,self.light_state)

            rospy.logdebug('box coordinates: %s', output_dict['detection_boxes'])
            rospy.logdebug('Detection Scores: %s',output_dict['detection_scores'])

            # draw the bounding boxes on the image
            detected_image = draw_bounding_boxes(detection_image, output_dict['detection_boxes'], output_dict['detection_classes'], self.inference_engine.color_list,thickness=4)

            # Convert the detection image to an image message and store for the publish loop
            self.detector_result_image_msg = self.bridge.cv2_to_imgmsg(detected_image, 'rgb8')#encoding="passthrough")
            debug_end_time = rospy.get_rostime()
    
    
        save_start_time = rospy.get_rostime()


        # No need to do this if we're not saving any data
        if(not self.save_data):
            return

        # Finally, save the image
        file_name = self.save_path + "/"+"img_" + str(self.save_count) + "_s" + str(self.light_state) + ".png"
        cv2.imwrite(file_name, self.image)
        self.save_count += 1
        save_end_time = rospy.get_rostime()

        end_time = rospy.get_rostime()
        latency.total_duration = end_time - start_time
        if(inference_start_time):
            latency.inference_duration = inference_end_time - inference_start_time
            latency.debug_duration = debug_end_time - debug_start_time
        latency.save_duration = save_end_time - save_start_time


        self.latency_pub.publish(latency)



    def traffic_light_status(self):
        '''
        '''
        return self.light_ind, self.light_state

#-----------------------------------------------------------------------------#
# "Real" Detector                                                             #
#-----------------------------------------------------------------------------#

#NOTE: Everything below here is beig transformed from error :/ Beware.

class Detector(BaseDetector):
    '''
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

        # What was the last light wp and the state 
        self.last_closest_light_wp = -1
        self.last_closest_light_state = -1
        self.state_count_threshold = 4
        self.state_count = 0

        self.predicted_light_ind  =-1
        self.predicted_state = 0

        # Do we want to gather training data?
        self.save_data = False
        self.save_path = save_path
        self.save_count = 0

        #Path to the model for inference
        self.model_path = model_path

        #Instantiate a inference engine
        # TODO Move thresh to parameter server
        self.run_inference_engine = True
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


        # Set final status:
        #  - light_ind -> where to stop for next light
        #  - light_state -> status of next light
        # NOTE: if there's no know light, the "next light" waypoint is -1
        # and we can set the status to unknown

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
            output_dict,predicted_light_state =self.inference_engine.run_inference(detection_image_ex)

            rospy.loginfo('Detection Scores:  - %s',output_dict['detection_scores']) 

            # Just some debug output of the live boxes
            print('predicted_light_state: ', predicted_light_state)
            print('box coordinates: ', output_dict['detection_boxes'])
            print('Detection Scores: ',output_dict['detection_scores'])
            # draw the bounding boxes on the image
            detected_image = draw_bounding_boxes(detection_image, output_dict['detection_boxes'], output_dict['detection_classes'], self.inference_engine.color_list,thickness=4)

            # Convert the detection image to an image message and store for the publish loop
            self.detector_result_image_msg = self.bridge.cv2_to_imgmsg(detected_image, 'rgb8')#encoding="passthrough")

        # No need to do this if we're not saving any data
        if(not self.save_data):
            return

        # If we don't know where we are, we cant update the light status
        if(self.closest_wp == -1):
            return

        rospy.loginfo('Predicted Light State - %s for light at %s', 
            predicted_light_state,self.next_light[self.closest_wp][0])


        if(self.last_closest_light_wp == self.next_light[self.closest_wp][2]):
            if(self.last_closest_light_state == predicted_light_state):

                if(self.state_count >= self.state_count_threshold):
                    self.predicted_state = predicted_light_state
                    self.predicted_light_ind = self.next_light[self.closest_wp][2]

            else:
                self.last_closest_light_state = predicted_light_state
                self.state_count = 0

        else:
            self.last_closest_light_wp = self.next_light[self.closest_wp][2]    
            self.last_closest_light_state = predicted_light_state
            self.state_count = 0
        self.state_count +=1


        self.light_state = self.predicted_state
        self.light_ind = self.predicted_light_ind


        # Finally, save the image
        file_name = self.save_path + "/"+"img_" + str(self.save_count) + "_s" + str(self.light_state) + ".png"
        cv2.imwrite(file_name, self.image)
        self.save_count += 1

    def traffic_light_status(self):
        '''
        '''
        return self.light_ind, self.light_state
