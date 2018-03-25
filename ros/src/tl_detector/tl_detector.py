#!/usr/bin/env python

'''
###############################################################################
# tl_detector.py                                                              #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the Traffic Light Detector Node as part #
# of the Self-Driving Car System. This node updates the other systems of      #
# nearby traffic lights and their status, in order for the vehicle to         #
# properly choose and enact a trajectory that will stop if needed.            #
#                                                                             #
# Input/Output:                                                               #
# -------------                                                               #
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
# | /image_color       | Simulator          | Color images from the         | #
# |                    |                    | vehicles traffic light camera | #
# +--------------------+--------------------+-------------------------------+ #
# | /vehicle/          | Simulator          | Debug and test set gathering  | #
# |     traffic_lights |                    | ground truth values           | #
# +--------------------+--------------------+-------------------------------+ #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Outputs                                                                 | #
# +--------------------+---------------+------------------------------------+ #
# | Topic Path         | Update Rate + | Description                        | #
# |                    | Queue Size    |                                    | #
# +--------------------+---------------+------------------------------------+ #
# | /traffic_waypoint  | 5 Hz / 1      | Index of waypoint of a known red   | #
# |                    |               | traffic light, -1 otherwise        | #
# +--------------------+---------------+------------------------------------+ #
# | /traffic_light_  \ | ? Hz / 1      | Result of light detection          | #
# |detection_result_ \ |               |                                    | #
# |color               |               |                                    | #
# +--------------------+---------------+------------------------------------+ #
#                                                                             # 
#                                                                             #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 3/07/2018          | Henry Savage  | Added boilerplate documentation as | #
# |                    |               | well as cascading of ground truth  | #
# |                    |               | data so downstream nodes can do    | #
# |                    |               | development in parallel            | #
# +--------------------+---------------+------------------------------------+ #
# | 3/13/2018          | Henry Savage  | Updated publish rate to 5hz as the | #
# |                    |               | documentation from above suggests  | #
# |                    |               | (it was 1hz previously)            | #
# +--------------------+---------------+------------------------------------+ #
# | 3/15/2018          | Henry Savage  | Made the traffic_waypoint report   | #
# |                    |               | a negative version of the next     | #
# |                    |               | light as opposed to always -1      | #
# +--------------------+---------------+------------------------------------+ #
# | 3/18/2018          | Jason Clemons | Adding in a stream of the detector | #
# |                    |               | result so we can visualize as      | #
# |                    |               | vehicle is traveling               | #
# +--------------------+---------------+------------------------------------+ #
# | 3/21/2018          | Henry Savage  | Updated traffic light state report | #
# |                    |               | mechanism to just report -1 when a | #
# |                    |               | light is green or yellow (previous | #
# |                    |               | method was return -1*next_ind so   | #
# |                    |               | we could print the future light    | #
# |                    |               | status in the updater. This will   | #
# |                    |               | likely be moved to a debug node    | #
# |                    |               | this node can just send the status | #
# |                    |               | directly instead)                  | #
# +--------------------+---------------+------------------------------------+ #
# | 3/21/2018          | Jason  Clemons| Added param for model path         | #
# |                    |               | and also changed image types to    | #
# |                    |               | png                                | #
# +--------------------+---------------+------------------------------------+ #
# | 3/25/2018          | Jason  Clemons| Added debug log level              | #
# |                    |               |                                    | #
# |                    |               |                                    | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# ROS core
import rospy

# Data types
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

# Traffic light info is in a YAML file
import yaml

# Our Detector objects
from detector import SimDetector

class TLDetector(object):
    def __init__(self):
        '''
        '''

        # Init this as a ROS node
        rospy.init_node('tl_detector',log_level=rospy.DEBUG)

        # Load traffic light config data -- YAML file with:
        # camera_info - Might be the header of the YAML file?
        # image_width - [800 Default]
        # image_height - [600 Default]
        # stop_line_positions - [Array of known stop lines for lights]
        config_string = rospy.get_param("/traffic_light_config")
        lights_config = yaml.load(config_string)

        model_file_path = rospy.get_param("/traffic_light_detector_model") + '/frozen_inference_graph.pb'
        training_data_path = rospy.get_param("~traffic_light_training_data_directory")


        print("Model File Path:", model_file_path)

        print("Training Data Path:", training_data_path)

        # Main detector object
        self.detector = SimDetector(lights_config=lights_config, model_path=model_file_path, save_path=training_data_path)

        # Data we're subscribing to
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # Data we're publishing
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.traffic_light_detection_result_color = rospy.Publisher('/traffic_light_detection_result_color', Image, queue_size=1)


        # Run the control loop
        self.run()

    def run(self):
        '''
        The main control loop for the traffic light detection node
        '''

        # Set the refresh rate in Hz
        rate = rospy.Rate(5)

        # Use the start time sentinel to know when to start processing
        start_time = 0

        # Poll until we're ready to start
        while not start_time:
            start_time = rospy.Time.now().to_sec()

        # Control loop
        while not rospy.is_shutdown():

            # Get current detector info
            light_ind, light_status = self.detector.traffic_light_status()

            # If its not a red light, we dont have a stopping waypoint
            if(light_status != TrafficLight.RED):
                light_ind = -1

            # Publish the traffic light status
            # light_ind will be where we want to stop. If its a -1 then
            # We don't need to stop
            self.upcoming_red_light_pub.publish(Int32(light_ind))

            if (self.detector.detector_result_image_msg != None):
                #Publish the marked up image
                self.traffic_light_detection_result_color.publish(self.detector.detector_result_image_msg)

            # Sleep
            rate.sleep()

    def pose_cb(self, msg):
        '''
        Callback for current_pose data
        '''
        self.detector.set_vehicle_pose(msg.pose)

    def waypoints_cb(self, waypoints):
        '''
        Callback for the base_waypoints list data
        '''
        self.detector.set_waypoints(waypoints.waypoints)

    def image_cb(self, image):
        '''
        Callback for handling image data
        '''
        self.detector.set_image(image)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
