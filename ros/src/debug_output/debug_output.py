#!/usr/bin/env python

'''
###############################################################################
# debug_output.py                                                             #
# --------------------------------------------------------------------------- #
#                                                                             #
# Description:                                                                #
# ------------                                                                #
# This module contains the source for the debug terminal outputter that will  #
# listen for special new topics that the system publishes and update the      #
# terminal for a holistic real time view of all the debug items we want to    #
# monitor.                                                                    #
#                                                                             #
# +-------------------------------------------------------------------------+ #
# | Inputs                                                                  | #
# +----------------------+----------------+---------------------------------+ #
# | Topic Path           | Source         | Description                     | #
# +----------------------+----------------+---------------------------------+ #
# | /debug               | Any ROS node   | Contains a request to add text  | #
# |                      |                | to the debug console at a       | #
# |                      |                | given line staring location     | #
# +----------------------+----------------+---------------------------------+ #
#                                                                             #
# Change Log:                                                                 #
# -----------                                                                 #
# +--------------------+---------------+------------------------------------+ #
# | Date               | Author        | Description                        | #
# +--------------------+---------------+------------------------------------+ #
# | 3/25/2018          | Henry Savage  | Initial pass on the code           | #
# +--------------------+---------------+------------------------------------+ #
# | 3/27/2018          | Henry Savage  | Updated topic documentation        | #
# +--------------------+---------------+------------------------------------+ #
###############################################################################
'''

# ROS Code libraries
import rospy

# Debug message type so other nodes can request output to screen
from debug_msgs.msg import Debug

#-----------------------------------------------------------------------------#
# Debug Output Node                                                           #
#-----------------------------------------------------------------------------#

class DebugOutput(object):
    def __init__(self):

        # Initialize our node
        rospy.init_node('debug_output')

        # Attributes
        self.enabled = rospy.get_param("~debug_enabled", False)
        print("DEBUG ENABLED? '" + str(self.enabled) + "'")
        self.length = 120

        # Maintain frame object
        self.lines = ["" for i in range(0, self.length)]

        # Subscribe to all the debug channels we want to get data from
        rospy.Subscriber('/debug', Debug, self.debug_msg_handler)

        # Run
        self.run()

    def debug_msg_handler(self, msg):
        '''
        Handle an incoming debug message value and save it to the frame
        '''

        l_start = msg.line_start
        s_raw = msg.text.strip("\n")
        s_set = s_raw.split("\n")

        if(l_start >= len(self.lines)):
            return

        i = l_start
        for s in s_set:
            if(i >= len(self.lines)):
                break
            self.lines[i] = s
            i += 1

    def __render_to_terminal(self, frame):
        '''
        Render the text frame to the terminal
        '''
        s = "\033[?25l" + "\033[1J" + "\033[1;1H" + str(frame) + "\033[?25h"
        print(s)

    def update(self):
        '''
        Update our terminal with the current debug values
        '''
        frame_str = '\n'.join([line for line in self.lines]).strip("\n")
        self.__render_to_terminal(frame_str)

    def run(self):

        if(not self.enabled):
            return

        # Set the refresh rate
        rate = rospy.Rate(5)

        # Use the start time sentinel to know when to start processing
        start_time = 0

        # Poll until we're ready to start
        while not start_time:
            start_time = rospy.Time.now().to_sec()

        # Control loop
        while not rospy.is_shutdown():

            # Render debug info
            if(self.enabled):
                self.update()

            # Sleep
            rate.sleep()

#-----------------------------------------------------------------------------#
# Run                                                                         #
#-----------------------------------------------------------------------------#

if __name__ == '__main__':
    try:
        DebugOutput()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start debug output node.')
