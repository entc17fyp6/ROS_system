#! /usr/bin/env python3.6

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""

import numpy as np
import cv2
import time
import rospy

from sensor_msgs.msg import Image as SensorImage

output_frame_publisher = rospy.Publisher('/output_frame',SensorImage,queue_size=1)

frame_count = 0

frame = None


if rospy.get_param("save_output") == True:
    output_video = cv2.VideoWriter(rospy.get_param("output_video"), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920,1080))



def image_callback(data):

    global frame, frame_count

    if frame is not None:


        output_frame = SensorImage()
        output_frame.header.stamp = rospy.Time.now()
        output_frame.height = frame.shape[0]
        output_frame.width = frame.shape[1]
        output_frame.encoding = "rgb8"
        output_frame.is_bigendian = False
        output_frame.step = 3 * frame.shape[1]
        output_frame.data = np.array(frame).tobytes()

        output_frame_publisher.publish(output_frame)

        # saving output video if save_output is true
        if rospy.get_param("save_output")==True:
            output_video.write(cv2.cvtColor(cv2.resize(frame, (1920,1080)), cv2.COLOR_RGB2BGR))

    frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    frame_count += 1

def visualizer():
    rospy.loginfo("Visualizer initiated...")
    rospy.init_node('visualizer',anonymous=True)
    rospy.Subscriber('/input_frame',SensorImage, image_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        visualizer()
    except rospy.ROSInterruptException:
        pass