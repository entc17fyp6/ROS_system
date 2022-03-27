#! /usr/bin/env python3.6
"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""

import rospy
import numpy as np
import cv2 
import argparse

from time import sleep

from sensor_msgs.msg import Image as SensorImage

def image_feeder(input_video):
    input_frame_publisher = rospy.Publisher('/input_frame',SensorImage,queue_size=1)
    rospy.init_node('image_feeder_node', anonymous=True)

    cap = cv2.VideoCapture(input_video)

    #setting fps to 5 if saving uotput video else 30
    # if rospy.get_param("save_output")==True:
    #     frame_rate = 5
    # else:
    #     frame_rate = 30
    frame_rate = 30
    rate = rospy.Rate(frame_rate)

    while not rospy.is_shutdown() and cap.isOpened():

        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1920,1080))

        input_frame = SensorImage()

        input_frame.header.stamp = rospy.Time.now()
        input_frame.height = frame.shape[0]
        input_frame.width = frame.shape[1]
        input_frame.encoding = "rgb8"
        input_frame.is_bigendian = False
        input_frame.step = 3* frame.shape[1]
        input_frame.data = frame.tobytes()

        input_frame_publisher.publish(input_frame)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    try:
        image_feeder(rospy.get_param("input_video"))
    except rospy.ROSInternalException:
        pass
