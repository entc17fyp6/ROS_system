#!/usr/bin/env python3

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""

from matplotlib import image
import numpy as np
import time
import cv2
import rospy
from inference import *

from sensor_msgs.msg import Image as SensorImage # done

traffic_light_publisher = rospy.Publisher('/traffic_light_output', SensorImage , queue_size = 1)

print("Started traffic light Detector")


def callback(data):

    global frame_count, total_fps
    
    t1 = time.time()
    
    frame_height = data.height
    frame_width = data.width
    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame = inference.inference(frame)
    # cv2.imshow(frame, "aa")
    # cv2.waitKey(0)
    
    t2 = time.time()

    out_frame = SensorImage() # done
    out_frame.header.stamp = rospy.Time.now() # done
    out_frame.height = frame.shape[0]         # done
    out_frame.width = frame.shape[1]          # done
    out_frame.encoding = "rgb8"               # done
    out_frame.is_bigendian = False            # done
    out_frame.step = 3 * frame.shape[1]       # done
    out_frame.data = frame.tobytes()

    traffic_light_publisher.publish(out_frame)    

def traffic_light_detector():
    rospy.loginfo("Traffic light detector initiated...")
    rospy.init_node('traffic', anonymous = True)
    rospy.Subscriber('/input_frame', SensorImage, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        inference = inference()
        traffic_light_detector()
    except rospy.ROSInterruptException:
        pass
