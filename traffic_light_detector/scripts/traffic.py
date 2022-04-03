#!/usr/bin/env python3

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""
import sys

from flask import g
sys.path.append('/home/fyp2selfdriving/ROS/catkin_ws/src/traffic_light_detector/scripts')
print(sys.path)
import imp
from matplotlib import image
import numpy as np
import time
import cv2
import rospy
from inferencing import *

from sensor_msgs.msg import Image as SensorImage 
from camera_system.msg import img_pair_msg
from traffic_light_detector.msg import bbox_msg, bbox_array_msg

traffic_light_publisher = rospy.Publisher('/traffic_light_output', SensorImage , queue_size = 1)
traffic_light_annotation_publisher = rospy.Publisher('/traffic_light_annotation', bbox_array_msg, queue_size=1)
raw_wide_img_publisher = rospy.Publisher('/raw_wide_img', SensorImage, queue_size=1)

img_height = 1080
img_width = 1920
old_time = time.time()

print("Started traffic light Detector")

def dual_frame_callback(data):
    global old_time

    frame_height = data.im_narrow.height
    frame_width = data.im_narrow.width
    
    narrow_frame = np.frombuffer(data.im_narrow.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    wide_frame = np.frombuffer(data.im_wide.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    if (mobile_app_enable):
        frame, annotations, wide_raw_img = inference2.inference2(narrow_frame,wide_frame)
    else:
        frame = inference2.inference2(narrow_frame, wide_frame)
    
    out_frame = SensorImage() 
    out_frame.header.stamp = rospy.Time.now() 
    out_frame.height = img_height       
    out_frame.width = img_width          
    out_frame.encoding = "rgb8"               
    out_frame.is_bigendian = False            
    out_frame.step = 3 * img_width   
    out_frame.data = frame.tobytes()

    traffic_light_publisher.publish(out_frame)   

    if (mobile_app_enable):

        annotations_array = bbox_array_msg()
        annotations_array.box_count = len(annotations)
        for annotation in annotations:
            bbox_data = bbox_msg()
            bbox_data.type = annotation['type']
            bbox_data.xmin = int(annotation['xmin'])
            bbox_data.ymin = int(annotation['ymin'])
            bbox_data.xmax = int(annotation['xmax'])
            bbox_data.ymax = int(annotation['ymax'])
            annotations_array.bbox_array.append(bbox_data)

        traffic_light_annotation_publisher.publish(annotations_array)

         ## send images only if there are annotations and after time limit from last publish
        if (len(annotations)>5 and (time.time()-old_time > 1)):
            raw_wide_frame = SensorImage() 
            raw_wide_frame.header.stamp = rospy.Time.now() 
            raw_wide_frame.height = img_height         
            raw_wide_frame.width = img_width        
            raw_wide_frame.encoding = "rgb8"               
            raw_wide_frame.is_bigendian = False            
            raw_wide_frame.step = 3 * img_width     
            raw_wide_frame.data = wide_raw_img.tobytes()

            raw_wide_img_publisher.publish(raw_wide_frame)

            old_time = time.time()



def single_frame_callback(data):
    
    frame_height = data.height
    frame_width = data.width
    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame = inference.inference(frame)
    
    out_frame = SensorImage() 
    out_frame.header.stamp = rospy.Time.now() 
    out_frame.height = img_height        
    out_frame.width = img_width          
    out_frame.encoding = "rgb8"               
    out_frame.is_bigendian = False            
    out_frame.step = 3 * img_width     
    out_frame.data = frame.tobytes()

    traffic_light_publisher.publish(out_frame)


def traffic_light_detector():
    global mobile_app_enable
    
    rospy.loginfo("Traffic light detector initiated...")
    rospy.init_node('traffic', anonymous = True)


    if (cam_count == 1):
        rospy.Subscriber('/single_input_frame', SensorImage, single_frame_callback)
    elif(cam_count ==2):
        rospy.Subscriber('/dual_input_frames', img_pair_msg, dual_frame_callback)
    else:
        print("error")
    rospy.spin()

if __name__ == '__main__':
    try:
        cam_count = int(rospy.get_param("cam_count"))
        mobile_app_enable = bool(rospy.get_param("mobile_app_enable"))
        inference = inference()
        inference2 = inference2(use_tracker=False, mobile_app_enable=mobile_app_enable)
        traffic_light_detector()
    except rospy.ROSInterruptException:
        pass
