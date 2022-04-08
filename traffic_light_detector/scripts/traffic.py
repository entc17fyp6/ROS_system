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
from traffic_light_detector.msg import bbox_msg, bbox_array_msg, annotation_app_msg

traffic_light_publisher = rospy.Publisher('/traffic_light_output', SensorImage , queue_size = 1)
traffic_light_annotation_publisher = rospy.Publisher('/traffic_light_annotation', bbox_array_msg, queue_size=1)
annotation_app_publisher = rospy.Publisher('/annotation_app_data', annotation_app_msg, queue_size=1)

img_height = 1080
img_width = 1920

print("Started traffic light Detector")

def dual_frame_callback(data):

    frame_height = data.im_narrow.height
    frame_width = data.im_narrow.width
    
    narrow_frame = np.frombuffer(data.im_narrow.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    wide_frame = np.frombuffer(data.im_wide.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    t1 = time.time()
    inference_output = inference2.inference2(narrow_frame,wide_frame) #{"output_img", "all_annotations","narrow_raw_img", "narrow_annotations"}
    t2 = time.time()
    # print("dual inference time",(t2-t1)*1000)

    frame = inference_output["output_img"]
    
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
        all_annotations = inference_output["all_annotations"]
        if (len(all_annotations)):
            annotations_array = bbox_array_msg()
            annotations_array.box_count = len(all_annotations)
            for annotation in all_annotations:
                bbox_data = bbox_msg()
                bbox_data.type = annotation['type']
                bbox_data.xmin = int(annotation['xmin'])
                bbox_data.ymin = int(annotation['ymin'])
                bbox_data.xmax = int(annotation['xmax'])
                bbox_data.ymax = int(annotation['ymax'])
                annotations_array.bbox_array.append(bbox_data)

            traffic_light_annotation_publisher.publish(annotations_array)

    if (traffic_light_annotator_app_enable):
        narrow_annotations = inference_output["narrow_annotations"]
         ## send images only if there are all_annotations and after time limit from last publish
        if (not (narrow_annotations == None)):
            annotation_app_msg_obj = annotation_app_msg()

            annotation_app_msg_obj.img_narrow.header.stamp = rospy.Time.now() 
            annotation_app_msg_obj.img_narrow.height = img_height         
            annotation_app_msg_obj.img_narrow.width = img_width        
            annotation_app_msg_obj.img_narrow.encoding = "rgb8"               
            annotation_app_msg_obj.img_narrow.is_bigendian = False            
            annotation_app_msg_obj.img_narrow.step = 3 * img_width     
            annotation_app_msg_obj.img_narrow.data = narrow_frame.tobytes()

            annotation_app_msg_obj.box_count = len(narrow_annotations)

            for annotation in narrow_annotations:
                bbox_data = bbox_msg()
                bbox_data.type = annotation['type']
                bbox_data.xmin = int(annotation['xmin'])
                bbox_data.ymin = int(annotation['ymin'])
                bbox_data.xmax = int(annotation['xmax'])
                bbox_data.ymax = int(annotation['ymax'])
                annotation_app_msg_obj.bbox_array.append(bbox_data)

            annotation_app_publisher.publish(annotation_app_msg_obj)
            print("publishedd to annotator")


def single_frame_callback(data):
    
    frame_height = data.height
    frame_width = data.width
    
    t1 = time.time()
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    inference_output = inference.inference(frame.copy())
    t2 = time.time()
    print("single inference time",(t2-t1)*1000)


    output_frame = inference_output["output_img"]
    
    out_frame = SensorImage() 
    out_frame.header.stamp = rospy.Time.now() 
    out_frame.height = img_height        
    out_frame.width = img_width          
    out_frame.encoding = "rgb8"               
    out_frame.is_bigendian = False            
    out_frame.step = 3 * img_width     
    out_frame.data = output_frame.tobytes()

    traffic_light_publisher.publish(out_frame)


    if (mobile_app_enable or traffic_light_annotator_app_enable):
        all_annotations = inference_output["all_annotations"]
        if (all_annotations and len(all_annotations)):
            annotations_array = bbox_array_msg()
            annotations_array.box_count = len(all_annotations)
            for annotation in all_annotations:
                bbox_data = bbox_msg()
                bbox_data.type = annotation['type']
                bbox_data.xmin = int(annotation['xmin'])
                bbox_data.ymin = int(annotation['ymin'])
                bbox_data.xmax = int(annotation['xmax'])
                bbox_data.ymax = int(annotation['ymax'])
                annotations_array.bbox_array.append(bbox_data)

            if (mobile_app_enable):
                traffic_light_annotation_publisher.publish(annotations_array)

            if (traffic_light_annotator_app_enable and (len(all_annotations)>0) and inference_output["annotation_app_data_available"]):

                annotation_app_msg_obj = annotation_app_msg()

                annotation_app_msg_obj.img_narrow.header.stamp = rospy.Time.now() 
                annotation_app_msg_obj.img_narrow.height = img_height         
                annotation_app_msg_obj.img_narrow.width = img_width        
                annotation_app_msg_obj.img_narrow.encoding = "rgb8"               
                annotation_app_msg_obj.img_narrow.is_bigendian = False            
                annotation_app_msg_obj.img_narrow.step = 3 * img_width  
                annotation_app_msg_obj.img_narrow.data = frame.tobytes()

                annotation_app_msg_obj.box_count = len(all_annotations)
                annotation_app_msg_obj.bbox_array = annotations_array.bbox_array

                annotation_app_publisher.publish(annotation_app_msg_obj)


def traffic_light_detector():
    
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
        traffic_light_annotator_app_enable = bool(rospy.get_param("traffic_light_annotator_app_enable"))
        inference = inference(use_tracker = False,mobile_app_enable = mobile_app_enable,traffic_light_annotator_app_enable=traffic_light_annotator_app_enable)
        inference2 = inference2(use_tracker=True, mobile_app_enable=mobile_app_enable, traffic_light_annotator_app_enable=traffic_light_annotator_app_enable)
        traffic_light_detector()
    except rospy.ROSInterruptException:
        pass
