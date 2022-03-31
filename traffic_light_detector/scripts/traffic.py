#!/usr/bin/env python3

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""
import sys
sys.path.append('/home/fyp2selfdriving/ROS/catkin_ws/src/traffic_light_detector/scripts')
print(sys.path)
import imp
from matplotlib import image
import numpy as np
import time
import cv2
import rospy
from inferencing import *

from sensor_msgs.msg import Image as SensorImage # done
from camera_system.msg import img_pair_msg
from traffic_light_detector.msg import bbox_msg, bbox_array_msg

traffic_light_publisher = rospy.Publisher('/traffic_light_output', SensorImage , queue_size = 1)
traffic_light_annotation_publisher = rospy.Publisher('/traffic_light_annotation', bbox_array_msg, queue_size=1)

print("Started traffic light Detector")

def dual_frame_callback(data):
    global frame_count, total_fps

    t1 = time.time()


    frame_height = data.im_narrow.height
    frame_width = data.im_narrow.width
    
    narrow_frame = np.frombuffer(data.im_narrow.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    wide_frame = np.frombuffer(data.im_wide.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame, annotations = inference2.inference2(narrow_frame,wide_frame)
    
    out_frame = SensorImage() # done
    out_frame.header.stamp = rospy.Time.now() # done
    out_frame.height = frame.shape[0]         # done
    out_frame.width = frame.shape[1]          # done
    out_frame.encoding = "rgb8"               # done
    out_frame.is_bigendian = False            # done
    out_frame.step = 3 * frame.shape[1]       # done
    out_frame.data = frame.tobytes()

    traffic_light_publisher.publish(out_frame)   

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


    # json_annotations = json.dumps(annotations,indent=2)
    traffic_light_annotation_publisher.publish(annotations_array)
    # with open('/home/fyp/catkin_ws/src/traffic_light_detector/json_data.json','w') as json_file:
        #     json_file.write(json_annotations)


def single_frame_callback(data):

    global frame_count, total_fps
    
    t1 = time.time()
    
    frame_height = data.height
    frame_width = data.width
    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame = inference.inference(frame)
    # cv2.imshow(frame, "aa")
    # cv2.waitKey(0)
    
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
    cam_count = int(rospy.get_param("cam_count"))
    if (cam_count == 1):
        rospy.Subscriber('/single_input_frame', SensorImage, single_frame_callback)
    elif(cam_count ==2):
        rospy.Subscriber('/dual_input_frames', img_pair_msg, dual_frame_callback)
    else:
        print("error")
    rospy.spin()

if __name__ == '__main__':
    try:
        inference = inference()
        inference2 = inference2(use_tracker=True)
        traffic_light_detector()
    except rospy.ROSInterruptException:
        pass
