#!/usr/bin/env python3

from cv2 import COLOR_BGR2RGB
from flask import Flask, jsonify
import rospy
from traffic_light_detector.msg import bbox_msg, bbox_array_msg, annotation_app_msg
from sensor_msgs.msg import Image as SensorImage 
import json
import urllib3
import numpy as np
import requests
import cv2
import subprocess

app = Flask(__name__)
http = urllib3.PoolManager()

# def web_mobile_app_data_send(data):
#     global web_mobile_app_enable

#     if (web_mobile_app_enable):

#         bboxes = data.bbox_array
#         label_sample = []
#         for bbox in bboxes:
#             annotation = {
#                 'type':bbox.type,
#                 'cordinates':{
#                     "xmin":bbox.xmin,
#                     "ymin":bbox.ymin,
#                     "xmax":bbox.xmax,
#                     "ymax":bbox.ymax
#                 }
#             }
#             label_sample.append(annotation)
        
#         json_annotations = {"label_sample":label_sample}
#         json_annotations = json.dumps(json_annotations,indent=2)
#         http = urllib3.PoolManager()
#         url="https://sample-node-phase1.herokuapp.com/"
#         res = http.request('POST', url, headers={'Content-Type': 'application/json'},body=json_annotations)
#         # print(res.status)  
#         # print(res.data)
#         print("web mobile app data set successfully...!")

def usb_mobile_app_data_send(data):
    global count, usb_mobile_app_enable
    if (usb_mobile_app_enable):

        bboxes = data.bbox_array
        bbox_ids = list(data.bbox_ids)
        usb_data = {
            "json_id": count,
            "count": len(bboxes),
            "ids": bbox_ids
        }
        
        usb_data = json.dumps(usb_data,indent=2)
        with open('/home/fyp/catkin_ws/src/traffic_light_detector/scripts/app/traffic_light_data.json','w') as json_file:
                json_file.write(usb_data)
        subprocess.call('adb push /home/fyp/catkin_ws/src/traffic_light_detector/scripts/app/traffic_light_data.json /sdcard/Download/fyp',shell = True,  stdout=subprocess.DEVNULL)
        count+=1

    
def mobile_app_traffic_light_detection():
    global web_mobile_app_enable, usb_mobile_app_enable
    rospy.loginfo("mobile app initiated...")
    rospy.init_node('mobile_app_traffic_light_detection', anonymous = True)
    # web_mobile_app_enable = bool(rospy.get_param("web_mobile_app_enable"))
    usb_mobile_app_enable = bool(rospy.get_param("usb_mobile_app_enable"))
    # rospy.Subscriber('/traffic_light_annotation', bbox_array_msg, web_mobile_app_data_send)
    rospy.Subscriber('/traffic_light_annotation', bbox_array_msg, usb_mobile_app_data_send)
    rospy.spin()


    
if __name__ == '__main__':
    try:
        global count, web_mobile_app_enable, usb_mobile_app_enable
        count = 0
        mobile_app_traffic_light_detection()
    except rospy.ROSInterruptException:
        pass