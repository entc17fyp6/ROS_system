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

def annotation_app_data_send(data):

    bboxes = data.bbox_array
    frame_height = data.img_narrow.height
    frame_width = data.img_narrow.width
    
    frame_binary = data.img_narrow.data
    frame = np.frombuffer(data.img_narrow.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    print("Image sending")
    url="https://sample-node-phase1.herokuapp.com/image"
    imencoded = cv2.imencode(".jpg", frame)[1]
    file = {'image': ('frame.jpg', imencoded.tostring(), 'image/jpeg')}
    response = requests.post(url, files=file)
    if (not response.ok):
    #     print("Upload completed successfully!")
    #     print(response.text)
    # else:
        print("Something went wrong!")
    
    label_sample = []
    for bbox in bboxes:
        annotation = {
            'class':bbox.type,
            'cordinates':{
                "xmin":bbox.xmin,
                "ymin":bbox.ymin,
                "xmax":bbox.xmax,
                "ymax":bbox.ymax
            }
        }
        label_sample.append(annotation)
    json_annotations = {"label_sample":label_sample}
    json_annotations = json.dumps(json_annotations,indent=2)

    http = urllib3.PoolManager()
    url="https://sample-node-phase1.herokuapp.com/trafficlight"
    res = http.request('POST', url, headers={'Content-Type': 'application/json'},body=json_annotations)
    print("annotation app data sent successfully...!")  
    print(res.data)


    
def annotation_app():
    rospy.loginfo("mobile app initiated...")
    rospy.init_node('annotation_web_app', anonymous = True)
    rospy.Subscriber('/annotation_app_data', annotation_app_msg,  annotation_app_data_send)
    rospy.spin()


    
if __name__ == '__main__':
    try:
        annotation_app()
    except rospy.ROSInterruptException:
        pass