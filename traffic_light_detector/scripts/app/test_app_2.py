#!/usr/bin/env python3

from flask import Flask, jsonify
import rospy
from traffic_light_detector.msg import bbox_msg, bbox_array_msg, annotation_app_msg
from sensor_msgs.msg import Image as SensorImage 
import json
import urllib3
import numpy as np

app = Flask(__name__)
http = urllib3.PoolManager()

def annotation_app_data_user(data):
    bboxes = data.bbox_array
    frame_height = data.img_narrow.height
    frame_width = data.img_narrow.width
    
    frame_binary = data.img_narrow.data
    frame = np.frombuffer(data.img_narrow.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    print("app received the frame", frame.shape)


def annotation_annotation_user(data):
    # box_count = data.box_count
    bboxes = data.bbox_array
    
    # label_sample = []
    for bbox in bboxes:
        annotation ={
        'type':str(bbox.type),
        "xmin":str(bbox.xmin),
        "ymin":str(bbox.ymin),
        "xmax":str(bbox.xmax),
        "ymax":str(bbox.ymax),
        "frameid":"0"
        # 'type':"100",
        # "xmin":"100",
        # "ymin":"100",
        # "xmax":"100",
        # "ymax":"100",
        # "frameid":"0"
        }
        json_annotations = json.dumps(annotation,indent=2)
        url="https://sample-node-phase1.herokuapp.com/trafficlight"
        res = http.request('POST', url, headers={'Content-Type': 'application/json'},body=json_annotations)
        print(res.status)  
        print(res.data)

    # label_sample = []
    # for bbox in bboxes:
    #     annotation = {
    #         'type':bbox.type,
    #         'cordinates':{
    #             "xmin":bbox.xmin,
    #             "ymin":bbox.ymin,
    #             "xmax":bbox.xmax,
    #             "ymax":bbox.ymax
    #         }
    #     }
    #     label_sample.append(annotation)
    
    # json_annotations = {"label_sample":label_sample}
    # json_annotations = json.dumps(json_annotations,indent=2)
    # http = urllib3.PoolManager()
    # url="https://sample-node-phase1.herokuapp.com/"
    # res = http.request('POST', url, headers={'Content-Type': 'application/json'},body=json_annotations)
    # print(res.status)  
    # print(res.data)

   

    # with open('/home/fyp/catkin_ws/src/traffic_light_detector/json_data.json','w') as json_file:
    #         json_file.write(json_annotations)

    
def mobile_app():
    rospy.loginfo("mobile app initiated...")
    rospy.init_node('mobile_app', anonymous = True)
    rospy.Subscriber('/traffic_light_annotation', bbox_array_msg, annotation_annotation_user)
    rospy.Subscriber('/annotation_app_data', annotation_app_msg, annotation_app_data_user )
    rospy.spin()


    
if __name__ == '__main__':
    try:
        mobile_app()
    except rospy.ROSInterruptException:
        pass