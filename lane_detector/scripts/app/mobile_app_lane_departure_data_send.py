#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import json
import subprocess


def mobile_app_data_send(data):
    global count
    beta = data.data
    
    usb_data = {
        "json_id": count,
        "beta": beta
    }
    usb_data = json.dumps(usb_data,indent=2)
    with open('/home/fyp/catkin_ws/src/lane_detector/scripts/app/lane_departure_data.json','w') as json_file:
            json_file.write(usb_data)
    subprocess.call('adb push /home/fyp/catkin_ws/src/lane_detector/scripts/app/lane_departure_data.json /sdcard/Download/fyp',shell = True,  stdout=subprocess.DEVNULL)
    count+=1

    
def mobile_app_lane_departure_warning_system():
    rospy.loginfo("mobile app lane departure warning system initiated...")
    rospy.init_node('mobile_app_lane_departure', anonymous = True)
    rospy.Subscriber('/lane_departure_warning', Float32, mobile_app_data_send)
    rospy.spin()


    
if __name__ == '__main__':
    try:
        global count
        count = 0
        mobile_app_lane_departure_warning_system()
    except rospy.ROSInterruptException:
        pass