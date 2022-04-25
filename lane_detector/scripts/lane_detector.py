#!/usr/bin/env python3

import imghdr
from matplotlib.pyplot import colormaps
import torch
import cv2
from torchvision import transforms
from PIL import Image
from model.model import parsingNet2
import numpy as np
import scipy.special
from scipy.stats import pearsonr
import math

from tracking import Sort

import rospy
from sensor_msgs.msg import Image as SensorImage
from std_msgs.msg import Float32

from torch2trt import TRTModule

import time


use_tensorRT = True

lane_departure_warning_publisher = rospy.Publisher('/lane_departure_warning', Float32, queue_size=1)
lane_detector_publisher = rospy.Publisher('/lane_detector_output', SensorImage , queue_size = 1)

model_path = '/media/fyp/sdCard/detectors/lane_detector/models/ultrafast_Ceylane.pth'
trt_model_path = '/media/fyp/sdCard/detectors/lane_detector/models/ultrafast_Ceylane_trt.pth'

row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

cls_num_per_lane = 18
no_of_gridding_cells = 100
no_of_row_anchors = 18
img_w, img_h = 1640,590


color_map = {"red": (255,0,127),
             "cream": (255,102,178),
             "orange": (255,128,0), 
             "yellow": (255,255,0), 
             "green": (0,255,0), 
             "cyan": (0,255,255), 
             "blue": (0,128,255), 
             "purple": (178,102,255), 
             "pink": (255,0,255)}


class lane_detector:
    def __init__(self, use_tracker = False, use_tensorRT = True) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.use_tracker = use_tracker
        self.tracker = Sort(max_age=7, min_hits=10, use_dlib = False)
        self.lane_class_labels = ["unknown" ,"single white solid", "single white dotted", "single yellow solid" ,"double white solid" ,"double yellow solid" ,"White dotted solid","White solid dotted"]
        self.color_map = color_map
        self.use_tensorRT = use_tensorRT

        if (self.use_tensorRT):

            self.net = TRTModule()
            self.net.load_state_dict(torch.load(trt_model_path))
        else:
            self.net = parsingNet2(pretrained = False, backbone='18',cls_dim = (no_of_gridding_cells+1,no_of_row_anchors,4),
                        use_aux=False, class_nums=8).cuda() # we dont need auxiliary segmentation in testing

            self.state_dict = torch.load(model_path, map_location='cpu')['model']
            self.net.load_state_dict(self.state_dict)
            self.net.eval()

        self.seen = 0
        self.dt = 0.0

    def time_sync(self):
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


    def polar2cart(self, rho, theta):
        a = -math.tan(theta)
        b = rho* math.sqrt(1 + a**2)
        return a,b

    def cart2polar(self,a, b):
        rho = b/math.sqrt(1+a**2)
        theta = math.atan(-a)
        return rho,theta

    def draw_lines(self,polar_list, image):

        for polar in polar_list:

            a,b = self.polar2cart(polar[0], polar[1])
            y = np.array([350,590,450])
            x = (y - b)/a
            
            image = cv2.line(image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 0, 128), 2)
            image = self.draw_text(image, (int(x[2]),int(y[2])), str(int(polar[4])),color_map["cyan"] )
            
        return image

    def draw_text(self, img, label_point, text, color):  # for highlighted text

        font_scale = 0.7
        thickness = 4
        text_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(str(text), font, font_scale, thickness)
        text_location = (label_point[0] - 3, label_point[1] + text_size[1] - 41)

        cv2.rectangle(img, (text_location[0] - 2 // 2, text_location[1] - 2 - baseline),
                    (text_location[0] + text_size[0], text_location[1] + text_size[1]), color, -1)
        cv2.putText(img, str(text), (text_location[0], text_location[1] + baseline), font, font_scale, (0, 0, 0), text_thickness, 8)

        return img
        
    def inference(self,orig_frame):

        # t1 = self.time_sync()

        x_min, y_min = 245,240
        x_max, y_max = x_min+self.img_w, y_min+self.img_h
        orig_frame = orig_frame[y_min:y_max,x_min:x_max]
        # orig_frame = orig_frame[245:835,140:1780]
        frame = cv2.resize(orig_frame.copy(), (800, 288))

        frame = torch.from_numpy(frame).cuda()
        frame = frame.permute(2, 0, 1)/255.0
        frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame)
        frame = torch.reshape(frame,(1, 3, 288, 800))

        with torch.no_grad():
            out,classes = self.net(frame)

            
        col_sample = np.linspace(0, 800 - 1, no_of_gridding_cells)
        col_sample_w = col_sample[1] - col_sample[0]
        classes_cpu = classes[0].data.cpu().numpy()
        classes_cpu = np.argmax(classes_cpu, axis=0)

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(no_of_gridding_cells) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == no_of_gridding_cells] = 0
        
        # t2 = self.time_sync()
        # self.dt += t2 -t1
        # self.seen += 1

        # print("inference time", self.seen/self.dt)

        out_j = loc

        vis = orig_frame
        all_lanes = []
        polar_coords = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                single_lane = []
                for k in range(out_j.shape[0]):
                    if k == int(out_j.shape[0]/2):
                        x_cordinate = int(out_j[k, i] * col_sample_w * self.img_w / 800) - 1
                        y_cordinate = int(self.img_h * (row_anchor[no_of_row_anchors-1-k]/288)) - 1
                        ppp = (x_cordinate, y_cordinate)
                        vis = self.draw_text(vis, ppp, self.lane_class_labels[classes_cpu[i]], self.color_map["cream"])
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        ppp2 = (out_j[k, i] * col_sample_w * img_w / 800 - 1, img_h * (row_anchor[cls_num_per_lane-1-k]/288) - 1 )
                        single_lane.append(list(ppp2))
                        cv2.circle(vis,ppp,5,(255,69,0),-1)
                all_lanes.append(single_lane)
                
                if i == 0 or i == 3: ## track only ego lanes
                    continue

                if (self.use_tracker):                
                    if len(single_lane) >= 10:
                        nearest5p = single_lane[:10]
                    elif len(single_lane) > 5:
                        nearest5p = single_lane
                    else:
                        continue
                    nearest5p = np.array(nearest5p)
                    x = nearest5p[:,0]
                    y = nearest5p[:,1]
                    coeff, _ = pearsonr(x,y)
                    if abs(coeff) < 0.97:
                        continue
                    a,b = np.polyfit(x, y, 1)
                    polar = list(self.cart2polar(a, b))
                    polar_coords.append(polar)

        beta_out = None
        if (self.use_tracker):
            polar_coords = np.array(polar_coords)

            if (len(polar_coords)==2):
                beta = polar_coords[0][1]+polar_coords[1][1]
                if (abs(beta) > 0.7):
                    beta_out = beta
                    print("lane departure occured. ( beta =",beta,")")

            tracked_coords = self.tracker.update(polar_coords)
            vis = self.draw_lines(tracked_coords,vis)

        return vis,beta_out


def lane_detector_callback(data):

    frame_height = data.height
    frame_width = data.width    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame,beta = lane_detector.inference(frame)

    out_frame = SensorImage() # done
    out_frame.header.stamp = rospy.Time.now() # done
    out_frame.height = frame.shape[0]         # done
    out_frame.width = frame.shape[1]          # done
    out_frame.encoding = "rgb8"               # done
    out_frame.is_bigendian = False            # done
    out_frame.step = 3 * frame.shape[1]       # done
    out_frame.data = frame.tobytes()

    lane_detector_publisher.publish(out_frame)

    if (usb_mobile_app_enable and (beta != None) ):
        lane_departure_warning_publisher.publish(beta)



def lane_detector_func():
    rospy.loginfo("Lane detector initiated...")
    cam_count = int(rospy.get_param("cam_count"))
    rospy.init_node('lane_detector', anonymous = True)
    if (cam_count == 1):
        rospy.Subscriber('/single_input_frame', SensorImage, lane_detector_callback)
    elif(cam_count ==2):
        rospy.Subscriber('/wide_camera_frame', SensorImage, lane_detector_callback)
    else:
        print("error")

    rospy.spin()

if __name__ == '__main__':
    try:
        usb_mobile_app_enable = bool(rospy.get_param("usb_mobile_app_enable"))
        lane_detector = lane_detector(use_tracker=True, use_tensorRT = use_tensorRT)
        lane_detector_func()
    except rospy.ROSInterruptException:
        pass
