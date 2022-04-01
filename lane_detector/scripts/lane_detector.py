#!/usr/bin/env python3

import imghdr
import torch
import cv2
from torchvision import transforms
from PIL import Image
from model.model import parsingNet
import numpy as np
import scipy.special
from scipy.stats import pearsonr
import math

from tracking import Sort

import rospy
from sensor_msgs.msg import Image as SensorImage

lane_detector_publisher = rospy.Publisher('/lane_detector_output', SensorImage , queue_size = 1)





model_path = '/media/fyp/sdCard/lane_detector/models/culane_18.pth'

row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

cls_num_per_lane = 18

img_w, img_h = 1640,590



class lane_detector:
    def __init__(self) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.tracker = Sort(max_age=7, min_hits=10, use_dlib = False)

        self.net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,cls_num_per_lane,4), use_aux=False).cuda()

        self.state_dict = torch.load(model_path, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in self.state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()

    def polar2cart(self, rho, theta):
        a = -math.tan(theta)
        b = rho* math.sqrt(1 + a**2)
        return a,b

    def cart2polar(self,a, b):
        rho = b/math.sqrt(1+a**2)
        theta = math.atan(-a)
        return rho,theta

    def draw_lines(self, polar_list, image):

        for polar in polar_list:
            a,b = self.polar2cart(polar[0], polar[1])
            y = np.array([295,590])
            x = (y - b)/a
            
            image = cv2.line(image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)
            
        return image
        
    def inference(self,frame):
        frame = cv2.resize(frame, (1920, 1080))
        frame = frame[450:,:]
        im_pil = Image.fromarray(frame.copy())
        imgs = img_transforms(im_pil)
        
        imgs = imgs.cuda()
        imgs = imgs[None, :]
        with torch.no_grad():
            out = self.net(imgs)
            
        col_sample = np.linspace(0, 800 - 1, 200)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(200) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == 200] = 0
        out_j = loc
        frame = cv2.resize(frame,(1640,590))
        vis = frame
        all_lanes = []
        polar_coords = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                single_lane = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        ppp2 = (out_j[k, i] * col_sample_w * img_w / 800 - 1, img_h * (row_anchor[cls_num_per_lane-1-k]/288) - 1 )
                        single_lane.append(list(ppp2))
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
                all_lanes.append(single_lane)

                
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
        polar_coords = np.array(polar_coords)

        tracked_coords = self.tracker.update(polar_coords)
        out_vis = self.draw_lines(tracked_coords,vis)
        # cv2.imshow("preview", out_vis)
        # cv2.waitKey(0)
        return out_vis


def lane_detector_callback(data):

    frame_height = data.height
    frame_width = data.width    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    frame = lane_detector.inference(frame)

    out_frame = SensorImage() # done
    out_frame.header.stamp = rospy.Time.now() # done
    out_frame.height = frame.shape[0]         # done
    out_frame.width = frame.shape[1]          # done
    out_frame.encoding = "rgb8"               # done
    out_frame.is_bigendian = False            # done
    out_frame.step = 3 * frame.shape[1]       # done
    out_frame.data = frame.tobytes()

    lane_detector_publisher.publish(out_frame)



def lane_detector_func():
    rospy.loginfo("Lane detector initiated...")
    rospy.init_node('lane_detector', anonymous = True)
    rospy.Subscriber('/wide_camera_frame', SensorImage, lane_detector_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        lane_detector = lane_detector()
        lane_detector_func()
    except rospy.ROSInterruptException:
        pass
