#!/usr/bin/env python3

"""
Created on Tue Jun 8 2021

@author: Oshada Jayasinghe
"""

import numpy as np
import torchvision.transforms as transforms
import torch
import time
import cv2
import rospy

from model import lane_net
from scipy.stats import pearsonr

from sensor_msgs.msg import Image as SensorImage
from lane_detector.msg import Lane_msg


no_of_row_anchors = 18

row_anchor_locations = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

no_of_gridding_cells = 100

net = lane_net(cls_dim = (no_of_gridding_cells+1, no_of_row_anchors, 4)).cuda()

state_dict = torch.load("/media/fyp/sdCard/last_year/swiftlane/lane_detector.pth", map_location = 'cpu')['model']
classifier = torch.load("/home/ace/detector_ws/src/lane_detector/scripts/lane_detector/lane_class.pth")
net.load_state_dict(state_dict)
net.eval()
classifier.eval()

lanes_publisher = rospy.Publisher('/lane_detections',Lane_msg, queue_size = 1)

frame_count = 0
total_fps = 0

# warming up GPU
for i in range(10):
    net(torch.rand((1, 3, 288, 800)).cuda())  

print("Started Lane Detector")

def four_point_transform(image, pts):

	(tl, tr, br, bl) = pts
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def detect_lanes(frame, frame_height, frame_width):

    t1 = time.time()

    no_of_lanes = 0
    lane_coordinates = []
    lane_classes = []

    original_frame = frame.copy()

    frame = frame[450:,:]
    frame = cv2.resize(frame, (800, 288))

    frame = torch.from_numpy(frame).cuda()
    frame = frame.permute(2, 0, 1)/255.0
    frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame)
    frame = torch.reshape(frame,(1, 3, 288, 800))

    out = net(frame)[0].data.cpu().numpy()

    out = out[:, ::-1, :]
    out = np.argmax(out, axis = 0)
    out[out == no_of_gridding_cells] = -1
    out = out + 1

# Lane classes 
# 0 - Dashed line
# 1 - Single line

    for l in range(4):
        lane = out[:,l]
        x = np.nonzero(lane)[0]             
        lane = lane[lane != 0]
        if len(lane) >= 6:   # filtering based on minimum lane points threshold
            coeff, pvalue = pearsonr(lane,x)
            if abs(coeff) >= 0.95:   # filtering based on the Perason correlation coefficient threshold
                poly = np.poly1d(np.polyfit(x,lane,1))
                y = poly(x)
                y[y < 1] = 1
                y[y > no_of_gridding_cells] = no_of_gridding_cells
                start_lane_point = [int(y[0] * frame_width / no_of_gridding_cells), int((frame_height - 450) * (row_anchor_locations[no_of_row_anchors - 1 - x[0]] / 288)) - 1 + 450]
                end_lane_point = [int(y[-1] * frame_width / no_of_gridding_cells), int((frame_height - 450) * (row_anchor_locations[no_of_row_anchors - 1 - x[-1]] / 288)) - 1 + 450]
                lane_coordinates.extend(start_lane_point)
                lane_coordinates.extend(end_lane_point)
                no_of_lanes += 1
    
    lane_coordinates_classification = np.reshape(lane_coordinates, (int(len(lane_coordinates) / 4), 4))
    if no_of_lanes != 0:
        pts = []
        for j in range(len(lane_coordinates_classification)):
            x1, y1, x2, y2 = lane_coordinates_classification[j]
            m = (y2 - y1)/(x2 - x1)
            c = y1 - m*x1
            # for y in range(y2, y1, 20):
            #     x = int((y-c)/m)
            #     img = cv2.circle(img, (x,y), 8, color_map["green"], -1)

            ymin = int(y2+(y1-y2)*18/20) # Points closer to me / bottom
            ymax = int(y2+(y1-y2)*2/20) # Points distant to me / top

            x_1 = int((ymin-c)/m)-70 # bottom left
            x_2 = int((ymin-c)/m)+70 # bottom right
            x_3 = int((ymax-c)/m)-20 # top left
            x_4 = int((ymax-c)/m)+20 # top right

            arg_coords = [(x_3, ymax), (x_4, ymax),(x_2, ymin),(x_1, ymin)]
            pts.append(np.array(arg_coords, dtype = "float32"))

        for i in range(len(pts)):
            warped = four_point_transform(original_frame, pts[i])
            warped = cv2.resize(warped,(50, 400)).astype(np.float32)
            warped = torch.from_numpy(warped).cuda()
            warped = warped.permute(2,0,1)/255.0

            if i == 0:
                crop = torch.unsqueeze(warped,0)
            else:
                crop = torch.cat((crop,torch.unsqueeze(warped,0)))

        out = torch.nn.Softmax(dim = 1)(classifier(crop))
        class_ids = torch.argmax(out, axis = 1) # 0 - Dashed 1 - Single
        lane_classes = class_ids.tolist()
        # print(class_ids)

    return no_of_lanes, lane_coordinates, lane_classes




def callback(data):

    global frame_count, total_fps
    
    t1 = time.time()
    
    frame_height = data.height
    frame_width = data.width
    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)
    
    no_of_lanes, lane_coordinates, lane_classes = detect_lanes(frame, frame_height, frame_width)
    
    t2 = time.time()

    lanes = Lanes()

    lanes.header.stamp = rospy.Time.now()
    lanes.header.seq = data.header.seq
    lanes.frame_height = frame_height
    lanes.frame_width = frame_width
    lanes.inference_time = t2 - t1
    lanes.no_of_lanes = int(no_of_lanes)
    lanes.lane_coordinates = np.array(lane_coordinates, dtype = np.int32)
    lanes.class_ids = np.array(lane_classes, dtype = np.int32)

    lanes_publisher.publish(lanes)    

    fps = 1 / (t2 - t1)
    frame_count = frame_count + 1
    total_fps = total_fps + fps

    print("Lane Detection Average FPS :", total_fps / frame_count)

def lane_detector():
    rospy.loginfo("Lane detector initiated...")
    rospy.init_node('lane_detector', anonymous = True)
    rospy.Subscriber('/input_frame', SensorImage, callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        lane_detector()
    except rospy.ROSInterruptException:
        pass