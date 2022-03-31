import numpy as np
from PIL import Image

import pathlib

import torch
import torchvision
import torchvision.transforms as transforms
from scipy.stats import pearsonr
import cv2
import time
# from single_camera_image_feeder import *

color_map = {"red": (255,0,127),
             "cream": (255,102,178),
             "orange": (255,128,0), 
             "yellow": (255,255,0), 
             "green": (0,255,0), 
             "cyan": (0,255,255), 
             "blue": (0,128,255), 
             "purple": (178,102,255), 
             "pink": (255,0,255)}

class backbone(torch.nn.Module):
    def __init__(self):
        super(backbone,self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class lane_net(torch.nn.Module):
    def __init__(self, size=(288, 800), cls_dim=(101, 18, 4)):
        super(lane_net, self).__init__()
        self.cls_dim = cls_dim # (no_of_gridding_cells, no_of_row_anchors, no_of_lanes)
        self.total_dim = np.prod(cls_dim)
        self.model = backbone()

        self.cls = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(2048, self.total_dim),
        )

        self.pool2 = torch.nn.Conv2d(256,8,1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.model(x)
        x = self.pool1(x)
        x = self.pool2(x).view(-1,1800)
        x = self.cls(x).view(-1, *self.cls_dim)

        return x

def draw_text(img, label_point, text, color):  # for highlighted text

    font_scale = 0.9
    thickness = 5
    text_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(str(text), font, font_scale, thickness)
    text_location = (label_point[0] - 3, label_point[1] + text_size[1] - 41)

    cv2.rectangle(img, (text_location[0] - 2 // 2, text_location[1] - 2 - baseline),
                  (text_location[0] + text_size[0], text_location[1] + text_size[1]), color, -1)
    cv2.putText(img, str(text), (text_location[0], text_location[1] + baseline), font, font_scale, (0, 0, 0), text_thickness, 8)

    return img

def detect_lanes(frame, frame_height, frame_width):

    t1 = time.time()

    no_of_lanes = 0
    lane_coordinates = []

    frame = frame[450:,:]
    frame = cv2.resize(frame, (800, 288))

    frame = torch.from_numpy(frame).cuda()
    frame = frame.permute(2, 0, 1)/255.0
    frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame)
    frame = torch.reshape(frame,(1, 3, 288, 800))

    t2 = time.time()
    out = net(frame)[0].data.cpu().numpy()
    t3 = time.time()
    print("model_speed =", 1/(t3-t2))
    print(out.shape)

    out = out[:, ::-1, :]
    out = np.argmax(out, axis = 0)
    out[out == no_of_gridding_cells] = -1
    out = out + 1

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
    
    return no_of_lanes, lane_coordinates


if __name__ == "__main__":


    no_of_row_anchors = 18

    row_anchor_locations = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

    no_of_gridding_cells = 100

    net = lane_net(cls_dim = (no_of_gridding_cells+1, no_of_row_anchors, 4)).cuda()

    state_dict = torch.load('/media/fyp/sdCard/last_year/swiftlane/lane_detector.pth')['model']   #swiftlane model path

    net.load_state_dict(state_dict)
    net.eval()
    
    # ['mannar_1.mp4', 'Colombo_1_1.mp4', 'Colombo_1_3.mp4', 'mannar_3.mp4', 
	# 'test_neluwa.mp4', 'katunayake_2.mp4', 'neluwa_2.mp4', 'Colombo_1_2.mp4', 
	# 'test_colombo.mp4', 'mannar_4.mp4', 'neluwa_1.mp4', 'katunayake_1.mp4', 
	# 'highway_1_1.mp4', 'Colombo_2_1.mp4', 'test_highway.mp4', 'Colombo_2_2.mp4',
	#  'highway_1_2.mp4', 'mannar_2.mp4']

    cap = cv2.VideoCapture('/media/fyp/sdCard/traffic_light_videos/27-02-2022_11-02_narrow_cam.mp4')
    # cap = cv2.VideoCapture('/home/ace/test_videos/Traffic_lights/Traffic_light1.MP4')

    frame_count = 0
    total_fps = 0
    total_vis_fps = 0


    while True:

        r, image = cap.read()

        image = cv2.resize(image, (1920, 1080))
        t1 = time.time()
        

        frame_height = 1080
        frame_width = 1920

        no_of_lanes, lane_coordinates = detect_lanes(image, frame_height, frame_width)

        lane_coordinates = np.reshape(lane_coordinates, (int(len(lane_coordinates) / 4), 4))

        # print(no_of_lanes)
        # print(lane_coordinates)

        t2 = time.time()

        # Visualization of the results of a detection.
        for j in range(len(lane_coordinates)):
            x1, y1, x2, y2 = lane_coordinates[j]
            m = (y2 - y1)/(x2 - x1)
            c = y1 - m*x1
            for y in range(y2, y1, 20):
                x = int((y-c)/m)
                image = cv2.circle(image, (x,y), 8, color_map["green"], -1)
        t3 = time.time()
        fps = 1 / (t2 - t1)
        # fps = 1/ (t3 - t1)
        visualize_fps = 1/(t3-t2)
        frame_count = frame_count + 1
        total_fps = total_fps + fps
        total_vis_fps = total_vis_fps + visualize_fps

        # print("Lane Detection Average FPS :", total_fps / frame_count, "\t visualize_fps =", total_vis_fps/frame_count)
        # print("Lane Detection Average FPS :", total_fps / frame_count)

        image = cv2.resize(image, (960, 540)) 
        cv2.imshow("preview", image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break