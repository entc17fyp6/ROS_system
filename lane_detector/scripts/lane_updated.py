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


def polar2cart(rho, theta):
    a = -math.tan(theta)
    b = rho* math.sqrt(1 + a**2)
    return a,b

def cart2polar(a, b):
    rho = b/math.sqrt(1+a**2)
    theta = math.atan(-a)
    return rho,theta

def draw_lines(polar_list, image):

    for polar in polar_list:

        a,b = polar2cart(polar[0], polar[1])
        y = np.array([295,590,450])
        x = (y - b)/a
        
        image = cv2.line(image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 0, 128), 2)
        image = draw_text(image, (int(x[2]),int(y[2])), str(int(polar[4])),color_map["cyan"] )
        
    return image

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

color_map = {"red": (255,0,127),
             "cream": (255,102,178),
             "orange": (255,128,0), 
             "yellow": (255,255,0), 
             "green": (0,255,0), 
             "cyan": (0,255,255), 
             "blue": (0,128,255), 
             "purple": (178,102,255), 
             "pink": (255,0,255)}
row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
lane_class_labels = ["unknown" ,"single white solid", "single white dotted", "single yellow solid" ,"double white solid" ,"double yellow solid" ,"White dotted solid","White solid dotted"]

no_of_row_anchors = 18
no_of_gridding_cells = 100

net = parsingNet2(pretrained = False, backbone='18',cls_dim = (no_of_gridding_cells+1,no_of_row_anchors,4),
                use_aux=False, class_nums=8).cuda() # we dont need auxiliary segmentation in testing


state_dict = torch.load('/media/fyp/sdCard/lane_detector/models/ultrafast_Ceylane.pth')['model']   #swiftlane model path

net.load_state_dict(state_dict)
net.eval()


cap = cv2.VideoCapture('/media/fyp/sdCard/traffic_light_videos/27-02-2022_11-02_wide_cam.mp4')
img_w, img_h = 1640,590

tracker = Sort(max_age=7, min_hits=10, use_dlib = False)

while True:


    r, orig_frame = cap.read()
    if not r:
        break
    orig_frame = orig_frame[245:835,140:1780]
    
    frame = cv2.resize(orig_frame.copy(), (800, 288))

    frame = torch.from_numpy(frame).cuda()
    frame = frame.permute(2, 0, 1)/255.0
    frame = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(frame)
    frame = torch.reshape(frame,(1, 3, 288, 800))

    with torch.no_grad():
        out,classes = net(frame)

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
    out_j = loc

    vis = orig_frame

    all_lanes = []
    polar_coords = []
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            single_lane = []
            for k in range(out_j.shape[0]):
                if k == int(out_j.shape[0]/2):
                    x_cordinate = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                    y_cordinate = int(img_h * (row_anchor[no_of_row_anchors-1-k]/288)) - 1
                    ppp = (x_cordinate, y_cordinate)
                    vis = draw_text(vis, ppp, lane_class_labels[classes_cpu[i]], color_map["cream"])
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[no_of_row_anchors-1-k]/288)) - 1 )
                    ppp2 = (out_j[k, i] * col_sample_w * img_w / 800 - 1, img_h * (row_anchor[no_of_row_anchors-1-k]/288) - 1 )
                    single_lane.append(list(ppp2))
                    cv2.circle(vis,ppp,5,(0,255,0),-1)
            all_lanes.append(single_lane)

            if i == 0 or i == 3:
                continue

            
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
            polar = list(cart2polar(a, b))
            polar_coords.append(polar)
    polar_coords = np.array(polar_coords)
    
    if (len(polar_coords)==2):
        beta = abs(polar_coords[0][1]+polar_coords[1][1])
        if (beta > 0.8):
            print(beta, "lane crossing")

    tracked_coords = tracker.update(polar_coords)
    out_vis = draw_lines(tracked_coords,vis)
    cv2.imshow("preview", out_vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()