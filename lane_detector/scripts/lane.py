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
        y = np.array([295,590])
        x = (y - b)/a
        
        image = cv2.line(image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)
        
    return image



model_path = '/home/fyp2selfdriving/Documents/lane_detection/culane_18.pth'

row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

cls_num_per_lane = 18
net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,cls_num_per_lane,4),
                use_aux=False).cuda()

state_dict = torch.load(model_path, map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()


cap = cv2.VideoCapture("/home/fyp2selfdriving/Documents/lane_detection/Sri Lankan lane dataset/lane_tracker/cut_3.mp4")

img_w, img_h = 1640,590

tracker = Sort(max_age=7, min_hits=10, use_dlib = False)

while True:


    r, frame = cap.read()
    if not r:
        break
    frame = cv2.resize(frame, (1920, 1080))
    frame = frame[450:,:]
    im_pil = Image.fromarray(frame.copy())
    imgs = img_transforms(im_pil)
    
    imgs = imgs.cuda()
    imgs = imgs[None, :]
    with torch.no_grad():
        out = net(imgs)

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
            polar = list(cart2polar(a, b))
            polar_coords.append(polar)
    polar_coords = np.array(polar_coords)

    tracked_coords = tracker.update(polar_coords)
    out_vis = draw_lines(tracked_coords,vis)
    cv2.imshow("preview", out_vis)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()