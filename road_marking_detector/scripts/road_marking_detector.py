#!/usr/bin/env python3

from pathlib import Path
import os
import cv2
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.backends.cudnn as cudnn
from collections import OrderedDict, namedtuple
import tensorrt as trt 
import time
import yaml
from tracking.sort import Sort

import rospy
from sensor_msgs.msg import Image as SensorImage

# Load model
device = torch.device('cuda:0')
# weights = 'runs/train/exp4/weights/tensort_batch2/best.engine'
weights = '/media/fyp/sdCard/detectors/road_marking/road_marking_448_half.engine'
data = '/media/fyp/sdCard/detectors/road_marking/ceymo.yaml'
imgsz = [448, 448]
view_img = True
half = True
sources = "streams.txt"
save_dir = 'runs/detect/experiment'
use_tracker = False

road_marking_detector_publisher = rospy.Publisher('/road_marking_detector_output', SensorImage , queue_size = 1)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, data=None):

        super().__init__()

        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()

        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()
        batch_size = bindings['images'].shape[0]


        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, val=False):

        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 448, 448), half=False):
        # Warmup model by running inference once

        if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
            im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
            self.forward(im)  # warmup

class Annotator:

    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'

        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label


        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)


    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

class Colors:

    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class road_marking_detector:
    def __init__(self, use_tracker):
        self.device = device
        self.colors = Colors()
        self.model = DetectMultiBackend(weights, device=self.device, data=data)

        self.scale_factor = 3
        self.ipt_transform, self.max_height, self.max_width = self.get_ipt_transform(frame_height=1080, frame_width=1920, scale_factor=self.scale_factor)
        self.ipt_transform_inverse = np.linalg.inv(self.ipt_transform)

        self.use_tracker = use_tracker

        # Half
        self.half = half & (device.type != 'cpu')  # FP16 supported on limited backends with CUDA
        cudnn.benchmark = True  # set True to speed up constant image size inference

        self.model.warmup(imgsz=(1, 3, *imgsz), half=self.half)  # warmup
        self.tracker = Sort(max_age=4, min_hits=4, use_dlib = False)
        self.names = self.model.names

        self.seen = 0
        self.dt = 0.0

    def time_sync(self):
        # pytorch-accurate time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def clip_coords(self, boxes, shape):

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):

        coords[:, [0, 2]] *= img0_shape[1]/img1_shape[0]  # x padding
        coords[:, [1, 3]] *= img0_shape[0]/img1_shape[1]  # y padding

        self.clip_coords(coords, img0_shape)
        return coords

    def box_iou(self, box1, box2):

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


    def get_ipt_transform(self, frame_height, frame_width, scale_factor):   # obatin the inverse perspective transform

        # top_left = (int(0.1823 * frame_width / scale_factor), int(0.7176 * frame_height / scale_factor))
        # top_right = (int(0.8333 * frame_width / scale_factor), int(0.7176 * frame_height / scale_factor))
        # bottom_right = (int(1.4167 * frame_width / scale_factor), int(1 * frame_height / scale_factor))
        # bottom_left = (int(-0.3906 * frame_width / scale_factor), int(1 * frame_height / scale_factor))

        top_left = (int(0.3323 * frame_width / scale_factor), int(0.5176 * frame_height / scale_factor))
        top_right = (int(0.6833 * frame_width / scale_factor), int(0.5176 * frame_height / scale_factor))
        bottom_right = (int(1.4167 * frame_width / scale_factor), int(0.85 * frame_height / scale_factor))
        bottom_left = (int(-0.3906 * frame_width / scale_factor), int(0.85 * frame_height / scale_factor))

        src = np.array((top_left, top_right, bottom_right, bottom_left), np.float32) 

        width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0])**2) + ((bottom_right[1] - bottom_left[1])**2))
        width_top = np.sqrt(((top_right[0] - top_left[0])**2) + ((top_right[1] - top_left[1])**2))

        height_right = np.sqrt(((top_right[0] - bottom_right[0])**2) + ((top_right[1] - bottom_right[1])**2))
        height_left = np.sqrt(((top_left[0] - bottom_left[0])**2) + ((top_left[1] - bottom_left[1])**2))

        max_width = max(int(width_bottom), int(width_top))
        max_height = max(int(height_right), int(height_left))

        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype = "float32")

        ipt_transform = cv2.getPerspectiveTransform(src, dst)

        return ipt_transform, max_height, max_width

    def polygon_from_bboxes(self, bboxes, ipt_transform_inverse, max_height, max_width, scale_factor, input_size):   # convert the bounding box to a 4-sided polygon


        polygons = []
        for bbox in bboxes:
            polygon = []
            out_matrix1 = np.matmul(ipt_transform_inverse, np.array([bbox[0] * (max_width / input_size[1]), bbox[1] * (max_height / input_size[0]), 1]).reshape(3,1))
            out_matrix2 = np.matmul(ipt_transform_inverse, np.array([bbox[2] * (max_width / input_size[1]), bbox[1] * (max_height / input_size[0]), 1]).reshape(3,1))
            out_matrix3 = np.matmul(ipt_transform_inverse, np.array([bbox[2] * (max_width / input_size[1]), bbox[3] * (max_height / input_size[0]), 1]).reshape(3,1))
            out_matrix4 = np.matmul(ipt_transform_inverse, np.array([bbox[0] * (max_width / input_size[1]), bbox[3] * (max_height / input_size[0]), 1]).reshape(3,1))
            polygon.append([int(out_matrix1[0,0] * scale_factor / out_matrix1[2,0]), int(out_matrix1[1,0] * scale_factor / out_matrix1[2,0])])
            polygon.append([int(out_matrix2[0,0] * scale_factor / out_matrix2[2,0]), int(out_matrix2[1,0] * scale_factor / out_matrix2[2,0])])
            polygon.append([int(out_matrix3[0,0] * scale_factor / out_matrix3[2,0]), int(out_matrix3[1,0] * scale_factor / out_matrix3[2,0])])
            polygon.append([int(out_matrix4[0,0] * scale_factor / out_matrix4[2,0]), int(out_matrix4[1,0] * scale_factor / out_matrix4[2,0])])
            polygons.append(polygon)
        return polygons

    def draw_text(self, img, label_point, text, color):  # for highlighted text

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

    def non_max_suppression(self, prediction, conf_thres=0.5, iou_thres=0.45, classes=None, agnostic=True, multi_label=False,
                            labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output


    def inference(self,image):

        # t1 = self.time_sync()

        im = cv2.resize(image.copy(),(int(1920 / self.scale_factor),int(1080 / self.scale_factor)))
        im = cv2.warpPerspective(im, self.ipt_transform, (self.max_width, self.max_height))
        # t3 = self.time_sync()
        im = cv2.resize(im, tuple(imgsz))
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB


        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        #Inference
        pred = self.model(im)

        # NMS
        pred = self.non_max_suppression(prediction = pred)
        s = "Detection frame: "

        wide_pred = pred[0]
        wide_pred[:, :4] = self.scale_coords(im.shape[2:], wide_pred[:, :4], tuple(imgsz)).round()

        # Tracker
        numpy_boxes = wide_pred.cpu().numpy()
        # track_out = self.tracker.update(numpy_boxes)
        # track_out_torch = torch.from_numpy(track_out)

        # annotator_wid = Annotator(image, line_width=2)

        classeslist = numpy_boxes[:,5]

        # t2 = self.time_sync()
        # self.dt += t2 - t1
        # self.seen += 1

        # print("inference time", self.seen/self.dt)

        polygonlist = self.polygon_from_bboxes(numpy_boxes[:,:4], self.ipt_transform_inverse, self.max_height, self.max_width, self.scale_factor, tuple(imgsz))
        for q in range(len(classeslist)):
            points = np.array(polygonlist[q])
            image = cv2.polylines(image, [points], True, self.colors(int(classeslist[q]), True), thickness = 5)
            image = self.draw_text(image,(int(polygonlist[q][0][0]), int(polygonlist[q][0][1])), self.names[int(classeslist[q])], self.colors(int(classeslist[q]), True))


        return image

def road_marking_detector_callback(data):
    
    frame_height = data.height
    frame_width = data.width    
    frame = np.frombuffer(data.data, dtype = np.uint8).reshape(frame_height, frame_width, -1)

    frame = road_marking_detector.inference(image=frame)

    out_frame = SensorImage()
    out_frame.header.stamp = rospy.Time.now()
    out_frame.height = frame.shape[0]        
    out_frame.width = frame.shape[1]         
    out_frame.encoding = "rgb8"              
    out_frame.is_bigendian = False           
    out_frame.step = 3 * frame.shape[1]      
    out_frame.data = frame.tobytes()

    road_marking_detector_publisher.publish(out_frame)

def road_marking_detector_func():
    rospy.loginfo("Road marking detector initiated...")
    rospy.init_node('road_marking_detector', anonymous = True)
    cam_count = int(rospy.get_param("cam_count"))

    if (cam_count == 1):
        rospy.Subscriber('/single_input_frame', SensorImage, road_marking_detector_callback)
    elif(cam_count ==2):
        rospy.Subscriber('/wide_camera_frame', SensorImage, road_marking_detector_callback)
    else:
        print("error")


    rospy.spin()

if __name__=='__main__':
    
    try:

        road_marking_detector = road_marking_detector(use_tracker = use_tracker)
        road_marking_detector_func()

    except rospy.ROSInterruptException:
        pass
