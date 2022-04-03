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
from tracking import Sort
import json
# Load model
device = torch.device('cuda:0')
weights = '/media/fyp/sdCard/yolov5/models/yolov5s/448_half_batch_2.engine'
data = '/home/fyp/Documents/yolov5/dualcam.yaml'
imgsz = (448, 448)
view_img = True
half = True
save_dir = 'runs/detect/experiment'
narrow_camera_data_file = "/home/fyp/catkin_ws/src/traffic_light_detector/scripts/camera/narrow_param.txt"
homography_matrix_file = '/home/fyp/catkin_ws/src/traffic_light_detector/scripts/camera/homography_matrix.txt'

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

class inference2:
    def  __init__(self, use_tracker = False):

        self.colors = Colors()
        self.model = DetectMultiBackend(weights, device=device, data=data)
        self.use_tracker = use_tracker
        if use_tracker:
            self.tracker = Sort(max_age=5, min_hits=4, use_dlib = False, min_age = 0)
        self.names = self.model.names

        # Half
        # self.half = True
        self.half = half & (device.type != 'cpu')  # FP16 supported on limited backends with CUDA
        self.stride = 64
        cudnn.benchmark = True  # set True to speed up constant image size inference

        self.fps = 30

        self.camera_mtx_narrow, self.dist_coff_narrow, self.newcameramtx_narrow, self.homography_matrix = self.get_camera_param(narrow_camera_data_file,homography_matrix_file)

        self.camera_bounding = torch.tensor([[0,0,1920,1080]], device=torch.device("cuda"))
        self.camera_bounding = self.camera_transform(self.camera_bounding,self.homography_matrix,self.camera_mtx_narrow, self.dist_coff_narrow, self.newcameramtx_narrow)
        self.camera_bounding = self.camera_bounding.cpu().numpy()
        self.camera_bounding = self.camera_bounding[0].tolist()
        self.camera_bounding = list(map(int,self.camera_bounding))

        self.model.warmup(imgsz=(2, 3, *imgsz), half=half)  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
        self.narrow_det = None


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

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):

        coords[:, [0, 2]] *= img0_shape[1]/img1_shape[0]  # x padding
        coords[:, [1, 3]] *= img0_shape[0]/img1_shape[1]  # y padding

        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self,boxes, shape):

        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def box_iou(self,box1, box2):

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def size_conf_filter(self, det, min_size = 5, min_conf = 0.8):
        choice = det[:,4] > min_conf
        conf_filtered = det[choice]
        
        min_values = torch.minimum(abs(conf_filtered[:,3]-conf_filtered[:,1]),abs(conf_filtered[:,2]-conf_filtered[:,0]))
        choice_size = min_values > min_size
        filtered = conf_filtered[choice_size]
        return filtered


    def non_max_suppression(self,prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=True, multi_label=False,
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


    def check_position(self, boxes, topx,topy, botx, boty):
    
        keep = (boxes[:, 0]>topx) & (boxes[:, 1]>topy) & (boxes[:, 2]<botx) & (boxes[:, 3]<boty)

        return keep

    def _upcast(self,t):
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()

    def box_area(self,boxes):
        boxes = self._upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def _box_inter_union(self, boxes1, boxes2):
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = self._upcast(rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        return inter, union

    def iou_algo(self,boxes1, boxes2):

        inter, union = self._box_inter_union(boxes1, boxes2)
        iou = (inter / union)>0.5
        return iou

    def clip_boxes_to_box(self, boxes):
        """
        Clip boxes so that they lie inside an image of size `size`.

        Args:
            boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
                with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            size (Tuple[height, width]): size of the image

        Returns:
            Tensor[N, 4]: clipped boxes
        """
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]

        boxes_x = boxes_x.clamp(min=707, max=1160)
        boxes_y = boxes_y.clamp(min=387, max=654)

        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    def get_camera_param(self, cam_details_file,homograpy_mtx_file):

        camera_matrix_raw = 15
        distortion_raw = 20
        rectification_raw = 23
        projection_raw = 27

        w,h = 1920,1080

        camera_mtx_narrow = np.loadtxt(cam_details_file, dtype=np.float32, skiprows=camera_matrix_raw-1, max_rows=3, delimiter=' ')
        dist_coff_narrow = np.loadtxt(cam_details_file, dtype=np.float32, skiprows=distortion_raw-1,max_rows=1, delimiter=' ')
        newcameramtx_narrow, roi = cv2.getOptimalNewCameraMatrix(camera_mtx_narrow, dist_coff_narrow, (w,h), 1, (w,h))
        homograpy_mtx = np.loadtxt(homograpy_mtx_file, dtype=np.float32, usecols=range(3))

        return (camera_mtx_narrow, dist_coff_narrow, newcameramtx_narrow, homograpy_mtx)

    def camera_transform(self, coords,M,camera_mtx, dist_coff, newcameramtx):
        coords = coords.cpu()
        npcoords = np.array(coords)
        pts = np.float32(np.vstack(np.hsplit(npcoords,2))).reshape(-1,1,2)


        undist_cords =  cv2.undistortPoints(pts,camera_mtx,dist_coff,P=newcameramtx)
        # narrow_undist_cords = np.float32(narrow_undist_cords).reshape(-1,1,2)


        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(undist_cords,M) 
        dst_final = np.hstack(np.vsplit(np.squeeze(dst),2))
        return torch.from_numpy(dst_final)

    def inference2(self, img_narrow, img_wide):

        # Convert
        img_narrow_ = cv2.resize(img_narrow.copy(),imgsz,interpolation=cv2.INTER_LINEAR)
        img_wide_ = cv2.resize(img_wide.copy(),imgsz,interpolation=cv2.INTER_LINEAR)

        im = np.stack((img_narrow_,img_wide_),axis=0)
        im = im.transpose((0, 3, 1, 2))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        t1 = self.time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = self.time_sync()
        self.dt[0] += t2 - t1

        # Inference

        pred = self.model(im)
        t3 = self.time_sync()
        self.dt[1] += t3 - t2


        # NMS
  
        pred = self.non_max_suppression(prediction = pred)
        self.dt[2] += self.time_sync() - t3
        s = "Detection frame: "

        narrow_pred = pred[1]
        wide_pred = pred[0]
        wide_pred[:, :4] = self.scale_coords(im.shape[2:], wide_pred[:, :4], img_wide.shape).round()
        narrow_pred[:, :4] = self.scale_coords(im.shape[2:], narrow_pred[:, :4], img_narrow.shape).round()
        narrow_pred.type

        # Converting N to No
        if len(narrow_pred):
            narrow_pred[:, :4] = self.camera_transform(narrow_pred[:, :4],self.homography_matrix,self.camera_mtx_narrow, self.dist_coff_narrow, self.newcameramtx_narrow)

        # Bounding box control algorithm
        rev_wide_det = reversed(wide_pred)
        keep = torch.logical_not(self.check_position(rev_wide_det[:,:4],self.camera_bounding[0], self.camera_bounding[1],self.camera_bounding[2], self.camera_bounding[3]))
        rev_wide_det = rev_wide_det[keep]
        rev_wide_det_old = torch.clone(rev_wide_det)
        rev_wide_det[:,:4] = self.clip_boxes_to_box(rev_wide_det[:,:4])
        rev_narrow_det = reversed(narrow_pred)
        q = self.iou_algo(rev_wide_det[:,:4], rev_narrow_det[:,:4])
        rev_narrow_det = rev_narrow_det[torch.all(torch.logical_not(q),0)]
        wide_pred = torch.cat((rev_wide_det_old, rev_narrow_det), 0)

        annotator_wid = Annotator(img_wide, line_width=2)

        if self.use_tracker:

            filtered_dets = self.size_conf_filter(wide_pred, min_size = 10, min_conf = 0.8)

            # Tracker
            numpy_boxes = filtered_dets.cpu().numpy()
            track_out = self.tracker.update(numpy_boxes)
            track_out_torch = torch.from_numpy(track_out)


            annotations = self.get_annotations(track_out_torch)
            # Visualization with tracker
            for *xyxy, id_val, cls in track_out_torch:
                c = int(cls)  # integer class
                label = str(int(id_val))
                annotator_wid.box_label(xyxy, label, color=self.colors(0, True))

        else:
            annotations = self.get_annotations(wide_pred)
            for *xyxy, conf, cls in wide_pred:
                c = int(cls)  # integer class
                label = f'{self.names[c]} {conf:.2f}'
                # label = "out"
                annotator_wid.box_label(xyxy, label, color=self.colors(c, True))
            

        im_view_wid = annotator_wid.result()

        return im_view_wid,annotations

    def get_annotations(self, pred):

        annotations = []
        
        if len(pred):
    
            for *xyxy, conf, cls in (pred):
                
                xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                c = int(cls)  # integer class
                label = f'{self.names[c]}'

                annotation = {'type':label,
                            "xmin":xmin,
                            "ymin":ymin,
                            "xmax":xmax,
                            "ymax":ymax
                            }
                annotations.append(annotation)

        return annotations
