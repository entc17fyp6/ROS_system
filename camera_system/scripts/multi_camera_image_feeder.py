#! /usr/bin/env python3.6

import pypylon.pylon as py
import numpy as np
import matplotlib.pyplot as plt
import traceback
import time
from datetime import datetime
import cv2
import subprocess as sp
import os

import rospy
from sensor_msgs.msg import Image as SensorImage

should_feed_video = True
should_visualize = False
# should_filter = True

width = 1080
height = 1920
# fps = 5
quality = '25'
PixelFormat = "YCbCr422_8"   ## BayerGB8 YCbCr422_8
# wide_AutoExposureTimeUpperLimit = 20000
# narrow_AutoExposureTimeUpperLimit = 20000


def initialize_cam(cam,camera_name):
    cam.UserSetSelector = 'Default'
    cam.UserSetLoad.Execute()

    cam.PixelFormat = PixelFormat
    # cam.ExposureTime = 2
    # cam.PixelFormat = 'BayerGB8'
    cam.ExposureTime = 300  #300
    cam.AcquisitionFrameRate = fps
    # cam.BslBrightness = 0.4
    # cam.BslContrast = 0.4
    if (camera_name == 'Narrow'):
        cam.ReverseX = True
        cam.ReverseY = True
        cam.ExposureAuto = 'Continuous'
        cam.AutoExposureTimeUpperLimit = narrow_AutoExposureTimeUpperLimit #1000
        cam.AutoGainUpperLimit = 5.0
    if (camera_name == 'Wide'):
        cam.ExposureAuto = 'Continuous'
        cam.AutoExposureTimeUpperLimit = wide_AutoExposureTimeUpperLimit  #2000
        cam.AutoGainUpperLimit = 5.0




class ffmpeg_filter_class:
    def __init__(self, h=1080,w=1920,fps=30):
        self.w = w
        self.h = h
        self.command = ['ffmpeg', '-y',
                '-s', '{}x{}'.format(h, w),
                '-f', 'rawvideo',
                '-r', '%.02f' % fps,
                '-an',
                '-pix_fmt', 'bgr24',
                '-i', '-',
                '-vf', "curves=r='0/0 0.25/0.4 0.5/0.5 1/1':g='0/0 0.25/0.4 0.5/0.5 1/1':b='0/0 0.25/0.4 0.5/0.5 1/1', drawtext='fontfile=c\:/Windows/Fonts/Calibri.ttf:text=%{localtime}:fontcolor=yellow:fontsize=35:x=1600:y=20:'",
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vcodec', 'rawvideo', '-']

        self.proc = sp.Popen(self.command, stderr=sp.DEVNULL ,stdout = sp.PIPE, stdin=sp.PIPE )
        

    def filter(self,frame):
        self.proc.stdin.write(frame.tobytes())
        frame = self.proc.stdout.read(self.w*self.h*3)
        frame = np.frombuffer(frame, np.uint8).reshape((self.h, self.w, 3))
        return frame

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()

        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def image_feed(frame,cam_id):
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1920,1080))

    input_frame = SensorImage()

    input_frame.header.stamp = rospy.Time.now()
    input_frame.height = frame.shape[0]
    input_frame.width = frame.shape[1]
    input_frame.encoding = "rgb8"
    input_frame.is_bigendian = False
    input_frame.step = 3* frame.shape[1]
    input_frame.data = frame.tobytes()

    if (cam_id == 1):
        wide_frame_publisher.publish(input_frame)
    elif (cam_id == 0):
        narrow_frame_publisher.publish(input_frame)

    return

def visualize(frame,cam_id):
    # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(str(cam_id))
    cv2.imshow(str(cam_id), frame)
    cv2.waitKey(1)
    return



class ImageHandler (py.ImageEventHandler):
    def __init__(self, *args):
        super().__init__(*args)
        self.time_old = time.time()

        self.converter = py.ImageFormatConverter()
        self.converter.OutputPixelFormat = py.PixelType_RGB8packed
        self.converter.OutputBitAlignment = "MsbAligned"
    
    def OnImageGrabbed(self, camera, grabResult):
        try:
            if grabResult.GrabSucceeded():
                
                if (~self.converter.ImageHasDestinationFormat(grabResult)):
                    grabResult = self.converter.Convert(grabResult)
                    
                img = grabResult.Array
                cam_id = camera.CameraContext
                time_new = time.time()
                rate = 1/(time_new-self.time_old)
                if (cam_id == 0):
                    print("                     ",rate,cam_id)
                else:
                    print(rate, cam_id)
                self.time_old = time_new
                if (should_filter):
                    img = ffmpeg_filter_dict[str(cam_id)].filter(img)
                if (should_feed_video):
                    image_feed(img, cam_id)
                if (should_visualize):
                    visualize(img, cam_id)
                # save_video(img,cam_id)
            else:
                raise RuntimeError("Grab failed")
        except Exception as e:
            traceback.print_exc()


def BackgroundLoop(cam_array):

    handler_array = []
    for i in range (cam_count):
        handler_array.append(ImageHandler())

    for cam in cam_array:
        cam.RegisterImageEventHandler(handler_array[i], py.RegistrationMode_ReplaceAll, py.Cleanup_None)
        


    # global writer
    # writer = []
    global ffmpeg_filter_dict
    ffmpeg_filter_dict = {}
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    # video_name = dt_string+"_camera_"

    if (wide_cam_connected):
        ffmpeg_filter_dict[str(wide_cam_id)] = ffmpeg_filter_class(h=cam.Height.Value,w=cam.Width.Value,fps=fps)

    if (narrow_cam_connected):
        ffmpeg_filter_dict[str(narrow_cam_id)] = ffmpeg_filter_class(h=cam.Height.Value,w=cam.Width.Value,fps=fps)
    
    for cam in cam_array:
        cam.StartGrabbing(py.GrabStrategy_LatestImageOnly, py.GrabLoop_ProvidedByInstantCamera)

    try:
        while cam_array[0].IsGrabbing():
            pass
    except KeyboardInterrupt:
        pass

    # cam_array.StopGrabbing()
    for i in range (cam_count):
        cam_array[i].StopGrabbing()
        cam_array[i].DeregisterImageEventHandler(handler_array[i])
        # writer[i].close()
    if (wide_cam_connected):
        ffmpeg_filter_dict[str(wide_cam_id)].close()
    if (narrow_cam_connected):
        ffmpeg_filter_dict[str(narrow_cam_id)].close()
    # cam_array.Close()
    cv2.destroyAllWindows()
    return

def multi_camera_image_feeder():

    global wide_frame_publisher,narrow_frame_publisher
    
    global narrow_cam_id, wide_cam_id
    global narrow_cam_connected, wide_cam_connected
    global cam_array,cam_count

    tlf = py.TlFactory.GetInstance()
    devices = tlf.EnumerateDevices()
    cam_count = len(devices)

    for d in devices:
        print(d.GetModelName(), d.GetUserDefinedName())
    
    cam_array = []
    for i in range (cam_count):
        cam_array.append(py.InstantCamera(tlf.CreateDevice(devices[i])))


    for i in range(cam_count):
        cam_array[i].Open()

    narrow_cam_id = 0
    wide_cam_id = 1
    narrow_cam_connected = False
    wide_cam_connected = False

    for idx, cam in enumerate(cam_array):
        camera_name = cam.DeviceInfo.GetUserDefinedName()
        print(f"set context {idx} for {camera_name} camera")
        if (camera_name == 'Wide'):
            cam.SetCameraContext(wide_cam_id)
            wide_cam_connected = True
        else:
            cam.SetCameraContext(narrow_cam_id)
            narrow_cam_connected = True
    
        for idx, cam in enumerate(cam_array):
            camera_name = cam.DeviceInfo.GetUserDefinedName()
            print(camera_name)
            initialize_cam(cam, camera_name)
        
    wide_frame_publisher = rospy.Publisher('/wide_frame',SensorImage,queue_size=1)
    narrow_frame_publisher = rospy.Publisher('/narrow_frame',SensorImage,queue_size=1)
    rospy.init_node('image_feeder_node', anonymous=True)

    BackgroundLoop(cam_array)

if __name__ == '__main__':
    try:
        fps = int(rospy.get_param("fps"))
        narrow_AutoExposureTimeUpperLimit = int(rospy.get_param("narrow_AutoExposureTimeUpperLimit"))
        wide_AutoExposureTimeUpperLimit = int(rospy.get_param("wide_AutoExposureTimeUpperLimit"))
        should_filter = rospy.get_param("should_filter")

        multi_camera_image_feeder()
    except rospy.ROSInternalException:
        pass