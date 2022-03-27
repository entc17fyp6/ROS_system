#! /usr/bin/env python3.6

from tkinter.tix import Tree
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


width = 1080
height = 1920
# fps = 30
# narrow_AutoExposureTimeUpperLimit = 1000
# wide_AutoExposureTimeUpperLimit = 1000
quality_factor = 30

should_feed_video = True
should_visualize = False
# should_filter = True


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

def image_feed(frame):
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

    input_frame_publisher.publish(input_frame)
    return

def visualize(frame):
    # frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Input")
    cv2.imshow("Input", frame)
    cv2.waitKey(1)
    return

def initialize_cam(cam,camera_name):
    cam.UserSetSelector = 'Default'
    cam.UserSetLoad.Execute()

    # cam.PixelFormat = 'YCbCr422_8'
    # cam.ExposureTime = 200
    cam.PixelFormat = 'BayerGB8'
    cam.ExposureTime = 300
    cam.AcquisitionFrameRate = fps
    # cam.BslBrightness = 0.4
    # cam.BslContrast = 0.4
    if (camera_name == 'Narrow'):
        cam.ReverseX = True
        cam.ReverseY = True
        cam.ExposureAuto = 'Continuous'
        cam.AutoExposureTimeUpperLimit = narrow_AutoExposureTimeUpperLimit
        cam.AutoGainUpperLimit = 5.0

        # cam.LightSourcePreset = 'Off'
        # cam.BalanceWhiteAuto = 'Off'
        # cam.Gain = 1
        # cam.GainAuto = 'Off'
        
        # cam.BalanceRatioSelector = 'Red'
        # cam.BalanceRatio = 2.1875
        # cam.BalanceRatioSelector = 'Green'
        # cam.BalanceRatio = 1.46875
        # cam.BalanceRatioSelector = 'Blue'
        # cam.BalanceRatio = 1.76

    if (camera_name == 'Wide'):
        cam.ExposureAuto = 'Continuous'
        cam.AutoExposureTimeUpperLimit = wide_AutoExposureTimeUpperLimit
        cam.AutoGainUpperLimit = 5.0
        

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
                time_new = time.time()
                rate = 1/(time_new-self.time_old)
                print(rate)
                self.time_old = time_new
                if (should_filter):
                    img = ffmpeg_filter.filter(img)
                if (should_feed_video):
                    image_feed(img)
                if (should_visualize):
                    visualize(img)
            else:
                raise RuntimeError("Grab failed")
        except Exception as e:
            traceback.print_exc()

def BackgroundLoop(cam):
    handler = ImageHandler()

    cam.RegisterImageEventHandler(handler, py.RegistrationMode_ReplaceAll, py.Cleanup_None)

    global ffmpeg_filter
    # with ffmpeg_filter_class((cam.Height.Value, cam.Width.Value), fps=fps, pixfmt="yuv420p", codec="h264_qsv", quality= str(quality_factor), preset= 'fast') as feeder:
    with ffmpeg_filter_class(h=cam.Height.Value,w=cam.Width.Value,fps=fps) as ffmpeg_filter:
        cam.StartGrabbing(py.GrabStrategy_LatestImageOnly, py.GrabLoop_ProvidedByInstantCamera)

        try:
            while cam.IsGrabbing():
                pass
        except KeyboardInterrupt:
            pass

        cam.StopGrabbing()
        cam.DeregisterImageEventHandler(handler)
        cam.Close()
        cv2.destroyAllWindows()

def single_camera_image_feeder():
    global input_frame_publisher
    
    tlf = py.TlFactory.GetInstance()
    cam = py.InstantCamera(tlf.CreateFirstDevice())
    cam.Open()

    camera_name = cam.DeviceInfo.GetUserDefinedName()
    print(f"connected to {camera_name} camera")

    initialize_cam(cam, camera_name)

    input_frame_publisher = rospy.Publisher('/output_frame',SensorImage,queue_size=1)
    rospy.init_node('image_feeder_node', anonymous=True)

    BackgroundLoop(cam)

    return

if __name__ == '__main__':
    try:
        fps = int(rospy.get_param("fps"))
        narrow_AutoExposureTimeUpperLimit = int(rospy.get_param("narrow_AutoExposureTimeUpperLimit"))
        wide_AutoExposureTimeUpperLimit = int(rospy.get_param("wide_AutoExposureTimeUpperLimit"))
        should_filter = rospy.get_param("should_filter")

        single_camera_image_feeder()
    except rospy.ROSInternalException:
        pass