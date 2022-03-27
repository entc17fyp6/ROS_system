
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
from sensor_msgs.msg import Image as CameraImage

width = 1080
height = 1920
fps = 30
narrow_AutoExposureTimeUpperLimit = 700
wide_AutoExposureTimeUpperLimit = 1000
quality_factor = 30
shold_save_video = False
should_visualize = True

# camera_name = None

# input_frame_publisher = rospy.Publisher('/input_frame',CameraImage,queue_size=1)
# rospy.init_node('image_feeder_node', anonymous=True)

# def publish_image(frame):

#     frame = draw_time_date(frame)

#     input_frame = CameraImage()
#     input_frame.header.stamp = rospy.Time.now()
#     input_frame.height = frame.shape[0]
#     input_frame.width = frame.shape[1]
#     input_frame.encoding = "rgb8"
#     input_frame.is_bigendian = False
#     input_frame.step = 3* frame.shape[1]
#     input_frame.data = frame.tobytes()

#     input_frame_publisher.publish(input_frame)
    

        
#     frame = cv2.cvtColor(cv2.resize(frame, (height,width)), cv2.COLOR_RGB2BGR)
#     cv2.namedWindow("Input")
#     cv2.imshow("Input", frame)
#     cv2.waitKey(1)

#     return


### this FFMPEG_VideoWriter class is copied and modified from https://github.com/basler/pypylon/issues/113 

### for demonstration of how to write video data
### this class is an excerpt from the project moviepy https://github.com/Zulko/moviepy.git moviepy/video/io/ffmpeg_writer.py
###

class FFMPEG_VideoWriter:
    """ A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Parameters
    -----------

    filename
      Any filename like 'video.mp4' etc. but if you want to avoid
      complications it is recommended to use the generic extension
      '.avi' for all your videos.

    size
      Size (width,height) of the output video in pixels.

    fps
      Frames per second in the output video file.

    codec
      FFMPEG codec. It seems that in terms of quality the hierarchy is
      'rawvideo' = 'png' > 'mpeg4' > 'libx264'
      'png' manages the same lossless quality as 'rawvideo' but yields
      smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
      of accepted codecs.

      Note for default 'libx264': by default the pixel format yuv420p
      is used. If the video dimensions are not both even (e.g. 720x405)
      another pixel format is used, and this can cause problem in some
      video readers.

      Experimentally found best options 
        libx264         - quality - very good     speed - ~30fps achieved       size - 16.74 GB/h
        libx265         - quality - very good     speed - ~15fps achieved       size - 1.396 GB/h
        mjpeg(-q:v=25)  - quality - good          speed - ~30fps achieved       size - 3.66 GB/h
        mpeg(-q:v=11)   - quality - very good     speed - ~30fps achieved       size - 1.624 GB/h
        h264_qsv(-q:v=30) - quality - good        speed - ~30fps achieved       size - 1.6GB/h      -hardware acceleration
 -
    audiofile
      Optional: The name of an audio file that will be incorporated
      to the video.

    preset
      Sets the time that FFMPEG will take to compress the video. The slower,
      the better the compression rate. Possibilities are: ultrafast,superfast,
      veryfast, faster, fast, medium (default), slow, slower, veryslow,
      placebo. 

      This and affects only for the libx264, libx265 libxvid etc. ('-crf' also affect these types)
      for mjpeg, mpeg4 etc. use -q:v factor

    bitrate
      Only relevant for codecs which accept a bitrate. "5000k" offers
      nice results in general.

    withmask
      Boolean. Set to ``True`` if there is a mask in the video to be
      encoded.

    """

    def __init__(self, filename, size, fps, codec="libx264", audiofile=None,
                 preset="medium", bitrate=None, pixfmt="rgba", quality = '11',crf = '20',
                 logfile=None, threads=None, ffmpeg_params=None):

        if logfile is None:
            logfile = sp.PIPE

        self.filename = filename
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        
        # order is important\
        cmd = [
            "ffmpeg",
            '-y',
            '-loglevel', 'error' if logfile == sp.PIPE else 'info',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (size[1], size[0]),
            '-pix_fmt', pixfmt,
            '-r', '%.02f' % fps,
            '-i', '-', '-an',
            # '-vf', "curves=r='0/0 0.25/0.4 0.5/0.5 1/1':g='0/0 0.25/0.4 0.5/0.5 1/1':b='0/0 0.25/0.4 0.5/0.5 1/1'",
            # '-vf', "drawtext='fontfile=c\:/Windows/Fonts/Calibri.ttf:text=%{localtime}:fontcolor=yellow:fontsize=35:x=1600:y=20:'"
            '-vf', "curves=r='0/0 0.25/0.4 0.5/0.5 1/1':g='0/0 0.25/0.4 0.5/0.5 1/1':b='0/0 0.25/0.4 0.5/0.5 1/1', drawtext='fontfile=c\:/Windows/Fonts/Calibri.ttf:text=%{localtime}:fontcolor=yellow:fontsize=35:x=1600:y=20:'"
        ]
        cmd.extend([
            '-vcodec', codec,
            '-q:v', quality,
            # '-crf', crf,
            '-preset', preset,
        ])
        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)
        if bitrate is not None:
            cmd.extend([
                '-b', bitrate
            ])
        if threads is not None:
            cmd.extend(["-threads", str(threads)])

        if ((codec == 'libx264') and
                (size[0] % 2 == 0) and
                (size[1] % 2 == 0)):
            cmd.extend([
                '-pix_fmt', 'yuv420p'
            ])
        cmd.extend([
            filename
        ])

        popen_params = {"stdout": sp.DEVNULL,
                        "stderr": logfile,
                        "stdin": sp.PIPE,
                        # "shell":True   ## keep this line in windows 10, commentout in ubuntu 20.04
                        }

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        self.proc = sp.Popen(cmd, **popen_params)


    def write_frame(self, img_array):
        """ Writes one frame in the file."""
        try:
               self.proc.stdin.write(img_array.tobytes())
        except IOError as err:
            _, ffmpeg_error = self.proc.communicate()
            error = (str(err) + ("\n\nMoviePy error: FFMPEG encountered "
                                 "the following error while writing file %s:"
                                 "\n\n %s" % (self.filename, str(ffmpeg_error))))

            if b"Unknown encoder" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed because FFMPEG didn't find the specified "
                  "codec for video encoding (%s). Please install "
                  "this codec or change the codec when calling "
                  "write_videofile. For instance:\n"
                  "  >>> clip.write_videofile('myvid.webm', codec='libvpx')")%(self.codec)

            elif b"incorrect codec parameters ?" in ffmpeg_error:

                 error = error+("\n\nThe video export "
                  "failed, possibly because the codec specified for "
                  "the video (%s) is not compatible with the given "
                  "extension (%s). Please specify a valid 'codec' "
                  "argument in write_videofile. This would be 'libx264' "
                  "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                  "Another possible reason is that the audio codec was not "
                  "compatible with the video codec. For instance the video "
                  "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                  "video codec."
                  )%(self.codec, self.ext)

            elif  b"encoder setup failed" in ffmpeg_error:

                error = error+("\n\nThe video export "
                  "failed, possibly because the bitrate you specified "
                  "was too high or too low for the video codec.")

            elif b"Invalid encoder type" in ffmpeg_error:

                error = error + ("\n\nThe video export failed because the codec "
                  "or file extension you provided is not a video")


            raise IOError(error)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()

        self.proc = None

    # Support the Context Manager protocol, to ensure that resources are cleaned up.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def save_video(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
    writer.write_frame(frame)
    return

def visualize(frame):
    frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
                if (shold_save_video):
                    save_video(img)
                if (should_visualize):
                    visualize(img)
                # writer.write_frame(img)
            else:
                raise RuntimeError("Grab failed")
        except Exception as e:
            traceback.print_exc()

def BackgroundLoop(cam):
    handler = ImageHandler()

    cam.RegisterImageEventHandler(handler, py.RegistrationMode_ReplaceAll, py.Cleanup_None)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    global writer
    with FFMPEG_VideoWriter("videos/"+dt_string+"_"+camera_name+"_camera_"+".mp4",(cam.Height.Value, cam.Width.Value), fps=fps, pixfmt="yuv420p", codec="h264_qsv", quality= str(quality_factor), preset= 'fast') as writer:
        # cam.StartGrabbingMax(100, py.GrabStrategy_LatestImages, py.GrabLoop_ProvidedByInstantCamera)
        cam.StartGrabbing(py.GrabStrategy_LatestImageOnly, py.GrabLoop_ProvidedByInstantCamera)
        # cam.StartGrabbing(py.GrabStrategy_LatestImages, py.GrabLoop_ProvidedByInstantCamera)

        try:
            while cam.IsGrabbing():
                pass
        except KeyboardInterrupt:
            pass

        cam.StopGrabbing()
        cam.DeregisterImageEventHandler(handler)
        cam.Close()
        cv2.destroyAllWindows()

    # return handler.img_sum






if __name__ == '__main__':
    try:
        tlf = py.TlFactory.GetInstance()
        cam = py.InstantCamera(tlf.CreateFirstDevice())
        cam.Open()

        camera_name = cam.DeviceInfo.GetUserDefinedName()
        print(f"connected to {camera_name} camera")

        initialize_cam(cam, camera_name)

        BackgroundLoop(cam)
        
    except rospy.ROSInternalException:
        pass