#include <pylon/PylonIncludes.h>
// #include <pylon/BaslerUniversalInstantCamera.h>
#include <string>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>

#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include <sensor_msgs/image_encodings.h>
#include <camera_system/img_pair_msg.h>

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using GenApi objects.
using namespace GenApi;

// Namespace for using cout.
using namespace std;

bool should_visualize = false;
bool should_publish = true;

uint32_t width = 1920;
uint32_t height = 1080;
uint8_t fps = 30;
uint16_t narrow_AutoExposureTimeUpperLimit = 10000;
uint16_t wide_AutoExposureTimeUpperLimit = 10000;
double narrow_AutoExposureTimeUpperLimit_launch_file, wide_AutoExposureTimeUpperLimit_launch_file; // will be set by the launch file
int fps_launch_file; // will be set by the launch file
String_t PixelFormat = "YCbCr422_8" ;  //YCbCr422_8  BayerGB8
double AutoGainUpperLimit = 5.0;

intptr_t narrow_cam_id = 0;
intptr_t wide_cam_id = 1;

bool filled_buffer[2] = {0};
cv::Mat frame_buffer[2]; 


ros::Publisher narrow_camera_frame_publisher;
ros::Publisher wide_camera_frame_publisher;
ros::Publisher dual_input_frame_publisher;

void Initialize_cam(CInstantCamera& camera);
void background_loop(CInstantCameraArray& cameras);
void visualize_image(cv::Mat& frame);
void publish_image(cv::Mat& frame);
void publish_image_sync(cv::Mat& frame, intptr_t cam_id);

int main( int argc, char* argv[]  )
{

    ros::init(argc, argv, "multi_camera_image_feeder");
    ros::NodeHandle n;
    
    std::string input_wide_video,input_narrow_video;
    int frame_rate;

    narrow_camera_frame_publisher = n.advertise<sensor_msgs::Image>("/narrow_camera_frame", 1);
    wide_camera_frame_publisher = n.advertise<sensor_msgs::Image>("/wide_camera_frame", 1);

    dual_input_frame_publisher = n.advertise<camera_system::img_pair_msg>("/dual_input_frames", 1);

    if (n.hasParam("narrow_AutoExposureTimeUpperLimit")){
        n.getParam("narrow_AutoExposureTimeUpperLimit", narrow_AutoExposureTimeUpperLimit_launch_file);
        narrow_AutoExposureTimeUpperLimit = (uint16_t) narrow_AutoExposureTimeUpperLimit_launch_file;
    }
    if (n.hasParam("wide_AutoExposureTimeUpperLimit")){
        n.getParam("wide_AutoExposureTimeUpperLimit", wide_AutoExposureTimeUpperLimit_launch_file);
        wide_AutoExposureTimeUpperLimit = (uint16_t) wide_AutoExposureTimeUpperLimit_launch_file;
    }

    if (n.hasParam("frame_rate")){
        n.getParam("frame_rate",fps_launch_file);
        fps = (int8_t) fps_launch_file;
    }

    int exitCode = 0;
    PylonInitialize();

    try
    {
        // Get the transport layer factory.
        CTlFactory& tlFactory = CTlFactory::GetInstance();

        // Get all attached devices and exit application if no device is found.
        DeviceInfoList_t devices;
        if (tlFactory.EnumerateDevices( devices ) == 0)
        {
            throw RUNTIME_EXCEPTION( "No camera present." );
        }

        // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        CInstantCameraArray cameras(devices.size());

        // Create and attach all Pylon Devices.
        for (size_t i = 0; i < cameras.GetSize(); ++i)
        {
            cameras[i].Attach( tlFactory.CreateDevice( devices[i] ) );
            Initialize_cam(cameras[i]);

            // Print the model name of the camera.
            cout << "Using device " << cameras[i].GetDeviceInfo().GetUserDefinedName() << endl;
        }




        background_loop(cameras);

        // cameras.Close();
        PylonTerminate();
    }
    catch (const GenericException& e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
            << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Comment the following two lines to disable waiting on exit.
    cerr << endl << "Press enter to exit." << endl;
    while (cin.get() != '\n');

    // Releases all pylon resources.
    PylonTerminate();

    return exitCode;
}

void Initialize_cam(CInstantCamera& camera){
    String_t cam_name = camera.GetDeviceInfo().GetUserDefinedName();
    if (camera.IsOpen() == false){
        camera.Open();
    }
    camera.ChunkNodeMapsEnable.SetValue(false);

    INodeMap& nodemap = camera.GetNodeMap();

    // set default configuration
    CEnumParameter (nodemap, "UserSetSelector").SetValue("Default");
    CCommandParameter(nodemap, "UserSetLoad").Execute();

    CIntegerParameter ( nodemap, "Width" ).SetValue(width);
    CIntegerParameter ( nodemap, "Height" ).SetValue(height);
    CEnumParameter (nodemap, "PixelFormat").SetValue(PixelFormat);
    CFloatParameter (nodemap, "AcquisitionFrameRate").SetValue(fps);
    CEnumParameter (nodemap, "ExposureAuto").SetValue("Continuous");
    CFloatParameter(nodemap, "AutoGainUpperLimit").SetValue(AutoGainUpperLimit);

    if (cam_name == "Wide"){
        CFloatParameter(nodemap,"AutoExposureTimeUpperLimit").SetValue(wide_AutoExposureTimeUpperLimit);
        camera.SetCameraContext(wide_cam_id);
    }
    else if (cam_name == "Narrow"){
        CFloatParameter(nodemap,"AutoExposureTimeUpperLimit").SetValue(narrow_AutoExposureTimeUpperLimit);
        CBooleanParameter(nodemap, "ReverseX").SetValue(true);
        CBooleanParameter(nodemap, "ReverseY").SetValue(true);
        camera.SetCameraContext(narrow_cam_id);
    }

}

// def save_video(frame,cam_id):
//     global filled_buffer
//     if (filled_buffer[cam_id]==False):
//         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420)
//         filled_buffer[cam_id] = True

//         if (should_save_video):
//             writer_dict[str(cam_id)].write_frame(frame)

//         else:   ## visualize if does not save
//             cv2.namedWindow(str(cam_id))
//             cv2.imshow(str(cam_id), frame)
//             cv2.waitKey(1)

//         if (filled_buffer == [True]*cam_count):
//             filled_buffer = [False]*cam_count

//     return
void publish_image_sync(cv::Mat& frame, intptr_t cam_id){
    if (filled_buffer[cam_id] == false){ 
        filled_buffer[cam_id] = true;

        frame_buffer[cam_id] = frame;

        if ((filled_buffer[0]=true) & (filled_buffer[1]=true)){  // publish frames and reset the array
            filled_buffer[0] = false;
            filled_buffer[1] = false;

            std_msgs::Header header; // empty header
            header.stamp = ros::Time::now(); // time

            sensor_msgs::ImagePtr narrow_frame_ptr = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame_buffer[narrow_cam_id]).toImageMsg();
            sensor_msgs::ImagePtr wide_frame_ptr = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame_buffer[wide_cam_id]).toImageMsg();
            
            camera_system::img_pair_msg dual_input_frames;
            dual_input_frames.im_narrow = *narrow_frame_ptr;
            dual_input_frames.im_wide = *wide_frame_ptr;
            
            dual_input_frame_publisher.publish(dual_input_frames);
            // narrow_camera_frame_publisher.publish(*narrow_frame_ptr);
            // wide_camera_frame_publisher.publish(*wide_frame_ptr);
        }

    }

}

void publish_image(cv::Mat& frame, intptr_t cam_id){
    sensor_msgs::Image input_frame;
    cv_bridge::CvImage img_bridge;
    std_msgs::Header header; // empty header


    header.stamp = ros::Time::now(); // time
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame);
    img_bridge.toImageMsg(input_frame); // from cv_bridge to sensor_msgs::Image
    
    if (cam_id == narrow_cam_id){
        narrow_camera_frame_publisher.publish(input_frame);
    }
    else if (cam_id == wide_cam_id){
        wide_camera_frame_publisher.publish(input_frame);
    }
}

void visualize_image(cv::Mat& frame, intptr_t cam_id){ 
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    string windowName = "frame_"+to_string(cam_id); //Name of the window
    cv::namedWindow(windowName); // Create a window
    cv::imshow(windowName, frame);
    cv::waitKey(1);
}


class ImageHandler : public CImageEventHandler
{
public:
    CImageFormatConverter format_converter;
    CPylonImage image_converted;

    chrono::steady_clock::time_point time_old = chrono::steady_clock::now();
    chrono::steady_clock::time_point time_new = chrono::steady_clock::now();
    float duration;

    ImageHandler(){
        format_converter.OutputPixelFormat = PixelType_RGB8packed;
        format_converter.OutputBitAlignment = OutputBitAlignment_MsbAligned;
    }
    void OnImageGrabbed( CInstantCamera& camera, const CGrabResultPtr& ptrGrabResult )
    {
        intptr_t cam_id = camera.GetCameraContext();
        cv::Mat image;
        if (ptrGrabResult->GrabSucceeded()){

            format_converter.Convert(image_converted, ptrGrabResult);
            image = cv::Mat(height, width, CV_8UC3, (uint8_t *) image_converted.GetBuffer());
            
            if (should_publish){
                // publish_image(image,cam_id);
                publish_image_sync(image,cam_id);
            }
            if (should_visualize){
                visualize_image(image,cam_id);
            }
            
            time_new = chrono::steady_clock::now();
            duration = chrono::duration_cast<std::chrono::microseconds>(time_new - time_old).count();  //micro seconds
            time_old = time_new;
            // if (cam_id == narrow_cam_id){
            //     cout << "\t\t\t\t\t\tnarrow rate = " << 1000000/(duration) << endl;  //Hz

            // }
            // else{
            //     cout << "wide rate = " << 1000000/(duration) << endl;  //Hz
            // }

        }
        else{
            std::cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << std::endl;
        }
    }
};

void background_loop(CInstantCameraArray& cameras){

    uint8_t cam_count = cameras.GetSize();
    CImageEventHandler* image_handlers [cam_count];

    for (uint8_t i=0;i<cam_count;i++){
        image_handlers[i] = new ImageHandler;
        cameras[i].RegisterImageEventHandler(image_handlers[i], RegistrationMode_Append, Cleanup_Delete);
    }
    cameras.StartGrabbing(GrabStrategy_LatestImageOnly, GrabLoop_ProvidedByInstantCamera);
    // CImageEventHandler* image_handler = new ImageHandler;
    // camera.RegisterImageEventHandler( image_handler, RegistrationMode_Append, Cleanup_Delete );
    // camera.StartGrabbing( GrabStrategy_LatestImageOnly, GrabLoop_ProvidedByInstantCamera );

    try{
        while (cameras.IsGrabbing()){
            continue;
        }
    }
    catch (const GenericException& e) {
        // Error handling.
        cerr << "An exception occurred." << endl << e.GetDescription() << endl;
        cameras.StopGrabbing();
        for(uint8_t i=0;i<cam_count;i++){
            cameras[i].DeregisterImageEventHandler(image_handlers[i]);
            cameras[i].Close();
        }
        // camera.DeregisterImageEventHandler(image_handler);
        // camera.Close();
        cv::destroyAllWindows();
    }
    

    return;
}