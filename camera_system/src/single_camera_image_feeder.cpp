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
uint16_t narrow_AutoExposureTimeUpperLimit = 1000;
uint16_t wide_AutoExposureTimeUpperLimit = 10000;
double narrow_AutoExposureTimeUpperLimit_launch_file, wide_AutoExposureTimeUpperLimit_launch_file; // will be set by the launch file
int fps_launch_file; // will be set by the launch file
String_t PixelFormat = "YCbCr422_8" ;
double AutoGainUpperLimit = 5.0;

ros::Publisher camera_frame_publisher;

void Initialize_cam(CInstantCamera& camera);
void background_loop(CInstantCamera& camera);
void visualize_image(cv::Mat& frame);
void publish_image(cv::Mat& frame);

int main( int argc, char* argv[] )
{

    ros::init(argc, argv, "single_camera_image_feeder");
    ros::NodeHandle n;
    camera_frame_publisher = n.advertise<sensor_msgs::Image>("/single_input_frame", 100);

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

    std::cout << "wide_AutoExposureTimeUpperLimit " << wide_AutoExposureTimeUpperLimit_launch_file << " narrow_AutoExposureTimeUpperLimit " << narrow_AutoExposureTimeUpperLimit_launch_file << std::endl;

    int exitCode = 0;
    PylonInitialize();

    try
    {

        CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice() );

        camera.Open();
        Initialize_cam(camera);
        INodeMap& nodemap = camera.GetNodeMap();
        cout << "chunk enable" << camera.ChunkNodeMapsEnable.GetValue() << endl;
        camera.ChunkNodeMapsEnable.SetValue(false);
        cout << "chunk enable_2" << camera.ChunkNodeMapsEnable.GetValue() << endl;



        background_loop(camera);

        camera.Close();
        PylonTerminate();
    }
    catch (const GenericException& e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
            << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Releases all pylon resources.
    PylonTerminate();

    return exitCode;
}

void Initialize_cam(CInstantCamera& camera){
    String_t cam_name = camera.GetDeviceInfo().GetUserDefinedName();
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
    }
    else if (cam_name == "Narrow"){
        CFloatParameter(nodemap,"AutoExposureTimeUpperLimit").SetValue(narrow_AutoExposureTimeUpperLimit);
        CBooleanParameter(nodemap, "ReverseX").SetValue(true);
        CBooleanParameter(nodemap, "ReverseY").SetValue(true);

    }

}

void publish_image(cv::Mat& frame){
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    sensor_msgs::Image input_frame;

    input_frame.header.stamp = ros::Time::now();
    input_frame.height       = height;
    input_frame.width        = width;
    input_frame.encoding     = "rgb8";
    input_frame.is_bigendian = false; // idont think this is a boolean value
    input_frame.step         = 3 * width; // Full row length in bytes

    cv_bridge::CvImage img_bridge;
    std_msgs::Header header; // empty header

    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame);
    img_bridge.toImageMsg(input_frame); // from cv_bridge to sensor_msgs::Image
    // std::cout << input_frame.height << " " << input_frame.width << " " << input_frame.header.stamp <<std::endl;
    camera_frame_publisher.publish(input_frame);
}

void visualize_image(cv::Mat& frame){ 
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    string windowName = "frame"; //Name of the window
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
        // cout << "OnImageGrabbed event for device " << camera.GetDeviceInfo().GetUserDefinedName()<< endl;
        cv::Mat image;
        if (ptrGrabResult->GrabSucceeded()){

            format_converter.Convert(image_converted, ptrGrabResult);


            image = cv::Mat(height, width, CV_8UC3, (uint8_t *) image_converted.GetBuffer());
            
            if (should_publish){
                publish_image(image);
            }
            if (should_visualize){
                visualize_image(image);
            }
            time_new = chrono::steady_clock::now();
            duration = chrono::duration_cast<std::chrono::microseconds>(time_new - time_old).count();  //micro seconds
            time_old = time_new;
            // cout << camera.GetDeviceInfo().GetUserDefinedName() << " rate = " << 1000000/(duration) << endl;  //Hz

        }
        else{
            std::cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << std::endl;
        }
    }
};

void background_loop(CInstantCamera& camera){

    CImageEventHandler* image_handler = new ImageHandler;
    camera.RegisterImageEventHandler( image_handler, RegistrationMode_Append, Cleanup_Delete );
    camera.StartGrabbing( GrabStrategy_LatestImageOnly, GrabLoop_ProvidedByInstantCamera );

    try{

        while (camera.IsGrabbing()){
            continue;
        }
    }
    catch (const GenericException& e) {
        // Error handling.
        cerr << "An exception occurred." << endl << e.GetDescription() << endl;
        camera.StopGrabbing();
        camera.DeregisterImageEventHandler(image_handler);
        camera.Close();
        cv::destroyAllWindows();
    }
    

    return;
}