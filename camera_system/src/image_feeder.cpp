/*
Created on Wed Feb 2 2022

@author: Asitha Divisekara
*/
#include <string>
#include "ros/ros.h"  // roscpp header file
#include "sensor_msgs/Image.h"
#include <sensor_msgs/image_encodings.h>

#include "opencv2/opencv.hpp"  // opencv header file
// #include <image_transport/image_transport.h> 
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include "std_msgs/String.h"
#include <camera_system/img_pair_msg.h>
// #include <"opencv2/highgui/highgui.hpp">

// image_transport::Publisher pub;

int main(int argc, char** argv){
    ros::init(argc, argv, "image_feeder");
    ros::NodeHandle nh;

    std::string input_wide_video,input_narrow_video;
    int frame_rate,camera_count;

    nh.getParam("frame_rate", frame_rate);
    nh.getParam("camera_count", camera_count);
    nh.getParam("input_wide_video", input_wide_video);
    nh.getParam("input_narrow_video", input_narrow_video);


    ros::Publisher single_input_frame_publisher,dual_input_frame_publisher;
    ros::Publisher narrow_camera_frame_publisher;
    ros::Publisher wide_camera_frame_publisher;
    cv::VideoCapture cap_wide, cap_narrow;
    cv::Mat frame_narrow, frame_wide;
    
    ros::Rate rate(frame_rate); 

    if (camera_count == 1){
        single_input_frame_publisher = nh.advertise<sensor_msgs::Image>("/single_input_frame", 1);
        wide_camera_frame_publisher = nh.advertise<sensor_msgs::Image>("/wide_camera_frame", 1);
        cap_narrow.open(input_narrow_video); // path to the video
    }

    else if (camera_count == 2){  // wide video is published only if 2 cameras are used
        dual_input_frame_publisher = nh.advertise<camera_system::img_pair_msg>("/dual_input_frames", 1);
        narrow_camera_frame_publisher = nh.advertise<sensor_msgs::Image>("/narrow_camera_frame", 1);
        wide_camera_frame_publisher = nh.advertise<sensor_msgs::Image>("/wide_camera_frame", 1);

        cap_narrow.open(input_narrow_video); // path to the video
        cap_wide.open(input_wide_video); // path to the video
    }
    

    while (1)
    // while (cap.read(frame))
    {
        std::cout << "image feed" << std::endl;
        if (camera_count == 1){

            if(cap_narrow.read(frame_narrow)==false){
                break;
            }

            cvtColor(frame_narrow, frame_narrow, CV_BGR2RGB);
            // cv::Size s = frame_narrow.size();

            cv_bridge::CvImage img_bridge;
            sensor_msgs::Image input_frame_narrow; // >> message to be sent

            std_msgs::Header header; // empty header
            header.stamp = ros::Time::now(); // time

            img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame_narrow);
            img_bridge.toImageMsg(input_frame_narrow); // from cv_bridge to sensor_msgs::Image
            single_input_frame_publisher.publish(input_frame_narrow);
            wide_camera_frame_publisher.publish(input_frame_narrow);  // for lane detetion if only use single frame
        }

/////////////////////////////////////////////////////////////////////////////////////////
        else if (camera_count==2){
            if(cap_wide.read(frame_wide)==false || cap_narrow.read(frame_narrow)== false){
                break;
            }
            cvtColor(frame_wide, frame_wide, CV_BGR2RGB);
            cvtColor(frame_narrow, frame_narrow, CV_BGR2RGB);

            cv::Size s2 = frame_wide.size();

            std_msgs::Header header; // empty header
            header.stamp = ros::Time::now(); // time

            sensor_msgs::ImagePtr narrow_frame_ptr = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame_narrow).toImageMsg();
            sensor_msgs::ImagePtr wide_frame_ptr = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame_wide).toImageMsg();

            camera_system::img_pair_msg dual_input_frames;
            dual_input_frames.im_narrow = *narrow_frame_ptr;
            dual_input_frames.im_wide = *wide_frame_ptr;
            
            dual_input_frame_publisher.publish(dual_input_frames);
            narrow_camera_frame_publisher.publish(*narrow_frame_ptr);
            wide_camera_frame_publisher.publish(*wide_frame_ptr);

        }
////////////////////////////////////////////////////////////////////////////////////////////
        rate.sleep();

    }

    cap_narrow.release();
    if (camera_count==2){
        cap_wide.release();
    }

    return 0;
}






