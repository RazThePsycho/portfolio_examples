#include <cmath>
#include <iostream>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Image.h>
#include <hector_uav_msgs/EnableMotors.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Camera from Quadrator";


class SimpleMover {

    ros::NodeHandle nh;
    ros::Publisher cmd_vel_pub;
    ros::Subscriber image_sub;
    ros::ServiceClient motor_client;
    ros::Rate rate = ros::Rate(30);

    cv_bridge::CvImagePtr cv_ptr;

    double altitude_desired;

  public:

    SimpleMover() {
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        image_sub = nh.subscribe("/cam_1/camera/image", 1, &SimpleMover::camera_cb, this);
        motor_client = nh.serviceClient<hector_uav_msgs::EnableMotors>("/enable_motors");

        if (!nh.getParam("/profi2021_bachelor_solution/altitude_desired", altitude_desired)) {
            ROS_ERROR("Failed to get param '/profi2021_bachelor_solution/altitude_desired'");
        }

        cv::namedWindow(OPENCV_WINDOW);

        ros::Duration(1).sleep();       
    }                                   


    ~SimpleMover() {
        cv::destroyWindow(OPENCV_WINDOW);
    }


    void camera_cb(const sensor_msgs::Image::ConstPtr &msg) {
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        show_image(cv_ptr);
    }


    void show_image(const cv_bridge::CvImagePtr cv_ptr) {
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);
    }


    void enable_motors() {
        ros::service::waitForService("/enable_motors");
        hector_uav_msgs::EnableMotors srv;
        srv.request.enable = true;
        if (!motor_client.call(srv)) {
            ROS_ERROR("Failed to call service enable_motors");
        }
    }


    void take_off() {

        enable_motors();

        double cur_time = ros::Time::now().toSec();
        double end_time = cur_time + 3.0;
        geometry_msgs::Twist twist_msg;
        twist_msg.linear.z = 1.0;

        while ( nh.ok() && (cur_time < end_time)) {
            cmd_vel_pub.publish(twist_msg);
            ros::spinOnce();
            rate.sleep();
            cur_time = ros::Time::now().toSec();
        }
    }


    void spin() {

        take_off();

        double start_time = ros::Time::now().toSec();
        while (nh.ok()) {
            geometry_msgs::Twist twist_msg;

            double delta_time = ros::Time::now().toSec() - start_time;

            twist_msg.linear.z = 0.8 * cos(1.2 * delta_time);
            twist_msg.linear.y = 0.8 * sin(0.6 * delta_time);
            cmd_vel_pub.publish(twist_msg);

            ros::spinOnce();
            rate.sleep();
        }
    }
};


int main(int argc, char **argv) {

    ros::init(argc, argv, "simple_mover");

    SimpleMover simpleMover;
    simpleMover.spin();

  return 0;
}
