/****************************************************************************/
// Module:      ekf_multi_object_tracking_node.hpp
// Description: ekf_multi_object_tracking_node
//
// Authors: Jaeyoung Jo (wodud3743@gmail.com)
// Version: 0.1
//
// Revision History
//      July 19, 2024: Jaeyoung Jo - Created.
//      Jan  08, 2025: Jaeyoung Jo - Public data type.
//      XXXX XX, 2023: XXXXXXX XX -
/****************************************************************************/

#ifndef __MULTI_CLASS_OBJECT_TRACKING_NODE_HPP__
#define __MULTI_CLASS_OBJECT_TRACKING_NODE_HPP__
#pragma once

// STD header
#include <unistd.h>
#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
// #include <Eigen/Dense>

// ROS header
#include <ros/ros.h>


#include <geometry_msgs/Point.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TwistStamped.h"

#include <std_msgs/Float32MultiArray.h>
#include <tf/LinearMath/Quaternion.h> // tf::quaternion
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

// Algorithm
#include "algorithm/ekf_multi_object_tracking.hpp"


namespace ros_bridge {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get functions
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    double GetTimeStamp(const ros::Time& stamp) {
        return (double)stamp.sec + (double)stamp.nsec * 1e-9;
    }
} // namespace ros_bridge

namespace ros_interface {
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // enum
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //     
    typedef enum {
        UNKNOWN = 0,
        CAR = 1,
        TRUCK = 2,
        PEDESTRIAN = 3,
        BICYCLE = 4,
    } ObjectClass;

    typedef enum {
        UNKNOWN_STATE = 0,
        STATIC = 1,
        DYNAMIC = 2,
    } ObjectDynamicState;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // structs
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    typedef struct{
        uint32_t    seq;
        double      stamp{0.0};
        std::string frame_id;
    } Header;

    typedef struct {
        Header header;
        float x;
        float y;
        float z;
        float roll;
        float pitch;
        float yaw;
        float v_x;
        float v_y;
        float v_z;
        float a_x;
        float a_y;
        float a_z;
        float roll_rate;
        float pitch_rate;
        float yaw_rate;
    } Object3DState;

    typedef struct {
        float length;
        float width;
        float height;
    } ObjectDimension;

    typedef struct {
        uint32_t    id;
        float       confidence_score;
        ObjectClass classification;

        ObjectDimension dimension;        
        Object3DState   state;
    } DetectObject3D;

    typedef struct {        
        Header header;
        std::vector<DetectObject3D> object;
    } DetectObjects3D;

    typedef struct {
        uint32_t            id;
        ObjectClass         classification;
        ObjectDynamicState  dynamic_state;

        ObjectDimension dimension;
        Object3DState   state;
        Object3DState   state_covariance;
    } TrackObject;

    typedef struct {
        Header header;
        std::vector<TrackObject> object;
    } TrackObjects;

} // namespace ros_interface

class EkfMultiObjectTrackingNode{
public:
    // Constructor
    explicit EkfMultiObjectTrackingNode();
    // Destructor
    virtual ~EkfMultiObjectTrackingNode();

public:
    void Init();
    void Run();
    void Publish();
    void Terminate();
    void ProcessYAML();

    void Exec(int num_thread=4);
    void MainLoop();

private:

    // Data type conversion and Local-Global Transform
    // FIXME: Rewrite this functions for your custom detection data type
    bool DetectObjects2GlobMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                        mc_mot::Meastructs& o_glob_lidar_measurements);
    void DetectObjects2LocalMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                         mc_mot::Meastructs& o_local_lidar_measurements);
    void ConvertDetectObjectToMeastruct(const ros_interface::DetectObject3D& detect_object,
                                        mc_mot::Meastruct& meas);

    void VisualizeTrackObjects(const mc_mot::TrackStructs& track_structs, std::string frame_id);
    void ConvertTrackGlobalToLocal(mc_mot::TrackStructs& track_structs, mc_mot::ObjectState synced_lidar_state);
    bool IsVisualizeTrack(const mc_mot::TrackStruct& track);

    // Util
    mc_mot::ObjectState GetSyncedLidarState(double object_time,
                                            const std::deque<mc_mot::ObjectState>& deque_lidar_state);
    mc_mot::ObjectState PredictNextState(const mc_mot::ObjectState& state_t_minus_1,
                                         const mc_mot::ObjectState& state_t);
    void TransformMeasLiDAR2Global(mc_mot::Meastruct& i_meas, const std::deque<mc_mot::ObjectState>& deque_lidar_state);
    void AngleBasedTimeCompensation(mc_mot::Meastruct& i_meas);
    Eigen::Affine3d CreateTransformation(double x, double y, double z, double roll, double pitch, double yaw);

private:

    inline void CallbackBoundingBoxArray(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_lidar_objects_);

        // i_lidar_objects_.header = msg->header;
        i_lidar_objects_.header.frame_id = msg->header.frame_id;
        str_detection_frame_id_ = msg->header.frame_id;
        i_lidar_objects_.header.seq = msg->header.seq;
        i_lidar_objects_.header.stamp = ros_bridge::GetTimeStamp( msg->header.stamp);
        i_lidar_objects_.object.clear();

        unsigned int id = 0;
        for (const auto& bbox : msg->boxes) {
            ros_interface::DetectObject3D detect_object;
            detect_object.id = id++;
            detect_object.state.header.stamp = i_lidar_objects_.header.stamp;
            detect_object.state.x = bbox.pose.position.x;
            detect_object.state.y = bbox.pose.position.y;
            detect_object.state.z = bbox.pose.position.z;

            tf::Quaternion quat;
            tf::quaternionMsgToTF(bbox.pose.orientation, quat);
            double roll, pitch, yaw;
            tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
            detect_object.state.yaw = yaw;
            
            detect_object.dimension.length = bbox.dimensions.x;
            detect_object.dimension.width = bbox.dimensions.y;
            detect_object.dimension.height = bbox.dimensions.z;

            // FIXME: Use bbox label as class.
            detect_object.classification = static_cast<ros_interface::ObjectClass>(bbox.label);
            detect_object.confidence_score = 0.9; // FIXME: Fill with acture detection confidence score
            // if confidence score is lower than 0.5, algorithm 10x measurement noise

            i_lidar_objects_.object.push_back(detect_object);
        }

        std::cout<<"Callback BoundingBoxArray: " << i_lidar_objects_.object.size() << " objects" << std::endl;
        b_is_new_lidar_objects_ = true;
    }

    inline void CallbackOdometry(const nav_msgs::Odometry::ConstPtr& msg) {
        if (config_.input_localization != mc_mot::LocalizationType::ODOMETRY) return;

        std::lock_guard<std::mutex> lock(mutex_motion_);

        double odom_time = msg->header.stamp.toSec();

        double odom_vx = msg->twist.twist.linear.x;
        double odom_vy = msg->twist.twist.linear.y;
        double odom_yaw_rate = msg->twist.twist.angular.z;

        if (b_can_dr_init_ == false) {
            d_last_dr_time_ = odom_time;
            b_can_dr_init_ = true;
            return;
        }

        double dt = odom_time - d_last_dr_time_;

        double x_local = odom_vx * dt;
        double y_local = odom_vy * dt;

        motion_dr_state_(2) += odom_yaw_rate * dt;

        double sy = sin(motion_dr_state_(2));
        double cy = cos(motion_dr_state_(2));

        motion_dr_state_(0) += x_local * cy - y_local * sy;
        motion_dr_state_(1) += x_local * sy + y_local * cy;
        

        double vx_global = odom_vx * cy - odom_vy * sy;
        double vy_global = odom_vx * sy + odom_vy * cy;

        // ============================================================================================
        // Input generation
        // Position (ENU)
        double lidar_x = 0.0, lidar_y = 0.0, lidar_z = 0.0; // m

        lidar_x = motion_dr_state_(0) + cfg_vec_d_ego_to_lidar_xyz_m_[0] * cy;
        lidar_y = motion_dr_state_(1) + cfg_vec_d_ego_to_lidar_xyz_m_[0] * sy;

        // ----- Lidar transform matrix

        // This motion time do nat has to be synced with detection. Only used as delta time.
        lidar_state_.time_stamp = odom_time;
        lidar_state_.x = lidar_x;
        lidar_state_.y = lidar_y;
        lidar_state_.yaw = motion_dr_state_(2) + cfg_vec_d_ego_to_lidar_rpy_deg_[2] * M_PI / 180.0;
        lidar_state_.yaw_rate = odom_yaw_rate;

        lidar_state_.v_x = vx_global;
        lidar_state_.v_y = vy_global;

        // lidar_state_.a_x = ax_global;
        // lidar_state_.a_y = ay_global;

        deque_lidar_state_.push_back(lidar_state_);

        while (deque_lidar_state_.size() > 1000) {
            deque_lidar_state_.pop_front();
        }

        b_is_new_motion_input_ = true;

        d_last_dr_time_ = odom_time;
    }

    // Variables
private:
    // ROS
    ros::Subscriber s_lidar_objects_;
    ros::Subscriber s_odometry_;
    ros::Publisher p_track_objects_;

    ros::Publisher p_all_track_;
    ros::Publisher p_all_track_info_;
    ros::Publisher p_ego_stl_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::TransformBroadcaster tf_cam_broadcaster_;

    // Mutex
    std::mutex mutex_lidar_objects_;
    std::mutex mutex_motion_;

    // I/O Msg
    ros_interface::DetectObjects3D i_lidar_objects_;
    std::string str_detection_frame_id_ = "";

    jsk_recognition_msgs::BoundingBoxArray o_jsk_tracked_objects_;
    visualization_msgs::MarkerArray o_vis_track_info_;
    visualization_msgs::Marker o_vis_ego_stl_;

    // Main variables
    bool b_is_new_lidar_objects_;
    bool b_is_new_motion_input_;
    bool b_is_new_track_objects_;

    bool b_is_track_init_;

    Eigen::Vector3d motion_dr_state_{Eigen::Vector3d::Zero()};
    bool b_can_dr_init_{false};
    double d_last_dr_time_{0.0};

    mc_mot::ObjectState lidar_state_;
    mc_mot::ObjectState last_lidar_state_;

    double last_predicted_time_;

    std::deque<mc_mot::ObjectState> deque_lidar_state_;

    // lidar pose from ego vehicle (rear center)
    std::vector<double> cfg_vec_d_ego_to_lidar_xyz_m_;
    std::vector<double> cfg_vec_d_ego_to_lidar_rpy_deg_;

    // Config
    MultiClassObjectTrackingConfig config_;

    std::string cfg_lidar_objects_topic_ = "";
    std::string cfg_odometry_topic_ = "";

    // Algorithm

    EkfMultiObjectTracking mcot_algorithm_;
    bool b_is_init_{false};
};

#endif // __MULTI_CLASS_OBJECT_TRACKING_NODE_HPP__