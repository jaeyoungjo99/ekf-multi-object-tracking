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
        MOTORCYCLE = 3,
        PEDESTRIAN = 4,
        BARRIER = 5,
        TRAFFIC_LIGHT = 6,
        TRAFFIC_SIGN = 7,
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
    bool DetectObjects2GlobMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                        mc_mot::Meastructs& o_glob_lidar_measurements);
    void DetectObjects2LocalMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                         mc_mot::Meastructs& o_local_lidar_measurements);

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

        // 예시: BoundingBoxArray 메시지를 i_lidar_objects_로 변환하는 로직
        // i_lidar_objects_.header = msg->header;
        i_lidar_objects_.header.frame_id = msg->header.frame_id;
        i_lidar_objects_.header.seq = msg->header.seq;
        i_lidar_objects_.header.stamp = ros_bridge::GetTimeStamp( msg->header.stamp);
        i_lidar_objects_.object.clear();

        unsigned int id = 0;
        for (const auto& bbox : msg->boxes) {
            ros_interface::DetectObject3D detect_object;
            detect_object.id = id++;
            detect_object.state.x = bbox.pose.position.x;
            detect_object.state.y = bbox.pose.position.y;
            detect_object.state.z = bbox.pose.position.z;
            detect_object.dimension.length = bbox.dimensions.x;
            detect_object.dimension.width = bbox.dimensions.y;
            detect_object.dimension.height = bbox.dimensions.z;

            detect_object.classification = static_cast<ros_interface::ObjectClass>(bbox.label);

            i_lidar_objects_.object.push_back(detect_object);
        }

        std::cout<<"[MCOT] Callback BoundingBoxArray: " << i_lidar_objects_.object.size() << " objects" << std::endl;
        b_is_new_lidar_objects_ = true;
    }

    // inline void CallbackNovatelINSPVAX(const novatel_oem7_msgs::INSPVAX::ConstPtr& msg) {
    //     if (config_.input_localization != mc_mot::LocalizationType::NOVATEL) return;

    //     std::lock_guard<std::mutex> lock(mutex_motion_);

    //     i_novatel_inspvax_ = ros_bridge::GetNovatelInspvax(*msg);

    //     // Initialize reference point
    //     if (b_is_wgs84_reference_init_ == false) {
    //         if (config_.use_predefined_ref_point == false) {
    //             wgs84_reference_point_.lat = i_novatel_inspvax_.latitude;
    //             wgs84_reference_point_.lon = i_novatel_inspvax_.longitude;
    //             wgs84_reference_point_.ele = i_novatel_inspvax_.height;
    //         }
    //         else {
    //             wgs84_reference_point_.lat = config_.reference_lat;
    //             wgs84_reference_point_.lon = config_.reference_lon;
    //             wgs84_reference_point_.ele = config_.reference_height;
    //         }

    //         b_is_wgs84_reference_init_ = true;
    //     }

    //     lanelet::GPSPoint v_gps_point;
    //     v_gps_point.lat = i_novatel_inspvax_.latitude;
    //     v_gps_point.lon = i_novatel_inspvax_.longitude;
    //     v_gps_point.ele = i_novatel_inspvax_.height;

    //     lanelet::projection::LocalCartesianProjector v_projector_(
    //             lanelet::Origin({wgs84_reference_point_.lat, wgs84_reference_point_.lon}));

    //     // Ego frame xyz
    //     lanelet::BasicPoint3d cartesian_projpos = v_projector_.forward(v_gps_point);

    //     // ============================================================================================
    //     // Input generation
    //     // Position (ENU)
    //     double lidar_x = 0.0, lidar_y = 0.0, lidar_z = 0.0; // m
    //     // Attitude
    //     double novatel_roll = 0.0, novatel_pitch = 0.0, novatel_yaw = 0.0; // rad

    //     novatel_roll = i_novatel_inspvax_.roll * ros_interface::DEG2RAD;
    //     novatel_pitch = (-i_novatel_inspvax_.pitch) * ros_interface::DEG2RAD;
    //     novatel_yaw = (90.0 - i_novatel_inspvax_.azimuth) * ros_interface::DEG2RAD;

    //     double sy = sin(novatel_yaw);
    //     double cy = cos(novatel_yaw);
    //     lidar_x = cartesian_projpos.x() + cfg_vec_d_ego_to_lidar_xyz_m_[0] * cy;
    //     lidar_y = cartesian_projpos.y() + cfg_vec_d_ego_to_lidar_xyz_m_[0] * sy;
    //     lidar_z = i_novatel_inspvax_.height - wgs84_reference_point_.ele;

    //     // ----- Lidar transform matrix

    //     // This motion time do nat has to be synced with detection. Only used as delta time.
    //     lidar_state_.time_stamp = i_novatel_inspvax_.header.stamp;
    //     lidar_state_.x = lidar_x;
    //     lidar_state_.y = lidar_y;
    //     lidar_state_.yaw = novatel_yaw + cfg_vec_d_ego_to_lidar_rpy_deg_[2] * M_PI / 180.0;

    //     lidar_state_.v_x = i_novatel_inspvax_.east_velocity;
    //     lidar_state_.v_y = i_novatel_inspvax_.north_velocity;

    //     // TODO: INSPVAX 에는 yaw rate, a_x, a_y 가 없음
    //     lidar_state_.yaw_rate = 0.0;
    //     lidar_state_.a_x = 0.0;
    //     lidar_state_.a_y = 0.0;

    //     deque_lidar_state_.push_back(lidar_state_);

    //     while (deque_lidar_state_.size() > 1000) {
    //         deque_lidar_state_.pop_front();
    //     }

    //     b_is_new_motion_input_ = true;
    // }

    // Variables
private:
    // ROS
    ros::Subscriber s_lidar_objects_;
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

    jsk_recognition_msgs::BoundingBoxArray o_jsk_tracked_objects_;
    visualization_msgs::MarkerArray o_vis_track_info_;
    visualization_msgs::Marker o_vis_ego_stl_;

    // Main variables
    bool b_is_new_lidar_objects_;
    bool b_is_new_motion_input_;
    bool b_is_new_track_objects_;

    bool b_is_track_init_;

    // lanelet::GPSPoint wgs84_reference_point_;
    bool b_is_wgs84_reference_init_;

    Eigen::Vector3d can_dr_state_{Eigen::Vector3d::Zero()};
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

    // Algorithm

    EkfMultiObjectTracking mcot_algorithm_;
    bool b_is_init_{false};
};

#endif // __MULTI_CLASS_OBJECT_TRACKING_NODE_HPP__