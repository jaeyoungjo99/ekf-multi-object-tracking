/****************************************************************************/
// Module:      ekf_multi_object_tracking_node.cpp
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

#include "ekf_multi_object_tracking_node.hpp"

EkfMultiObjectTrackingNode::EkfMultiObjectTrackingNode(){
    b_is_wgs84_reference_init_ = false;

    // Data validation
    b_is_new_lidar_objects_ = false;
    b_is_new_track_objects_ = false;

    b_is_new_motion_input_ = false;

    b_is_track_init_ = false;

    last_predicted_time_ = 0.0;

    Init();
}

EkfMultiObjectTrackingNode::~EkfMultiObjectTrackingNode() {}

void EkfMultiObjectTrackingNode::Init() {
    ROS_INFO_STREAM("[MCOT] Init");
    // Node initialization
    ros::NodeHandle nh;

    ProcessYAML();

    if (!nh.getParam("/topic_name/lidar_objects", cfg_lidar_objects_topic_)) {
        ROS_ERROR_STREAM("Failed to get param: /topic_name/lidar_objects");
        cfg_lidar_objects_topic_ = "/app/perc/lidar_objects";
    }

    std::cout<<"cfg_lidar_objects_topic_: " << cfg_lidar_objects_topic_<<std::endl;

    // Subscriber init
    s_lidar_objects_ =
            nh.subscribe(cfg_lidar_objects_topic_, 10, &EkfMultiObjectTrackingNode::CallbackBoundingBoxArray, this);

    // if (config_.input_localization == mc_mot::LocalizationType::NOVATEL)
    //     s_inspvax_ =
    //             nh.subscribe("/novatel/oem7/inspvax", 10, &EkfMultiObjectTrackingNode::CallbackNovatelINSPVAX, this);
    // if (config_.input_localization == mc_mot::LocalizationType::VEHICLE_STATE)
    //     s_vehicle_state_ =
    //             nh.subscribe("/app/loc/vehicle_state", 10, &EkfMultiObjectTrackingNode::CallbackVehicleState, this);
    // if (config_.input_localization == mc_mot::LocalizationType::CAN)
    //     s_vehicle_can_ = nh.subscribe("/bsw/vehicle_can", 10, &EkfMultiObjectTrackingNode::CallbackVehicleCAN, this);

    // p_track_objects_ = nh.advertise<autohyu_msgs::TrackObjects>("/app/perc/track_objects", 10);
    p_all_track_ = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("app/perc/jsk_all_track", 10);
    p_all_track_info_ = nh.advertise<visualization_msgs::MarkerArray>("app/perc/vis_all_track", 10);
    p_ego_stl_ = nh.advertise<visualization_msgs::Marker>("app/perc/vis_ego_stl", 10);

    // algorithm init
    mcot_algorithm_ = EkfMultiObjectTracking(config_);

    b_is_init_ = true;
    mcot_algorithm_.UpdateConfig(config_);
}

void EkfMultiObjectTrackingNode::Run() {

    if (config_.output_period_lidar == true && b_is_new_lidar_objects_ == false) {
        return;
    }
    double cur_ros_time = ros::Time::now().toSec();

    // ----- Input -----
    ros_interface::DetectObjects3D lidar_objects;
    {
        std::lock_guard<std::mutex> lock(mutex_lidar_objects_);
        lidar_objects = i_lidar_objects_;
        lidar_objects.header.stamp = cur_ros_time; // FIXME:
    }

    mc_mot::ObjectState lidar_state;
    {
        std::lock_guard<std::mutex> lock(mutex_motion_);
        lidar_state = lidar_state_;
    }

    // ----- Algorithm -----
    // Motion Update with Input Localization
    if (b_is_new_motion_input_ == true && b_is_track_init_ == true) {
        double dt = lidar_state.time_stamp - last_predicted_time_;
        mcot_algorithm_.RunPrediction(dt);
        last_predicted_time_ = lidar_state.time_stamp;

        b_is_new_motion_input_ = false;
        b_is_new_track_objects_ = true;
    }

    // Motion Update with fixed time
    if (config_.input_localization == mc_mot::LocalizationType::NONE &&
        b_is_track_init_ == true) {

        double dt = cur_ros_time - last_predicted_time_;
        mcot_algorithm_.RunPrediction(dt);
        last_predicted_time_ = cur_ros_time;

        b_is_new_track_objects_ = true;
    }

    // Measurement Update
    if (b_is_new_lidar_objects_ == true) {
        if (config_.input_localization == mc_mot::LocalizationType::NONE && b_is_track_init_ == true) {
            double dt = lidar_objects.header.stamp - last_predicted_time_;
            mcot_algorithm_.RunPrediction(dt);
            last_predicted_time_ = lidar_objects.header.stamp;
        }

        mc_mot::Meastructs meas_structs;

        if (config_.input_localization == mc_mot::LocalizationType::NONE) {
            DetectObjects2LocalMeasurements(lidar_objects, meas_structs);
        }
        else {
            if (DetectObjects2GlobMeasurements(lidar_objects, meas_structs) == false) {
                ROS_WARN_STREAM("[MCOT] CANNOT CONVERT EGO TO GLOBAL. NO LIDAR STATES");
                return;
            }
        }

        mcot_algorithm_.RunUpdate(meas_structs);

        b_is_new_lidar_objects_ = false;
        b_is_new_track_objects_ = true;

        if(b_is_track_init_ == false){
            b_is_track_init_ = true;
            last_predicted_time_ = lidar_objects.header.stamp;
        }
    }


    if (b_is_new_track_objects_ == true) {
        // ----- Output -----
        mc_mot::TrackStructs mot_track_structs = mcot_algorithm_.GetTrackResults();

        std::string o_frame_id;
        if (config_.input_localization != mc_mot::LocalizationType::NONE) {
            if (config_.output_local_coord == false) { // output world coordinate
                o_frame_id = "world";
                VisualizeTrackObjects(mot_track_structs, o_frame_id);
            }
            else { // output velodyne coordinate
                std::deque<mc_mot::ObjectState> deque_lidar_state;
                {
                    std::lock_guard<std::mutex> lock(mutex_motion_);
                    deque_lidar_state = deque_lidar_state_;
                }
                mc_mot::ObjectState synced_lidar_state =
                        GetSyncedLidarState(mot_track_structs.time_stamp, deque_lidar_state);
                ConvertTrackGlobalToLocal(mot_track_structs, synced_lidar_state);
                o_frame_id = "velodyne";
                VisualizeTrackObjects(mot_track_structs, o_frame_id);
            }
        }
        else { // no localization. output velodyne coordinate
            o_frame_id = "velodyne";
            VisualizeTrackObjects(mot_track_structs, o_frame_id);
        }
    }
}

void EkfMultiObjectTrackingNode::Publish() {
    // std::lock_guard<std::mutex> lock(mutex_lidar_objects_);
    if (b_is_new_track_objects_ == true) {
        std::cout<<"config_.visualize_mesh: "<<config_.visualize_mesh<<std::endl; // FIXME:
        std::cout<<"[MCOT] Publish Track Objects: " << o_jsk_tracked_objects_.boxes.size()<<std::endl;
        
        p_all_track_.publish(o_jsk_tracked_objects_);
        p_all_track_info_.publish(o_vis_track_info_);
        p_ego_stl_.publish(o_vis_ego_stl_);

        b_is_new_track_objects_ = false;

    }
}

void EkfMultiObjectTrackingNode::Terminate() { ROS_INFO_STREAM("[MCOT] Terminate"); }

void EkfMultiObjectTrackingNode::ProcessYAML() {
    ros::NodeHandle nh; // NodeHandle with private namespace

    // Read parameters from YAML
    int i_input_localization = 0;
    nh.getParam("configure/input_localization", i_input_localization);
    config_.input_localization =  mc_mot::LocalizationType(i_input_localization);

    std::cout<<"i_input_localization: " << i_input_localization <<std::endl;

    nh.getParam("configure/output_local_coord", config_.output_local_coord);
    nh.getParam("configure/output_period_lidar", config_.output_period_lidar);
    nh.getParam("configure/output_confirmed_track", config_.output_confirmed_track);
    nh.getParam("configure/use_predefined_ref_point", config_.use_predefined_ref_point);
    nh.getParam("configure/reference_lat", config_.reference_lat);
    nh.getParam("configure/reference_lon", config_.reference_lon);
    nh.getParam("configure/reference_height", config_.reference_height);
    nh.getParam("configure/cal_detection_individual_time", config_.cal_detection_individual_time);
    nh.getParam("configure/lidar_rotation_period", config_.lidar_rotation_period);
    nh.getParam("configure/lidar_sync_scan_start", config_.lidar_sync_scan_start);
    nh.getParam("configure/max_association_dist_m", config_.max_association_dist_m);

    int i_prediction_model = 0;
    nh.getParam("prediction_model", i_prediction_model);
    config_.prediction_model = mc_mot::PredictionModel(i_prediction_model);

    nh.getParam("configure/system_noise_std_xy_m", config_.system_noise_std_xy_m);
    nh.getParam("configure/system_noise_std_yaw_deg", config_.system_noise_std_yaw_deg);
    nh.getParam("configure/system_noise_std_vx_vy_ms", config_.system_noise_std_vx_vy_ms);
    nh.getParam("configure/system_noise_std_yaw_rate_degs", config_.system_noise_std_yaw_rate_degs);
    nh.getParam("configure/system_noise_std_ax_ay_ms2", config_.system_noise_std_ax_ay_ms2);
    nh.getParam("configure/meas_noise_std_xy_m", config_.meas_noise_std_xy_m);
    nh.getParam("configure/meas_noise_std_yaw_deg", config_.meas_noise_std_yaw_deg);
    nh.getParam("configure/dimension_filter_alpha", config_.dimension_filter_alpha);
    nh.getParam("configure/use_kinematic_model", config_.use_kinematic_model);
    nh.getParam("configure/use_yaw_rate_filtering", config_.use_yaw_rate_filtering);
    nh.getParam("configure/max_steer_deg", config_.max_steer_deg);
    nh.getParam("configure/visualize_mesh", config_.visualize_mesh);

    // Vehicle origin parameters
    int i_vehicle_origin;
    nh.getParam("vehicle_origin", i_vehicle_origin);

    std::string str_origin = (i_vehicle_origin == 0) ? "rear" : "cg";
    nh.getParam(str_origin + "_to_main_lidar/transform_xyz_m", cfg_vec_d_ego_to_lidar_xyz_m_);
    nh.getParam(str_origin + "_to_main_lidar/rotation_rpy_deg", cfg_vec_d_ego_to_lidar_rpy_deg_);

    std::cout << "[MCOT] ego_to_lidar_xyz_m: ";
    for (const auto& val : cfg_vec_d_ego_to_lidar_xyz_m_) {
        std::cout << val << " ";
    }

    std::cout << "[MCOT] ego_to_lidar_rpy_deg: ";
    for (const auto& val : cfg_vec_d_ego_to_lidar_rpy_deg_) {
        std::cout << val << " ";
    }

    if (cfg_vec_d_ego_to_lidar_xyz_m_.size() != 3 || cfg_vec_d_ego_to_lidar_rpy_deg_.size() != 3) {
        ROS_ERROR("[MCOT] Not proper calibration!");
        ros::shutdown();
    }

    std::cout << std::endl;
    ROS_WARN("[DetectObject tracking] YAML file is updated!");

    if (b_is_init_) mcot_algorithm_.UpdateConfig(config_);
}

// --------------------------------------------------

bool EkfMultiObjectTrackingNode::DetectObjects2GlobMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                                                  mc_mot::Meastructs& o_glob_lidar_measurements) {

    // Copy deque lidar state from member variable with mutex
    std::deque<mc_mot::ObjectState> deque_lidar_state;
    {
        std::lock_guard<std::mutex> lock(mutex_motion_);
        deque_lidar_state = deque_lidar_state_;
    }

    if (deque_lidar_state.size() < 3) {
        return false;
    }

    // if lidar state is order than object time, Run state prediction
    if (deque_lidar_state.back().time_stamp < lidar_objects.header.stamp) {
        int prediction_step_count = 0;
        // Run state prediction
        while (deque_lidar_state.back().time_stamp < lidar_objects.header.stamp) {
            int i_state_size = deque_lidar_state.size();
            mc_mot::ObjectState new_state =
                    PredictNextState(deque_lidar_state[i_state_size - 1], deque_lidar_state[i_state_size]);
            deque_lidar_state.push_back(new_state);
            prediction_step_count++;
        }

        ROS_WARN_STREAM("[MCOT] Predict state " << prediction_step_count << " Count.");
    }

    int i_det_size = lidar_objects.object.size();
    o_glob_lidar_measurements.meas.resize(i_det_size);
    o_glob_lidar_measurements.time_stamp = lidar_objects.header.stamp;

    for (int i = 0; i < i_det_size; i++) {
        o_glob_lidar_measurements.meas[i].id = lidar_objects.object[i].id;
        o_glob_lidar_measurements.meas[i].detection_confidence = lidar_objects.object[i].confidence_score;
        o_glob_lidar_measurements.meas[i].classification = mc_mot::ObjectClass(lidar_objects.object[i].classification);

        o_glob_lidar_measurements.meas[i].state.time_stamp = lidar_objects.object[i].state.header.stamp;
        o_glob_lidar_measurements.meas[i].state.x = lidar_objects.object[i].state.x;
        o_glob_lidar_measurements.meas[i].state.y = lidar_objects.object[i].state.y;
        o_glob_lidar_measurements.meas[i].state.z = lidar_objects.object[i].state.z;
        o_glob_lidar_measurements.meas[i].state.yaw = lidar_objects.object[i].state.yaw;

        o_glob_lidar_measurements.meas[i].dimension.height = lidar_objects.object[i].dimension.height;
        o_glob_lidar_measurements.meas[i].dimension.width = lidar_objects.object[i].dimension.width;
        o_glob_lidar_measurements.meas[i].dimension.length = lidar_objects.object[i].dimension.length;

        // 1. Measured angle based time compensation
        if (config_.cal_detection_individual_time == true) {
            AngleBasedTimeCompensation(o_glob_lidar_measurements.meas[i]);
        }

        // 2. Transform LiDAR local meas to Global coordinate
        TransformMeasLiDAR2Global(o_glob_lidar_measurements.meas[i], deque_lidar_state);
    }

    return true;
}

void EkfMultiObjectTrackingNode::DetectObjects2LocalMeasurements(ros_interface::DetectObjects3D lidar_objects,
                                                                   mc_mot::Meastructs& o_local_lidar_measurements) {

    int i_det_size = lidar_objects.object.size();
    o_local_lidar_measurements.meas.resize(i_det_size);
    o_local_lidar_measurements.time_stamp = lidar_objects.header.stamp;

    for (int i = 0; i < i_det_size; i++) {
        o_local_lidar_measurements.meas[i].id = lidar_objects.object[i].id;
        o_local_lidar_measurements.meas[i].detection_confidence = lidar_objects.object[i].confidence_score;
        o_local_lidar_measurements.meas[i].classification = mc_mot::ObjectClass(lidar_objects.object[i].classification);

        o_local_lidar_measurements.meas[i].state.time_stamp = lidar_objects.object[i].state.header.stamp;
        o_local_lidar_measurements.meas[i].state.x = lidar_objects.object[i].state.x;
        o_local_lidar_measurements.meas[i].state.y = lidar_objects.object[i].state.y;
        o_local_lidar_measurements.meas[i].state.z = lidar_objects.object[i].state.z;
        o_local_lidar_measurements.meas[i].state.yaw = lidar_objects.object[i].state.yaw;

        o_local_lidar_measurements.meas[i].dimension.height = lidar_objects.object[i].dimension.height;
        o_local_lidar_measurements.meas[i].dimension.width = lidar_objects.object[i].dimension.width;
        o_local_lidar_measurements.meas[i].dimension.length = lidar_objects.object[i].dimension.length;
    }
}

void EkfMultiObjectTrackingNode::VisualizeTrackObjects(const mc_mot::TrackStructs& track_structs,
                                                         std::string frame_id) {
    o_jsk_tracked_objects_.header.frame_id = frame_id;
    o_jsk_tracked_objects_.header.stamp = ros::Time(track_structs.time_stamp);

    o_jsk_tracked_objects_.boxes.clear();
    o_vis_track_info_.markers.clear();

    if (config_.visualize_mesh == true) {
        // Visualize EGO STL
        o_vis_ego_stl_.header.frame_id = "velodyne";
        o_vis_ego_stl_.ns = "ego_stl";
        o_vis_ego_stl_.id = 0;
        o_vis_ego_stl_.type = visualization_msgs::Marker::MESH_RESOURCE;
        o_vis_ego_stl_.action = visualization_msgs::Marker::ADD;

        o_vis_ego_stl_.mesh_resource = "package://ekf_multi_object_tracking/dae/Ioniq5.stl";

        o_vis_ego_stl_.pose.position.x = 0.0;
        o_vis_ego_stl_.pose.position.y = 0;
        o_vis_ego_stl_.pose.position.z = -cfg_vec_d_ego_to_lidar_xyz_m_[2];

        tf2::Quaternion quat;
        quat.setRPY(0, 0, M_PI); // Roll=0, Pitch=0, Yaw=180 degrees (π radians)
        quat.normalize();

        o_vis_ego_stl_.pose.orientation.x = quat.x();
        o_vis_ego_stl_.pose.orientation.y = quat.y();
        o_vis_ego_stl_.pose.orientation.z = quat.z();
        o_vis_ego_stl_.pose.orientation.w = quat.w();

        o_vis_ego_stl_.scale.x = 1.0;
        o_vis_ego_stl_.scale.y = 1.0;
        o_vis_ego_stl_.scale.z = 1.0;

        // Set color
        o_vis_ego_stl_.color.r = 1.0f;
        o_vis_ego_stl_.color.g = 1.0f;
        o_vis_ego_stl_.color.b = 1.0f;
        o_vis_ego_stl_.color.a = 1.0;
    }

    for (auto track : track_structs.track) {
        double track_vel = sqrt(track.state_vec(3) * track.state_vec(3) + track.state_vec(4) * track.state_vec(4));
        double track_acc = sqrt(track.state_vec(6) * track.state_vec(6) + track.state_vec(7) * track.state_vec(7));

        bool b_visualize_cur_track = IsVisualizeTrack(track);

        // 1. JSK Box
        if (b_visualize_cur_track == true) {
            jsk_recognition_msgs::BoundingBox o_jsk_bbox;
            o_jsk_bbox.header.frame_id = frame_id;
            o_jsk_bbox.header.stamp = ros::Time(track_structs.time_stamp);

            // o_jsk_bbox.label = mc_mot::ObjectClass(track.classification);
            o_jsk_bbox.label = track.getRepClass();
            o_jsk_bbox.dimensions.x = track.dimension.length;
            o_jsk_bbox.dimensions.y = track.dimension.width;
            o_jsk_bbox.dimensions.z = track.dimension.height;
            o_jsk_bbox.pose.position.x = track.state_vec(0);
            o_jsk_bbox.pose.position.y = track.state_vec(1);
            o_jsk_bbox.pose.position.z = track.object_z;

            // bounding box
            tf2::Quaternion myQuaternion;
            myQuaternion.setRPY(0, 0, track.state_vec(2));
            myQuaternion = myQuaternion.normalize();
            o_jsk_bbox.pose.orientation.x = myQuaternion.getX();
            o_jsk_bbox.pose.orientation.y = myQuaternion.getY();
            o_jsk_bbox.pose.orientation.z = myQuaternion.getZ();
            o_jsk_bbox.pose.orientation.w = myQuaternion.getW();

            // Visualzation of velocity
            o_jsk_bbox.value = track_vel;

            o_jsk_tracked_objects_.boxes.push_back(o_jsk_bbox);
        }

        // 2. Markerarray
        if (track.track_id != -1) {
            // 1. Track Text
            visualization_msgs::Marker vis_track_info;
            vis_track_info.header.frame_id = frame_id;
            vis_track_info.header.stamp = ros::Time(track_structs.time_stamp);

            vis_track_info.ns = "track_info";
            vis_track_info.id = track.track_id;
            vis_track_info.action = visualization_msgs::Marker::ADD;
            vis_track_info.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            // vis_track_info.lifetime = ros::Duration(0.1);
           
            std::ostringstream display_text;
            display_text << "ID (" << track.track_id << ") score: " << std::fixed << std::setprecision(2)
                         << std::round(track.detection_confidence * 100.0) / 100.0
                         << "\ncls    : " << track.getRepClass() << " prob: " << std::fixed << std::setprecision(2)
                         << std::round(track.getRepClassProb() * 100.0) / 100.0 << "\nVel    : " << std::fixed
                         << std::setprecision(3) << std::round(track_vel * 1000.0) / 1000.0 << "m/s"
                         << "\nAccel  : " << std::fixed << std::setprecision(3)
                         << std::round(track_acc * 1000.0) / 1000.0 << "m/s^2"
                         << "\nYawVel : " << std::fixed << std::setprecision(3)
                         << std::round(track.state_vec(5) * 180.0 / M_PI * 1000.0) / 1000.0 << "deg/s"
                         << "\nHscore : " << std::fixed << std::setprecision(3)
                         << std::round(track.direction_score * 1000.0) / 1000.0;

            vis_track_info.text = display_text.str();

            vis_track_info.scale.z = 0.5;

            vis_track_info.color.r = 1.0f;
            vis_track_info.color.g = 1.0f;
            vis_track_info.color.b = 1.0f;
            vis_track_info.color.a = 1.0f;

            vis_track_info.pose.position.x = track.state_vec(0);
            vis_track_info.pose.position.y = track.state_vec(1);
            vis_track_info.pose.position.z = track.object_z + 3.0;
            vis_track_info.pose.orientation = tf::createQuaternionMsgFromYaw(track.state_vec(2));

            if (b_visualize_cur_track == false) {
                vis_track_info.action = visualization_msgs::Marker::DELETE;
                vis_track_info.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(vis_track_info);

            // 2. Cov Cylinder
            visualization_msgs::Marker cov_marker;
            cov_marker.header.frame_id = frame_id;
            cov_marker.header.stamp = ros::Time(track_structs.time_stamp);
            cov_marker.ns = "covariance";
            cov_marker.id = track.track_id;
            cov_marker.action = visualization_msgs::Marker::ADD;
            cov_marker.type = visualization_msgs::Marker::CYLINDER;
            cov_marker.pose.position.x = track.state_vec(0);
            cov_marker.pose.position.y = track.state_vec(1);
            cov_marker.pose.position.z = track.object_z - track.dimension.height / 2.0;
            cov_marker.pose.orientation.x = 0.0;
            cov_marker.pose.orientation.y = 0.0;
            cov_marker.pose.orientation.z = 0.0;
            cov_marker.pose.orientation.w = 1.0;

            // Calculate eigenvalues and eigenvectors of the covariance matrix
            Eigen::Matrix2d cov_matrix;
            cov_matrix << track.state_cov(0, 0), track.state_cov(0, 1), track.state_cov(1, 0), track.state_cov(1, 1);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov_matrix);
            Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
            Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();

            // Set the size of the ellipse using eigenvalues
            cov_marker.scale.x = 2 * std::sqrt(eigenvalues[0]); // Major axis
            cov_marker.scale.y = 2 * std::sqrt(eigenvalues[1]); // Minor axis

            // std::cout<<"Scale: "<< cov_marker.scale.x<<" , "<<cov_marker.scale.y<<std::endl;
            cov_marker.scale.z = 0.1; // Height of the cylinder (set small)

            cov_marker.color.r = 0.0f;
            cov_marker.color.g = 1.0f;
            cov_marker.color.b = 0.0f;
            cov_marker.color.a = 0.3f;

            // Set the direction of the ellipse using eigenvectors
            double angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0));
            cov_marker.pose.orientation = tf::createQuaternionMsgFromYaw(angle);

            if (b_visualize_cur_track == false) {
                cov_marker.action = visualization_msgs::Marker::DELETE;
                cov_marker.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(cov_marker);

            // 3. Velocity Arrow
            visualization_msgs::Marker vel_arrow_marker;
            vel_arrow_marker.header.frame_id = frame_id;
            vel_arrow_marker.header.stamp = ros::Time(track_structs.time_stamp);
            vel_arrow_marker.ns = "velocity_arrow";
            vel_arrow_marker.id = track.track_id;
            vel_arrow_marker.type = visualization_msgs::Marker::ARROW;
            vel_arrow_marker.action = visualization_msgs::Marker::ADD;

            geometry_msgs::Point start, end;
            start.x = track.state_vec(0);
            start.y = track.state_vec(1);
            start.z = track.object_z - track.dimension.height / 2.0;

            end.x = start.x + track.state_vec(3); 
            end.y = start.y + track.state_vec(4);
            end.z = start.z;

            vel_arrow_marker.points.push_back(start);
            vel_arrow_marker.points.push_back(end);

            vel_arrow_marker.scale.x = 0.3; 
            vel_arrow_marker.scale.y = 0.5; 
            vel_arrow_marker.scale.z = 0.5; 

            vel_arrow_marker.color.r = 1.0f;
            vel_arrow_marker.color.g = 1.0f;
            vel_arrow_marker.color.b = 1.0f;
            vel_arrow_marker.color.a = 0.6f;

            vel_arrow_marker.pose.orientation.x = 0.0;
            vel_arrow_marker.pose.orientation.y = 0.0;
            vel_arrow_marker.pose.orientation.z = 0.0;
            vel_arrow_marker.pose.orientation.w = 1.0;

            if (b_visualize_cur_track == false) {
                vel_arrow_marker.action = visualization_msgs::Marker::DELETE;
                vel_arrow_marker.color.a = 0.0f;
            }

            o_vis_track_info_.markers.push_back(vel_arrow_marker);

            // Target STL 시각화
            if (config_.visualize_mesh == true) {
                visualization_msgs::Marker track_stl;
                track_stl.header.frame_id = frame_id;
                track_stl.ns = "track_mesh";
                track_stl.id = track.track_id;
                track_stl.type = visualization_msgs::Marker::MESH_RESOURCE;
                track_stl.action = visualization_msgs::Marker::ADD;

                double stl_heading = track.state_vec(2);

                track_stl.scale.x = 1.0; // .stl 파일의 스케일을 설정합니다.
                track_stl.scale.y = 1.0;
                track_stl.scale.z = 1.0;

                switch (track.getRepClass()) {
                case mc_mot::ObjectClass::REGULAR_VEHICLE:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/SimpleCar_turn.dae";
                    track_stl.scale.x = track.dimension.length / 4.6;
                    track_stl.scale.y = track.dimension.width / 2.1;
                    track_stl.scale.z = track.dimension.height / 1.59;

                    break;
                case mc_mot::ObjectClass::BUS:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/bus.stl";

                    track_stl.scale.x = track.dimension.length / 15.952;
                    track_stl.scale.y = track.dimension.width / 3.66;
                    track_stl.scale.z = track.dimension.height / 4.87;

                    break;
                case mc_mot::ObjectClass::PEDESTRIAN:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/pedestrian.stl";
                    stl_heading += M_PI / 2.0;

                    break;
                case mc_mot::ObjectClass::BICYCLE:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/moto_simple_1.stl";

                    track_stl.scale.x = track.dimension.length / 3.19;
                    track_stl.scale.y = track.dimension.width / 1.1;
                    track_stl.scale.z = track.dimension.height / 1.48;

                    break;
                default:
                    track_stl.mesh_resource = "package://ekf_multi_object_tracking/dae/SimpleCar_turn.dae";

                    track_stl.scale.x = track.dimension.length / 4.6;
                    track_stl.scale.y = track.dimension.width / 2.1;
                    track_stl.scale.z = track.dimension.height / 1.59;

                    break;
                }

                track_stl.pose.position.x = track.state_vec(0);
                track_stl.pose.position.y = track.state_vec(1);
                track_stl.pose.position.z = track.object_z - track.dimension.height / 2.0;

                tf2::Quaternion quat;
                quat.setRPY(0, 0, stl_heading); // Roll=0, Pitch=0, Yaw=180 degrees (π radians)
                // quat.setRPY(0, 0, track.state_vec(2) ); // Roll=0, Pitch=0, Yaw=180 degrees (π radians)
                quat.normalize();

                track_stl.pose.orientation.x = quat.x();
                track_stl.pose.orientation.y = quat.y();
                track_stl.pose.orientation.z = quat.z();
                track_stl.pose.orientation.w = quat.w();

                // Set color
                track_stl.color.r = 0.8f;
                track_stl.color.g = 0.8f;
                track_stl.color.b = 0.8f;
                track_stl.color.a = 1.0;

                // Tracks with low detection score are gray
                if (track.detection_confidence < 0.5) {
                    track_stl.color.r = 0.4f;
                    track_stl.color.g = 0.4f;
                    track_stl.color.b = 0.4f;
                }

                if (b_visualize_cur_track == false) {
                    track_stl.action = visualization_msgs::Marker::DELETE;
                    track_stl.color.a = 0.0f;
                }

                o_vis_track_info_.markers.push_back(track_stl);
            }
        }
    }
}

void EkfMultiObjectTrackingNode::ConvertTrackGlobalToLocal(mc_mot::TrackStructs& track_structs,
                                                             mc_mot::ObjectState synced_lidar_state) {
    double cos_yaw = std::cos(synced_lidar_state.yaw);
    double sin_yaw = std::sin(synced_lidar_state.yaw);

    double glob_lidar_vx = synced_lidar_state.v_x;
    double glob_lidar_vy = synced_lidar_state.v_y;

    for (auto& track : track_structs.track) {
        if (track.is_init == true) {
            // Position in global coordinates
            double x_global = track.state_vec(0);
            double y_global = track.state_vec(1);

            // Relative position from Lidar (convert to Lidar coordinates)
            double x_relative = x_global - synced_lidar_state.x;
            double y_relative = y_global - synced_lidar_state.y;

            // Relative position in Lidar coordinates
            double x_lidar = cos_yaw * x_relative + sin_yaw * y_relative;
            double y_lidar = -sin_yaw * x_relative + cos_yaw * y_relative;

            track.state_vec(0) = x_lidar;
            track.state_vec(1) = y_lidar;
            track.state_vec(2) = track.state_vec(2) - synced_lidar_state.yaw;

            // Correct velocity component by yaw rate
            double yaw_rate_lidar = synced_lidar_state.yaw_rate;
            double v_add_x = -yaw_rate_lidar * y_lidar;
            double v_add_y = yaw_rate_lidar * x_lidar;

            // Convert global velocity (to Lidar coordinates)
            double vx_global = track.state_vec(3);
            double vy_global = track.state_vec(4);

            double glob_rel_vx = vx_global - glob_lidar_vx;
            double glob_rel_vy = vy_global - glob_lidar_vy;
            track.state_vec(3) = cos_yaw * glob_rel_vx + sin_yaw * glob_rel_vy + v_add_x;
            track.state_vec(4) = -sin_yaw * glob_rel_vx + cos_yaw * glob_rel_vy + v_add_y;

            // Convert acceleration (to Lidar coordinates)
            double ax_global = track.state_vec(6);
            double ay_global = track.state_vec(7);

            track.state_vec(6) = ax_global - synced_lidar_state.a_x;
            track.state_vec(7) = ay_global - synced_lidar_state.a_y;

            // Convert yaw rate (Global yaw rate - Lidar yaw rate)
            track.state_vec(5) = track.state_vec(5) - yaw_rate_lidar;
        }
    }
}

bool EkfMultiObjectTrackingNode::IsVisualizeTrack(const mc_mot::TrackStruct& track) {
    bool b_visualize_cur_track = true;
    if (config_.output_confirmed_track == true && track.is_confirmed == false) b_visualize_cur_track = false;
    if (track.is_init == false) b_visualize_cur_track = false;

    return b_visualize_cur_track;
}

// ========== Utils ==========

mc_mot::ObjectState EkfMultiObjectTrackingNode::GetSyncedLidarState(
        double object_time, const std::deque<mc_mot::ObjectState>& deque_lidar_state) {
    mc_mot::ObjectState object_synced_state = deque_lidar_state.back();
    double minimum_time_diff = FLT_MAX;

    for (auto& lidar_state : deque_lidar_state) {
        if (fabs(object_time - lidar_state.time_stamp) < minimum_time_diff) {
            minimum_time_diff = fabs(object_time - lidar_state.time_stamp);
            object_synced_state = lidar_state;
        }

        if (minimum_time_diff < FLT_MIN) {
            break;
        }
    }

    return object_synced_state;
}

mc_mot::ObjectState EkfMultiObjectTrackingNode::PredictNextState(const mc_mot::ObjectState& state_t_minus_1,
                                                                   const mc_mot::ObjectState& state_t) {
    mc_mot::ObjectState state_t_plus_1;

    // Calculate time interval
    double delta_t = state_t.time_stamp - state_t_minus_1.time_stamp;

    // Predict time
    state_t_plus_1.time_stamp = state_t.time_stamp + delta_t;

    // Predict position
    state_t_plus_1.x = state_t.x + state_t.v_x * delta_t + 0.5 * state_t.a_x * delta_t * delta_t;
    state_t_plus_1.y = state_t.y + state_t.v_y * delta_t + 0.5 * state_t.a_y * delta_t * delta_t;
    state_t_plus_1.z = state_t.z + state_t.v_z * delta_t + 0.5 * state_t.a_z * delta_t * delta_t;

    // Predict velocity
    state_t_plus_1.v_x = state_t.v_x + state_t.a_x * delta_t;
    state_t_plus_1.v_y = state_t.v_y + state_t.a_y * delta_t;
    state_t_plus_1.v_z = state_t.v_z + state_t.a_z * delta_t;

    // Maintain acceleration at T time
    state_t_plus_1.a_x = state_t.a_x;
    state_t_plus_1.a_y = state_t.a_y;
    state_t_plus_1.a_z = state_t.a_z;

    // Predict angle
    state_t_plus_1.roll = state_t.roll + state_t.roll_rate * delta_t;
    state_t_plus_1.pitch = state_t.pitch + state_t.pitch_rate * delta_t;
    state_t_plus_1.yaw = state_t.yaw + state_t.yaw_rate * delta_t;

    // Predict angular velocity
    state_t_plus_1.roll_rate = state_t.roll_rate;
    state_t_plus_1.pitch_rate = state_t.pitch_rate;
    state_t_plus_1.yaw_rate = state_t.yaw_rate;

    return state_t_plus_1;
}

void EkfMultiObjectTrackingNode::TransformMeasLiDAR2Global(mc_mot::Meastruct& i_meas,
                                                             const std::deque<mc_mot::ObjectState>& deque_lidar_state) {
    mc_mot::ObjectState object_synced_state = GetSyncedLidarState(i_meas.state.time_stamp, deque_lidar_state);

    Eigen::Affine3d world_to_lidar_affine =
            CreateTransformation(object_synced_state.x, object_synced_state.y, cfg_vec_d_ego_to_lidar_xyz_m_[2], 0, 0,
                                 object_synced_state.yaw);

    Eigen::Affine3d lidar_to_object_affine =
            CreateTransformation(i_meas.state.x, i_meas.state.y, i_meas.state.z, 0, 0, i_meas.state.yaw);

    Eigen::Affine3d world_to_object_affine = world_to_lidar_affine * lidar_to_object_affine;
    Eigen::Vector3d world_to_object_translation = world_to_object_affine.translation();
    Eigen::Matrix3d world_to_object_rotation = world_to_object_affine.rotation();

    i_meas.state.x = world_to_object_translation(0);
    i_meas.state.y = world_to_object_translation(1);
    i_meas.state.z = world_to_object_translation(2);
    i_meas.state.yaw = atan2(world_to_object_rotation(1, 0), world_to_object_rotation(0, 0));
}

void EkfMultiObjectTrackingNode::AngleBasedTimeCompensation(mc_mot::Meastruct& i_meas) {
    double object_angle_ego_ccw_rad, object_angle_behind_cw_rad;
    double dt_from_start_scan_sec;

    object_angle_ego_ccw_rad = atan2(i_meas.state.y, i_meas.state.x); // -pi ~ pi
    object_angle_behind_cw_rad = M_PI - object_angle_ego_ccw_rad;     // 0 ~ 2pi

    dt_from_start_scan_sec = config_.lidar_rotation_period * (object_angle_behind_cw_rad / (2.0 * M_PI)); // 0 ~ 0.1 sec

    if (config_.lidar_sync_scan_start == true) {
        i_meas.state.time_stamp += dt_from_start_scan_sec; // Using start packet lidar
    }
    else {
        i_meas.state.time_stamp -= (config_.lidar_rotation_period - dt_from_start_scan_sec);
    }
}

Eigen::Affine3d EkfMultiObjectTrackingNode::CreateTransformation(double x, double y, double z, double roll,
                                                                   double pitch, double yaw) {
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();

    // Set translation
    transform.translation() << x, y, z;

    // Set rotation (yaw -> pitch -> roll order)
    transform.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    transform.rotate(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    transform.rotate(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));

    return transform;
}

// --------------------------------------------------

void EkfMultiObjectTrackingNode::Exec(int num_thread) {
    boost::thread main_thread(boost::bind(&EkfMultiObjectTrackingNode::MainLoop, this));

    ros::AsyncSpinner spinner(num_thread);
    spinner.start();
    ros::waitForShutdown();

    main_thread.join();
}

void EkfMultiObjectTrackingNode::MainLoop() {
    ros::Rate loop_rate(100);
    ros::Time last_log_time = ros::Time::now();
    while (ros::ok()) {

        // Run algorithm
        Run();

        // Publish topics
        Publish();

        loop_rate.sleep();
    }
}
int main(int argc, char** argv) {
    std::string node_name = "ekf_multi_object_tracking";
    ros::init(argc, argv, node_name);

    double task_period = 0.01;
    double task_rate = 1.0/task_period;
    EkfMultiObjectTrackingNode main_task;

    std::cout<<"Period: "<< task_period << " Rate: "<< task_rate<<std::endl;

    main_task.Exec(4);

    return 0;
}