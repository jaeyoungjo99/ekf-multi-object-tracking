# ekf-multi-object-tracking
EKF-based Multi Object Tracking

![ekf_multi_object_tracking](docs/tracking_example.gif)

## Features

The `ekf_multi_object_tracking` package supports the following features:

- **Low-Frequency Detection to High-Frequency Tracking**: Converts low-frequency detection data into high-frequency tracking data, ensuring smooth and accurate object tracking.

- **Multiple Localization Inputs (TODO)**: Supports various localization input sources, allowing flexibility in integrating different localization systems.

- **Multiple Prediction Models**: Offers a selection of prediction models, including Constant Velocity (CV), Constant Turn Rate and Velocity (CTRV), Constant Acceleration (CA), and Constant Turn Rate and Acceleration (CTRA), to suit different tracking scenarios.

- **Individual Time Consideration for Rotating LiDAR**: Takes into account the individual detection times of objects from rotating LiDAR sensors, improving the accuracy of motion compensation and tracking.

- **Vehicle Heading Correction**: Provides functionality to correct the vehicle's heading direction, enhancing the accuracy of the tracking system.

- **Class Correction**: Supports class correction features, allowing for improved classification accuracy of tracked objects.


These features make the package versatile and adaptable to a wide range of multi-object tracking applications.

## How to use
### 1. Install ROS messages and other libraries
* for Ubuntu 20.04 (noetic)
```bash
sudo apt install ros-noetic-jsk-rviz-plugins
```

### 2. Install Code
```bash
cd ~/catkin_ws/src
git clone https://github.com/jaeyoungjo99/ekf-multi-object-tracking.git
cd ..
catkin_make
```

### 3. Launch Code
```bash
source devel/setup.bash
roslaunch ekf_multi_object_tracking ekf_multi_class_object_tracking.launch 
```


## Configuration

The behavior of the `ekf_multi_object_tracking` node can be customized using the `config/config.yaml` file. Below is a description of the key configuration parameters:

### Topic Names
<!-- - **vehicle_state**: Topic for vehicle state data. Default: `/app/loc/vehicle_state` -->
- **lidar_objects**: Topic for LiDAR object data. Topic type is **jsk_recognition_msgs::BoundingBoxArray**. Default: `/hmi/perc/jsk_pillar_objects`

### Configuration Options
- **input_localization**: Selects the input localization source.
  - `0`: None
  - `1`: nav_msgs::Odometry
  - `2`: NavSatFix (TODO)
- **global_coord_track**: Whether the tracker is performed in global coordinates
  - `true`: Global coordinate tracker
  - `false`: Local coordinate tracker
- **output_local_coord**: Determines if the output is in local coordinates.
  - `true`: Output local coordinate
  - `false`: Output global coordinate
- **output_period_lidar**: Synchronizes output with LiDAR.
  - `true`: LiDAR synced output
  - `false`: Motion synced output
- **output_confirmed_track**: Outputs only confirmed tracks if set to `true`.
- **use_predefined_ref_point**: Uses a predefined reference point if set to `true`.
- **reference_lat, reference_lon, reference_height**: Predefined reference point coordinates.
- **cal_detection_individual_time**: Enables LiDAR motion compensation.
- **lidar_rotation_period**: Sets the LiDAR rotation period.
- **lidar_sync_scan_start**: Synchronizes LiDAR time to scan start if set to `true`.
- **max_association_dist_m**: Maximum distance for track association.
- **prediction_model**: Selects the prediction model.
  - `0`: CV
  - `1`: CTRV
  - `2`: CA
  - `3`: CTRA
- **system_noise_std_xy_m, system_noise_std_yaw_deg, etc.**: System noise parameters.
- **meas_noise_std_xy_m, meas_noise_std_yaw_deg**: Measurement noise parameters.
- **dimension_filter_alpha**: Filtering parameter for object dimensions.
- **use_kinematic_model**: Aligns velocity direction to heading if set to `true`.
- **use_yaw_rate_filtering**: Restricts yaw rate based on velocity.
- **max_steer_deg**: Maximum steering angle.
- **visualize_mesh**: Enables mesh visualization if set to `true`.

### Vehicle Origin
- **vehicle_origin**: Defines the vehicle origin.
  - `0`: Rear Axle
  - `1`: C.G.

### Transformations
- **rear_to_main_lidar**: Transformation from rear axle to main LiDAR.
  - `parent_frame_id`: Parent frame ID
  - `child_frame_id`: Child frame ID
  - `transform_xyz_m`: Translation in meters
  - `rotation_rpy_deg`: Rotation in degrees
- **cg_to_main_lidar**: Transformation from C.G. to main LiDAR.

Users can modify these parameters in the `config/config.yaml` file to suit their specific requirements.
