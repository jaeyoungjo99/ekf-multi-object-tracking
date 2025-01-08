/****************************************************************************/
// Module:      ekf_multi_object_tracking.cpp
// Description: ekf_multi_object_tracking
//
// Authors: Jaeyoung Jo (wodud3743@gmail.com)
// Version: 0.1
//
// Revision History
//      July 19, 2024: Jaeyoung Jo - Created.
//      Jan  08, 2025: Jaeyoung Jo - Public data type.
//      XXXX XX, 2023: XXXXXXX XX -
/****************************************************************************/

#include "algorithm/ekf_multi_object_tracking.hpp"

/*
    Predict all track in same dt_sec
*/
void EkfMultiObjectTracking::RunPrediction(double dt_sec) {
    for (int i = 0; i < MAX_TRACKS; i++) {
        if (all_tracks_[i].is_init == true) {
            PredictTrack(all_tracks_[i], dt_sec);
        }
    }
    recent_timestamp_ += dt_sec;
}

void EkfMultiObjectTracking::RunUpdate(const mc_mot::Meastructs &measurements) {
    // std::cout << "[MCOT Algo] Run" << std::endl;

    recent_timestamp_ = measurements.time_stamp;

    int i_meas_num = measurements.meas.size();
    Eigen::MatrixXd cost_matrix(i_meas_num, MAX_TRACKS);
    cost_matrix.setConstant(std::numeric_limits<double>::max());

    // 모든 트랙의 is_associated 플래그를 초기화
    for (auto &track : all_tracks_) {
        track.is_associated = false;
    }

    double l2_distance, maha_distance;
    // 비용 행렬 계산 행: measurements, 열: tracks
    for (int meas_idx = 0; meas_idx < i_meas_num; ++meas_idx) {
        for (int track_idx = 0; track_idx < MAX_TRACKS; ++track_idx) {
            l2_distance = CalculateDistance(measurements.meas[meas_idx].state, all_tracks_[track_idx].state_vec);
            maha_distance = CalculateMahalanobisDistance(measurements.meas[meas_idx].state, all_tracks_[track_idx]);
            if (all_tracks_[track_idx].is_init == false || l2_distance > config_.max_association_dist_m)
                maha_distance = 1000.0; // 초기화 안된 track과는 dist 늘려서 pair 안되게

            if ((measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN &&
                 all_tracks_[track_idx].getRepClass() != mc_mot::ObjectClass::PEDESTRIAN) ||
                (all_tracks_[track_idx].getRepClass() == mc_mot::ObjectClass::PEDESTRIAN &&
                 measurements.meas[meas_idx].classification != mc_mot::ObjectClass::PEDESTRIAN)) {
                maha_distance = 1000.0; // Pedestrian 끼리만 association
            }

            if (measurements.meas[meas_idx].classification == mc_mot::ObjectClass::PEDESTRIAN && l2_distance > 2.0) {
                maha_distance = 1000.0; // Pedestrian은 association 거리 짧게
            }

            cost_matrix(meas_idx, track_idx) = maha_distance;
        }
    }

    std::vector<int> assignment; // i_meas_num 으로 초기화 됨. 각 원소에 asociation 된 track idx 저장
    std::vector<int> assignment_track;
    MatchPairs(cost_matrix, assignment, assignment_track);

    int init_count = 0;
    // 매칭 결과를 기반으로 Association 된 track update, 새 track 추가
    for (int meas_idx = 0; meas_idx < i_meas_num; ++meas_idx) {
        int track_idx = assignment[meas_idx];

        // 측정값 중에 연관이 되었고, 연관된 Track이 초기화가 되었다면
        if (track_idx != -1 && all_tracks_[track_idx].is_init == true) {
            UpdateTrack(all_tracks_[track_idx], measurements.meas[meas_idx]);

            // Valid한 Detection이 2개 이상은 있어야 KF 수행 가능
            if (all_tracks_[track_idx].age >= 3 && all_tracks_[track_idx].countDetectionNum() >= 2) {
                all_tracks_[track_idx].is_confirmed = true;
            }
        }
        else {
            // 측정값 중에 Track에 없는 신규 Meas. Track 추가
            mc_mot::TrackStruct new_track;

            InitTrack(new_track, measurements.meas[meas_idx]);

            all_tracks_[cur_track_id_] = new_track;

            UpdateTrackId(); // cur_track_id_ 1 추가
            init_count++;
        }
    }

    // Association 안된 track에 대한 정보 갱신.
    int i_deleted_num = 0;
    for (auto &track : all_tracks_) {
        if (track.is_associated == false) {
            track.age++;
            track.updateDetectionCount(false);

            // 오래된 Track reset
            if (track.is_init == true && track.isOutdated() == true) {
                track.reset();
                i_deleted_num++;
            }
            else if (sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) +
                          track.state_vec(S_VY) * track.state_vec(S_VY)) > MAX_TRACK_VEL) {
                // 과도한 속도의 Track reset
                track.reset();
                i_deleted_num++;
            }
            else if (sqrt(track.state_vec(S_AX) * track.state_vec(S_AX) +
                          track.state_vec(S_AY) * track.state_vec(S_AY)) > MAX_TRACK_ACC) {
                // 과도한 가속도의 Track reset
                track.reset();
                i_deleted_num++;
            }
        }
    }

    // FIXME:  ======= For debugging =============
    int asso_track_num = 0;
    int init_track_num = 0;
    int confirmed_track_num = 0;
    for (const auto &track : all_tracks_) {
        if (track.is_associated) asso_track_num++;
        if (track.is_init) init_track_num++;
        if (track.is_confirmed) confirmed_track_num++;
    }
    std::cout << "[RunUpdate] Detection: " << i_meas_num << " New: " << init_count << " Deleted: " << i_deleted_num << std::endl;
    std::cout << "[RunUpdate] Asso: " << asso_track_num << " Inited: " << init_track_num << " Confirmed: " << confirmed_track_num
              << std::endl;

    // ======= For debugging =============
}

mc_mot::TrackStructs EkfMultiObjectTracking::GetTrackResults() {
    // std::cout << "[MCOT Algo] GetTrackResults" << std::endl;

    mc_mot::TrackStructs o_track_results;

    o_track_results.time_stamp = recent_timestamp_;

    for (auto &track : all_tracks_) {
        mc_mot::TrackStruct o_track;
        o_track = track;
        double track_vel = sqrt(o_track.state_vec(S_VX) * o_track.state_vec(S_VX) +
                                o_track.state_vec(S_VY) * o_track.state_vec(S_VY));
        if (track_vel < 1.0 && o_track.classification != mc_mot::ObjectClass::PEDESTRIAN) {
            o_track.state_vec(S_VX) = 0.0;       // vx
            o_track.state_vec(S_VY) = 0.0;       // vy
            o_track.state_vec(S_YAW_RATE) = 0.0; // yawrate
            o_track.state_vec(S_AX) = 0.0;       // ax
            o_track.state_vec(S_AY) = 0.0;       // ay
        }
        o_track_results.track.push_back(o_track);
    }

    // std::cout << "[MCOT Algo] GetTrackResults Num: " << o_track_results.track.size() << std::endl;

    return o_track_results;
}

void EkfMultiObjectTracking::UpdateConfig(const MultiClassObjectTrackingConfig config) {
    std::cout << "[MCOT Algo] UpdateConfig !" << std::endl;
    config_ = config;
    UpdateMatrix();
}

// Utils

void EkfMultiObjectTracking::PredictTrack(mc_mot::TrackStruct &track, double dt) {
    double track_vel =
            sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) + track.state_vec(S_VY) * track.state_vec(S_VY));

    // 차량 헤딩 방향의 속도만 남김
    if (config_.use_kinematic_model == true) {
        double heading_align_vel =
                track.state_vec(S_VX) * cos(track.state_vec[2]) + track.state_vec(S_VY) * sin(track.state_vec[2]);
        double heading_align_vx = heading_align_vel * cos(track.state_vec[2]);
        double heading_align_vy = heading_align_vel * sin(track.state_vec[2]);

        track.state_vec(S_VX) = heading_align_vx;
        track.state_vec(S_VY) = heading_align_vy;
    }

    // 상태 전이 행렬 (자코비안)
    Eigen::Matrix8_8d F = Eigen::Matrix8_8d::Identity();

    if (track.getRepClass() == mc_mot::ObjectClass::PEDESTRIAN) {
        F(S_X, S_VX) = dt; // x' = x + vx * dt
        F(S_Y, S_VY) = dt; // y' = y + vy * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CV) {
        F(S_X, S_VX) = dt;         // x' = x + vx * dt
        F(S_Y, S_VY) = dt;         // y' = y + vy * dt
        F(S_YAW, S_YAW_RATE) = dt; // yaw' = yaw + yaw_rate * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CTRV) {
        double delta_theta = track.state_vec(S_YAW_RATE) * dt;

        // 회전 행렬 계산
        double cos_del_theta = std::cos(delta_theta);
        double sin_del_theta = std::sin(delta_theta);

        double vx_rotated = cos_del_theta * track.state_vec(S_VX) - sin_del_theta * track.state_vec(S_VY);
        double vy_rotated = sin_del_theta * track.state_vec(S_VX) + cos_del_theta * track.state_vec(S_VY);

        track.state_vec(S_VX) = vx_rotated;
        track.state_vec(S_VY) = vy_rotated;

        F(S_X, S_VX) = dt;         // x' = x + vx * dt
        F(S_Y, S_VY) = dt;         // y' = y + vy * dt
        F(S_YAW, S_YAW_RATE) = dt; // yaw' = yaw + yaw_rate * dt
    }
    else if (config_.prediction_model == mc_mot::PredictionModel::CA) {
        F(S_X, S_VX) = dt;            // x' = x + vx * dt
        F(S_X, S_AX) = 0.5 * dt * dt; // x' = x + 0.5 * ax * dt^2
        F(S_Y, S_VY) = dt;            // y' = y + vy * dt
        F(S_Y, S_AY) = 0.5 * dt * dt; // y' = y + 0.5 * ay * dt^2
        F(S_YAW, S_YAW_RATE) = dt;    // yaw' = yaw + yaw_rate * dt
        F(S_VX, S_AX) = dt;           // vx' = vx + ax * dt
        F(S_VY, S_AY) = dt;           // vy' = vy + ay * dt
    }
    else { // CTRA
        double delta_theta = track.state_vec(S_YAW_RATE) * dt;

        // 회전 행렬 계산
        double cos_del_theta = std::cos(delta_theta);
        double sin_del_theta = std::sin(delta_theta);

        double vx_rotated = cos_del_theta * track.state_vec(S_VX) - sin_del_theta * track.state_vec(S_VY);
        double vy_rotated = sin_del_theta * track.state_vec(S_VX) + cos_del_theta * track.state_vec(S_VY);

        track.state_vec(S_VX) = vx_rotated;
        track.state_vec(S_VY) = vy_rotated;

        F(S_X, S_VX) = dt;            // x' = x + vx * dt
        F(S_X, S_AX) = 0.5 * dt * dt; // x' = x + 0.5 * ax * dt^2
        F(S_Y, S_VY) = dt;            // y' = y + vy * dt
        F(S_Y, S_AY) = 0.5 * dt * dt; // y' = y + 0.5 * ay * dt^2
        F(S_YAW, S_YAW_RATE) = dt;    // yaw' = yaw + yaw_rate * dt
        F(S_VX, S_AX) = dt;           // vx' = vx + ax * dt
        F(S_VY, S_AY) = dt;           // vy' = vy + ay * dt
    }

    // 저속에선 Yaw Prediction 끄기
    if (track_vel < 3.0) {
        F(S_YAW, S_YAW_RATE) = 0.0;
    }

    // 진행 방향으로 더 큰 공분산을 추가하기 위해 방향 행렬을 추가
    Eigen::Matrix8_8d Q = Q_;

    // 속도 방향으로 더 큰 공분산 추가
    double angle = atan2(track.state_vec(S_VY), track.state_vec(S_VX));
    Eigen::Matrix2d rot_mat;
    rot_mat << cos(angle), -sin(angle), sin(angle), cos(angle);

    Eigen::Matrix2d direction_cov;
    direction_cov << std::max(track_vel * 10, 1.0), 0.0, // 진행 방향으로 더 큰 공분산
            0.0, 1.0;                                    // 진행 방향에 수직인 방향의 공분산

    Eigen::Matrix2d Q_skew = Q_.block<2, 2>(S_X, S_X).cwiseProduct(direction_cov);
    Eigen::Matrix2d Q_skew_rot = rot_mat * Q_skew * rot_mat.transpose();

    Q.block<2, 2>(0, 0) = Q_skew_rot;

    // 상태 벡터 예측
    track.state_vec = F * track.state_vec;
    // 공분산 행렬 예측
    track.state_cov = F * track.state_cov * F.transpose() + Q;

    track.update_time += dt;
}

void EkfMultiObjectTracking::UpdateTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement) {
    Eigen::Vector3d measurement_vec;
    measurement_vec << measurement.state.x, measurement.state.y, measurement.state.yaw;

    // 칼만 이득 계산
    Eigen::Matrix3d R = R_;

    // 측정 헤딩 <-> 트랙 헤딩 내적 (-1.0 ~ 1.0)
    // -1.0: 반대 방향, 1.0: 같은 방향
    double meas_track_yaw_inner = CalculateYawDotProduct(measurement_vec(2), track.state_vec(S_YAW));

    // 트랙 속도
    double track_vel =
            sqrt(track.state_vec(S_VX) * track.state_vec(S_VX) + track.state_vec(S_VY) * track.state_vec(S_VY));

    // 측정 헤딩과 track 헤딩의 각도 비교를 통해 direction score 업데이트
    track.direction_score = config_.dimension_filter_alpha * meas_track_yaw_inner +
                            (1.0 - config_.dimension_filter_alpha) * track.direction_score;

    if (meas_track_yaw_inner < -cos(M_PI / 4.0)) {
        // track의 헤딩이 맞지 않은 상태. track 헤딩 뒤집고, 다시 Direction score 0.5로.
        if (track.direction_score < 0) {
            track.state_vec(S_YAW) += M_PI;
            track.direction_score = 0.5;
        }
        else {
            // track의 헤딩은 맞으나 meas heading이 잘못된 상태. meas 헤딩 뒤집기.
            measurement_vec(2) += M_PI;
        }
    }

    // Detection confidence가 낮으면 측정 불확실성 높임
    if (measurement.detection_confidence < 0.5) {
        R = 10.0 * R;
    }

    Eigen::Matrix3d S = H_ * track.state_cov * H_.transpose() + R;
    Eigen::Matrix8_3d K = track.state_cov * H_.transpose() * S.inverse();

    // Yaw normalization
    while (measurement_vec(2) - track.state_vec(S_YAW) > M_PI) {
        measurement_vec(2) -= 2.0 * M_PI;
    }
    while (measurement_vec(2) - track.state_vec(S_YAW) < -M_PI) {
        measurement_vec(2) += 2.0 * M_PI;
    }

    // 상태 벡터 업데이트
    track.state_vec += K * (measurement_vec - H_ * track.state_vec);

    // 공분산 행렬 업데이트
    Eigen::Matrix8_8d I = Eigen::Matrix8_8d::Identity();
    track.state_cov = (I - K * H_) * track.state_cov;

    // Track 속성 업데이트
    track.is_associated = true;
    track.updateDetectionCount(true);
    track.update_time = measurement.state.time_stamp;
    track.detection_confidence = config_.dimension_filter_alpha * measurement.detection_confidence +
                                 (1.0 - config_.dimension_filter_alpha) * track.detection_confidence;
    track.age++;

    // Dimension Alpha filtering
    track.dimension.length = config_.dimension_filter_alpha * measurement.dimension.length +
                             (1.0 - config_.dimension_filter_alpha) * track.dimension.length;
    track.dimension.width = config_.dimension_filter_alpha * measurement.dimension.width +
                            (1.0 - config_.dimension_filter_alpha) * track.dimension.width;
    track.dimension.height = config_.dimension_filter_alpha * measurement.dimension.height +
                             (1.0 - config_.dimension_filter_alpha) * track.dimension.height;

    track.object_z = config_.dimension_filter_alpha * measurement.state.z +
                     (1.0 - config_.dimension_filter_alpha) * track.object_z;

    // Class Filtering
    track.updateClassScore(mc_mot::ObjectClass(measurement.classification));

    // Yaw rate Filtering based on Kinematic Model
    if (config_.use_yaw_rate_filtering && (track.getRepClass() == mc_mot::ObjectClass::REGULAR_VEHICLE ||
                                           track.getRepClass() == mc_mot::ObjectClass::BUS)) {
        double vel_heading_dot =
                CalculateYawDotProduct(atan2(track.state_vec(S_VY), track.state_vec(S_VX)), track.state_vec(S_YAW));
        double vel_heading_cross =
                CalculateYawCrossProduct(track.state_vec(S_YAW), atan2(track.state_vec(S_VY), track.state_vec(S_VX)));
        // if vel_heading_dot > 0, forward, if vel_heading_dot < 0, backward
        // if vel_heading_cross > 0, turn left, if vel_heading_cross < 0, turn right

        double heading_align_vel_ms = vel_heading_dot * track_vel;
        double target_wheel_base = track.dimension.length * 0.7; // 휠 베이스 는 차량 길이의 0.7 배 가정
        double max_yaw_rate_rad = heading_align_vel_ms * tan(config_.max_steer_deg * M_PI / 180.0) / target_wheel_base;

        // 트랙의 yaw rate 가 max_yaw_rate_rad 보다 크면 max_yaw_rate_rad 로 제한
        if (fabs(track.state_vec(S_YAW_RATE)) > fabs(max_yaw_rate_rad)) {
            track.state_vec(S_YAW_RATE) *= fabs(max_yaw_rate_rad) / fabs(track.state_vec(S_YAW_RATE));
        }

        // 트랙의 yaw rate 방향과 속도 방향이 다르면 yaw rate 를 0으로 설정
        if (track.state_vec(S_YAW_RATE) * vel_heading_cross < 0) {
            track.state_vec(S_YAW_RATE) = 0.0;
        }
    }
}

void EkfMultiObjectTracking::InitTrack(mc_mot::TrackStruct &track, const mc_mot::Meastruct &measurement) {
    track.track_id = cur_track_id_;
    track.update_time = measurement.state.time_stamp;
    track.detection_confidence = measurement.detection_confidence;

    track.state_vec(S_X) = measurement.state.x;
    track.state_vec(S_Y) = measurement.state.y;
    track.state_vec(S_YAW) = measurement.state.yaw;

    Eigen::Matrix3d S = H_ * track.state_cov * H_.transpose() + R_;
    Eigen::Matrix8_3d K = track.state_cov * H_.transpose() * S.inverse();

    // 공분산 행렬 업데이트
    Eigen::Matrix8_8d I = Eigen::Matrix8_8d::Identity();
    track.state_cov = (I - K * H_) * track.state_cov;

    track.dimension = measurement.dimension;
    track.class_score_arr[mc_mot::ObjectClass(measurement.classification)] = 1.0;
    track.object_z = measurement.state.z;
    track.age = 1;

    track.is_init = true;
    track.is_associated = true;

    track.updateDetectionCount(true);
}

void EkfMultiObjectTracking::MatchPairs(const Eigen::MatrixXd &cost_matrix, std::vector<int> &row_indices,
                                          std::vector<int> &col_indices) {
    // Initialize variables
    int num_rows = cost_matrix.rows(); // Meas num
    int num_cols = cost_matrix.cols(); // Track num
    row_indices.assign(num_rows, -1);
    col_indices.assign(num_cols, -1);
    std::vector<bool> row_assigned(num_rows, false);
    std::vector<bool> col_assigned(num_cols, false);

    // Create a list of all possible pairs (row, col) and sort them by cost
    struct Pair {
        int row;
        int col;
        double cost;
        bool operator<(const Pair &other) const { return cost < other.cost; }
    };

    std::vector<Pair> pairs;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            // Dist threshold 이내 있는 것만 pairs 에 추가
            if (cost_matrix(i, j) < config_.max_association_dist_m) {
                pairs.push_back({i, j, cost_matrix(i, j)});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end());

    // Greedily assign pairs with the smallest cost
    for (const auto &pair : pairs) {
        if (!row_assigned[pair.row] && !col_assigned[pair.col]) {
            row_indices[pair.row] = pair.col;
            col_indices[pair.col] = pair.row;
            row_assigned[pair.row] = true;
            col_assigned[pair.col] = true;
        }
    }
}

void EkfMultiObjectTracking::UpdateTrackId() {
    int i_cur_track_id = cur_track_id_;
    do {
        cur_track_id_++;
        if (cur_track_id_ >= MAX_TRACKS) {
            cur_track_id_ = 0;
        }

        if (cur_track_id_ == i_cur_track_id) {
            break; // 전체 트랙을 돌아도 자리가 없다면, 해당 트랙 불가피하게 교체
        }
    } while (all_tracks_[cur_track_id_].is_init == true); // 다음 track 번호가 점유중이면 track id 추가
}

void EkfMultiObjectTracking::UpdateMatrix() {
    Q_ = Eigen::Matrix8_8d::Zero();
    Q_.diagonal().array() = config_.system_noise_std_xy_m;
    Q_(S_X, S_X) = config_.system_noise_std_xy_m * config_.system_noise_std_xy_m;
    Q_(S_Y, S_Y) = config_.system_noise_std_xy_m * config_.system_noise_std_xy_m;
    Q_(S_YAW, S_YAW) = (config_.system_noise_std_yaw_deg * M_PI / 180.0) * 
                       (config_.system_noise_std_yaw_deg * M_PI / 180.0);
    Q_(S_VX, S_VX) = config_.system_noise_std_vx_vy_ms * config_.system_noise_std_vx_vy_ms;
    Q_(S_VY, S_VY) = config_.system_noise_std_vx_vy_ms * config_.system_noise_std_vx_vy_ms;
    Q_(S_YAW_RATE, S_YAW_RATE) = (config_.system_noise_std_yaw_rate_degs * M_PI / 180.0) *
                                 (config_.system_noise_std_yaw_rate_degs * M_PI / 180.0);
    Q_(S_AX, S_AX) = config_.system_noise_std_ax_ay_ms2 * config_.system_noise_std_ax_ay_ms2;
    Q_(S_AY, S_AY) = config_.system_noise_std_ax_ay_ms2 * config_.system_noise_std_ax_ay_ms2;

    R_ = Eigen::Matrix3d::Identity();
    R_(0, 0) = config_.meas_noise_std_xy_m * config_.meas_noise_std_xy_m; // x m
    R_(1, 1) = config_.meas_noise_std_xy_m * config_.meas_noise_std_xy_m; // y m
    R_(2, 2) = (config_.meas_noise_std_yaw_deg * M_PI / 180.0) *
               (config_.meas_noise_std_yaw_deg * M_PI / 180.0); // yaw rad

    H_ = Eigen::Matrix3_8d::Zero();
    H_(0, 0) = 1.0; // x
    H_(1, 1) = 1.0; // y
    H_(2, 2) = 1.0; // yaw
}

// ======== Utils Functions ============

double EkfMultiObjectTracking::CalculateDistance(const mc_mot::ObjectState &state1,
                                                   const mc_mot::ObjectState &state2) {
    return std::sqrt(std::pow(state1.x - state2.x, 2) + std::pow(state1.y - state2.y, 2));
}

double EkfMultiObjectTracking::CalculateDistance(const mc_mot::ObjectState &state1, const Eigen::Vector8d &state2) {
    return std::sqrt(std::pow(state1.x - state2(0), 2) + std::pow(state1.y - state2(1), 2));
}

double EkfMultiObjectTracking::CalculateMahalanobisDistance(const mc_mot::ObjectState &state1,
                                                              const mc_mot::TrackStruct &track) {
    Eigen::Vector2d mean_diff;
    mean_diff << state1.x - track.state_vec(S_X), state1.y - track.state_vec(S_Y);

    Eigen::Matrix2d covariance;
    covariance << track.state_cov(0, 0), track.state_cov(0, 1), track.state_cov(1, 0),
            track.state_cov(1, 1); // Covariance matrix는 state_cov의 상위 2x2 부분 사용

    // Covariance matrix의 역행렬을 계산
    Eigen::Matrix2d inv_covariance = covariance.inverse();

    // Mahalanobis distance 계산
    double mahalanobis_distance = std::sqrt(mean_diff.transpose() * inv_covariance * mean_diff);

    if (mahalanobis_distance > mean_diff.norm() * 3.0) mahalanobis_distance = mean_diff.norm() * 3.0;
    if (mahalanobis_distance < mean_diff.norm() / 5.0) mahalanobis_distance = mean_diff.norm() / 5.0;
    return mahalanobis_distance;
}

double EkfMultiObjectTracking::CalculateYawDotProduct(const double& yaw1, const double& yaw2) {
    // 방향 벡터 계산
    const double x1 = std::cos(yaw1);
    const double y1 = std::sin(yaw1);
    const double x2 = std::cos(yaw2);
    const double y2 = std::sin(yaw2);

    // 내적 계산
    const double dot_product = x1 * x2 + y1 * y2;
    return dot_product;
}

double EkfMultiObjectTracking::CalculateYawCrossProduct(const double& yaw1, const double& yaw2) {
    // 두 yaw 각도를 사용하여 단위 방향 벡터 생성
    const double x1 = std::cos(yaw1);
    const double y1 = std::sin(yaw1);
    const double x2 = std::cos(yaw2);
    const double y2 = std::sin(yaw2);

    // 2D 벡터의 외적 계산
    const double cross_product = x1 * y2 - y1 * x2;

    return cross_product;
}
