#ifndef SLAM_in_AUTO_DRIVING_SPKF_HPP_
#define SLAM_in_AUTO_DRIVING_SPKF_HPP_

#include "common/eigen_types.h"
#include "common/gnss.h"
#include "common/imu.h"
#include "common/math_sophus.h"
#include "common/math_utils.h"
#include "common/nav_state.h"
#include "common/odom.h"

#include <Eigen/src/Core/Block.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/Ref.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Cholesky>
#include <algorithm>
#include <boost/algorithm/string/finder.hpp>
#include <boost/fusion/container/set/set_fwd.hpp>
#include <cmath>
#include <iomanip>
#include <utility>
#include <vector>

namespace sad {

/**
 * 书本第3章作业无迹卡尔曼滤波器
 * 可以指定观测GNSS的读数，GNSS应该事先转换到车体坐标系
 *
 * 使用18维的UKF（SPKF），标量类型可以由S指定，默认取double
 * 变量顺序：p, v, R, ba, bg, g
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class SPKF {
 public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec18T = Eigen::Matrix<S, 18, 1>;         // 18维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 18, 18>;  // 运动噪声类型
    using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // 里程计噪声类型
    using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS噪声类型
    using Mat18T = Eigen::Matrix<S, 18, 18>;        // 18维方差类型
    using NavStateT = NavState<S>;                  // 整体名义状态变量类型
    using VecXT = Eigen::Matrix<S, -1, 1>;
    using MatXT = Eigen::Matrix<S, -1, -1>;
    enum spkfType {
        mean_set,
        scaled_set,
        gauss_set,
    };
    struct Options {
        Options() = default;

        /// IMU 测量与零偏参数
        double imu_dt_ = 0.01;  // IMU测量间隔
        // NOTE IMU噪声项都为离散时间，不需要再乘dt，可以由初始化器指定IMU噪声
        double gyro_var_ = 1e-5;       // 陀螺测量标准差
        double acce_var_ = 1e-2;       // 加计测量标准差
        double bias_gyro_var_ = 1e-6;  // 陀螺零偏游走标准差
        double bias_acce_var_ = 1e-4;  // 加计零偏游走标准差

        /// 里程计参数
        double odom_var_ = 0.25;
        double odom_span_ = 0.1;        // 里程计测量间隔
        double wheel_radius_ = 0.155;   // 轮子半径
        double circle_pulse_ = 1024.0;  // 编码器每圈脉冲数

        /// RTK 观测参数
        double gnss_pos_noise_ = 0.1;                   // GNSS位置噪声
        double gnss_height_noise_ = 0.1;                // GNSS高度噪声
        double gnss_ang_noise_ = 1.0 * math::kDEG2RAD;  // GNSS旋转噪声

        /// 其他配置
        bool update_bias_gyro_ = true;  // 是否更新陀螺bias
        bool update_bias_acce_ = true;  // 是否更新加计bias
    };

    struct SpkfSettings {
        SpkfSettings() = default;
        int L;
        double alpha = 0.001;
        double k = 1;
        double beta = 2;
        double wa0 = 0.3;
        double wc0;
        double wa;
        double wc;
        enum spkfType spkf_type = spkfType::mean_set;
    };

    /**
     * 初始零偏取零
     */
    SPKF(Options option = Options()) : options_(option) { BuildNoise(option); }

    /**
     * 设置初始条件
     * @param options 噪声项配置
     * @param init_bg 初始零偏 陀螺
     * @param init_ba 初始零偏 加计
     * @param gravity 重力
     */
    void SetInitialConditions(Options options, const VecT& init_bg, const VecT& init_ba,
                              const VecT& gravity = VecT(0, 0, -9.8)) {
        BuildNoise(options);
        options_ = options;
        bg_ = init_bg;
        ba_ = init_ba;
        g_ = gravity;
        cov_ = Mat18T::Identity() * 1e-4;
    }

    /**
     * 构建sigma点
     * @param[in] x    先验状态变量 均值
     * @param[in] P    先验状态变量 协方差
     * @param[inout] settings spkf滤波器参数
     * @param[out] sigmapoints 输出的sigma点
     */
    void ConstructSigmaPoint(const VecXT& x, const MatXT& P, SpkfSettings& setting, std::vector<VecXT>* sigmapoints);
    /**
     * sigma点重新构建
     * @param[in] sigmapoints
     * @param[inout] settings spkf滤波器参数
     * @param[out] x    重建的状态变量 均值
     * @param[out] P    重建的状态变量 协方差
     */
    void SigmaPointReconstruct(const std::vector<VecXT>& sigmapoints, SpkfSettings& setting, Eigen::Ref<VecXT> x,
                               Eigen::Ref<MatXT> P);

    /**
     * 观测过程 计算sigmapoints状态和观测的协方差
     * @param[in] x 状态sigma点
     * @param[in] y 观测sigma点
     * @param[in] mean_x 状态均值
     * @param[in] mean_y 观测均值
     * @param[in] settings spkf滤波器参数
     * @param[out] cross 交叉协方差
     */
    void ConputeCrossCovariance(const std::vector<VecXT>& x, const std::vector<VecXT>& y,
                                const Eigen::Ref<VecXT> mean_x, const Eigen::Ref<VecXT> mean_y,
                                const SpkfSettings& setting, Eigen::Ref<MatXT> cross);

    /// 使用IMU递推
    bool Predict(const IMU& imu);

    /// 使用轮速计观测
    bool ObserveWheelSpeed(const Odom& odom);

    /// 使用GPS观测
    bool ObserveGps(const GNSS& gnss);

    /**
     * 使用SE3进行观测
     * @param pose  观测位姿
     * @param trans_noise 平移噪声
     * @param ang_noise   角度噪声
     * @return
     */
    bool ObserveSE3(const SE3& pose, double trans_noise = 0.1, double ang_noise = 1.0 * math::kDEG2RAD);

    /// accessors
    /// 获取全量状态
    NavStateT GetNominalState() const { return NavStateT(current_time_, R_, p_, v_, bg_, ba_); }

    /// 获取SE3 状态
    SE3 GetNominalSE3() const { return SE3(R_, p_); }

    /// 设置状态X
    void SetX(const NavStated& x, const Vec3d& grav) {
        current_time_ = x.timestamp_;
        R_ = x.R_;
        p_ = x.p_;
        v_ = x.v_;
        bg_ = x.bg_;
        ba_ = x.ba_;
        g_ = grav;
    }

    /// 设置协方差
    void SetCov(const Mat18T& cov) { cov_ = cov; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }

    void PrintNoise() const {
        LOG(INFO) << "cov_\n" << cov_;
        LOG(INFO) << "Q\n" << Q_;
        LOG(INFO) << "odom_noise_\n" << odom_noise_;
        LOG(INFO) << "gnss_noise_\n" << gnss_noise_;
    }

 private:
    void BuildNoise(const Options& options) {
        double ev = options.acce_var_;
        double et = options.gyro_var_;
        double eg = options.bias_gyro_var_;
        double ea = options.bias_acce_var_;

        double ev2 = ev;  // * ev;
        double et2 = et;  // * et;
        double eg2 = eg;  // * eg;
        double ea2 = ea;  // * ea;

        // 设置过程噪声
        Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2, ea2, 0, 0, 0;

        // 设置里程计噪声
        double o2 = options_.odom_var_ * options_.odom_var_;
        odom_noise_.diagonal() << o2, o2, o2;

        // 设置GNSS状态
        double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
        double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
        double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
        gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
    }

    /// 更新名义状态变量，重置error state
    void UpdateAndReset() {
        p_ += dx_.template block<3, 1>(0, 0);
        v_ += dx_.template block<3, 1>(3, 0);
        R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

        if (options_.update_bias_gyro_) {
            bg_ += dx_.template block<3, 1>(9, 0);
        }

        if (options_.update_bias_acce_) {
            ba_ += dx_.template block<3, 1>(12, 0);
        }

        g_ += dx_.template block<3, 1>(15, 0);

        ProjectCov();
        dx_.setZero();
    }

    /// 对P阵进行投影，参考式(3.63)
    void ProjectCov() {
        Mat18T J = Mat18T::Identity();
        J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0));
        cov_ = J * cov_ * J.transpose();
    }

    /// 成员变量
    double current_time_ = 0.0;  // 当前时间

    /// 名义状态
    VecT p_ = VecT::Zero();
    VecT v_ = VecT::Zero();
    SO3 R_;
    VecT bg_ = VecT::Zero();
    VecT ba_ = VecT::Zero();
    VecT g_{0, 0, -9.8};

    /// 误差状态
    Vec18T dx_ = Vec18T::Zero();

    /// 协方差阵
    Mat18T cov_ = Mat18T::Identity();

    /// 噪声阵
    MotionNoiseT Q_ = MotionNoiseT::Zero();
    OdomNoiseT odom_noise_ = OdomNoiseT::Zero();
    GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

    /// 标志位
    bool first_gnss_ = true;  // 是否为第一个gnss数据

    /// 配置项
    Options options_;
};

using SPKFD = SPKF<double>;
using SPKFF = SPKF<float>;

template <typename S>
bool SPKF<S>::Predict(const IMU& imu) {
    assert(imu.timestamp_ >= current_time_);
    double dt = imu.timestamp_ - current_time_;
    if (dt > (5 * options_.imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        LOG(INFO) << "skip this imu because dt_ = " << dt;
        current_time_ = imu.timestamp_;
        return false;
    }
    VecXT state(30);
    state.template head(18) = dx_;                              // p v t bg ba g
    state.template tail(12) = Eigen::Matrix<S, 12, 1>::Zero();  // nv nt ng na
    MatXT cov = MatXT::Zero(30, 30);
    cov.template block<18, 18>(0, 0) = cov_;
    cov.template block<12, 12>(18, 18) = Q_.template block<12, 12>(3, 3);
    SpkfSettings pridict_setting;
    std::vector<VecXT> sigmapoints;
    ConstructSigmaPoint(state, cov, pridict_setting, &sigmapoints);

    // nominal state 递推
    VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt + 0.5 * g_ * dt * dt;
    VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt;
    SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt);

    R_ = new_R;
    v_ = new_v;
    p_ = new_p;  // 其余状态维度不变

    // error state 递推
    // 计算运动过程雅可比矩阵 F，见(3.47)
    // F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式，这里为了教学方便，使用矩阵形式
    Mat18T F = Mat18T::Identity();                                                 // 主对角线
    F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v对theta
    F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg
    // mean and cov prediction
    std::vector<VecXT> nav_ps;
    for (auto& sp : sigmapoints) {
        VecXT dx = sp.template head(18);
        dx = F.eval() * dx;
        dx.template segment<6>(3) -= sp.template segment<6>(18);  // v t
        dx.template segment<6>(9) += sp.template segment<6>(24);  // g a
        nav_ps.emplace_back(std::move(dx));
    }
    SigmaPointReconstruct(nav_ps, pridict_setting, dx_, cov_);
    EXPECT_TRUE(isfinite(dx_[0])) << "dx_:" << dx_.transpose();
    current_time_ = imu.timestamp_;
    return true;
}

template <typename S>
bool SPKF<S>::ObserveWheelSpeed(const Odom& odom) {
    assert(odom.timestamp_ >= current_time_);
    // odom 修正以及雅可比
    // 使用三维的轮速观测，H为3x18，大部分为零
    // Eigen::Matrix<S, 3, 18> H = Eigen::Matrix<S, 3, 18>::Zero();
    // H.template block<3, 3>(0, 3) = Mat3T::Identity();

    VecXT state(21);
    state.template head(18) = dx_;                            // p v t bg ba g
    state.template tail(3) = Eigen::Matrix<S, 3, 1>::Zero();  // nv
    MatXT cov = MatXT::Zero(21, 21);
    cov.template block<18, 18>(0, 0) = cov_;
    cov.template block<3, 3>(18, 18) = odom_noise_;
    SpkfSettings wheel_setting;
    std::vector<VecXT> sigmapoints;
    ConstructSigmaPoint(state, cov, wheel_setting, &sigmapoints);

    // velocity obs
    double velo_l = options_.wheel_radius_ * odom.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double velo_r =
        options_.wheel_radius_ * odom.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double average_vel = 0.5 * (velo_l + velo_r);
    VecT vel_odom(average_vel, 0.0, 0.0);
    VecT vel_world = R_ * vel_odom;

    std::vector<VecXT> x_ps;    // 状态sigma点
    std::vector<VecXT> obs_ps;  // 观测sigma点
    for (auto& sp : sigmapoints) {
        VecXT dx = sp.template head(18);
        VecXT noise = sp.template tail(3);
        dx.template segment<3>(3) += noise;
        x_ps.push_back(dx);
        obs_ps.emplace_back(noise);
    }
    VecT mean = VecT::Zero();
    Mat3T self = Mat3T::Zero();
    SigmaPointReconstruct(obs_ps, wheel_setting, mean, self);
    MatXT cross(18, 3);
    cross.setZero();
    ConputeCrossCovariance(x_ps, obs_ps, x_ps[0], mean, wheel_setting, cross);

    // 卡尔曼增益
    // Eigen::Matrix<S, 18, 3> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + odom_noise_).inverse();
    Eigen::Matrix<S, 18, 3> K = cross * self.inverse();
    LOG(INFO) << "F:\n" << K;
    dx_ += K * (vel_world - v_ - mean);
    cov_ = cov_ - K * cross.transpose();
    // dx_ = K * (vel_world - v_);

    // update cov
    // cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

template <typename S>
bool SPKF<S>::ObserveGps(const GNSS& gnss) {
    /// GNSS 观测的修正
    assert(gnss.unix_time_ >= current_time_);

    if (first_gnss_) {
        R_ = gnss.utm_pose_.so3();
        p_ = gnss.utm_pose_.translation();
        first_gnss_ = false;
        current_time_ = gnss.unix_time_;
        return true;
    }

    assert(gnss.heading_valid_);
    ObserveSE3(gnss.utm_pose_, options_.gnss_pos_noise_, options_.gnss_ang_noise_);
    current_time_ = gnss.unix_time_;

    return true;
}

template <typename S>
bool SPKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
    /// 既有旋转，也有平移
    /// 观测状态变量中的p, R，H为6x18，其余为零
    Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero();
    H.template block<3, 3>(0, 0) = Mat3T::Identity();  // P部分
    H.template block<3, 3>(3, 6) = Mat3T::Identity();  // R部分（3.66)

    // 卡尔曼增益和更新过程
    // Vec6d noise_vec;
    // noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise;
    // Mat6d V = noise_vec.asDiagonal();
    VecXT state(24);
    state.template head(18) = dx_;                            // p v t bg ba g
    state.template tail(6) = Eigen::Matrix<S, 6, 1>::Zero();  // np nt
    MatXT cov = MatXT::Zero(24, 24);
    cov.template block<18, 18>(0, 0) = cov_;
    cov.template block<6, 6>(18, 18) = gnss_noise_;
    SpkfSettings se3_setting;
    std::vector<VecXT> sigmapoints;
    ConstructSigmaPoint(state, cov, se3_setting, &sigmapoints);

    std::vector<VecXT> x_ps;    // 状态sigma点
    std::vector<VecXT> obs_ps;  // 观测sigma点
    for (auto& sp : sigmapoints) {
        VecXT dx = sp.template head(18);
        VecXT noise = sp.template tail(6);
        dx.template segment<3>(0) += noise.template head(3);  // np
        VecT dtheta = dx.template segment(6, 3);
        Mat3T J = math::JacRightInv(dtheta);
        dx.template segment<3>(6) += J * noise.template tail(3);  // nt
        x_ps.push_back(dx);
        obs_ps.emplace_back(noise);
    }
    Eigen::Matrix<S, 6, 1> mean = Eigen::Matrix<S, 6, 1>::Zero();
    Eigen::Matrix<S, 6, 6> self = Eigen::Matrix<S, 6, 6>::Zero();
    SigmaPointReconstruct(obs_ps, se3_setting, mean, self);
    MatXT cross(18, 6);
    cross.setZero();
    ConputeCrossCovariance(x_ps, obs_ps, x_ps[0], mean, se3_setting, cross);
    // LOG(INFO) << "mean:" << mean.transpose();
    // LOG(INFO) << "self:\n" << self;
    // LOG(INFO) << "cross:\n" << cross;

    // Eigen::Matrix<S, 18, 6> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse();

    // 更新x和cov
    Vec6d innov = Vec6d::Zero();
    innov.template head<3>() = (pose.translation() - p_ - mean.template head(3));          // 平移部分
    innov.template tail<3>() = (R_.inverse() * pose.so3()).log() - mean.template tail(3);  // 旋转部分(3.67)
    Eigen::Matrix<S, 18, 6> K = cross * self.inverse();
    // LOG(INFO) << "F:\n" << K;
    dx_ += K * innov;
    cov_ = cov_ - K * cross.transpose();
    // Vec6d innov = Vec6d::Zero();
    // innov.template head<3>() = (pose.translation() - p_);          // 平移部分
    // innov.template tail<3>() = (R_.inverse() * pose.so3()).log();  // 旋转部分(3.67)

    // dx_ = K * innov;
    // cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

template <typename S>
void SPKF<S>::ConstructSigmaPoint(const VecXT& x, const MatXT& P, SpkfSettings& settings,
                                  std::vector<VecXT>* sigmapoints) {
    settings.L = x.rows();
    const int& L = settings.L;
    int n_pts = 2 * L + 1;
    MatXT A = P.llt().matrixL();
    const double& alpha = settings.alpha;
    const double& beta = settings.beta;
    const double& k = settings.k;
    const double& wa0 = settings.wa0;
    double coef = std::sqrt(L / (1 - wa0));
    sigmapoints->resize(n_pts);
    sigmapoints->at(0) = x;
    for (size_t i = 1; i <= L; ++i) {
        sigmapoints->at(i) = x + coef * A.col(i - 1);
        sigmapoints->at(i + L) = x - coef * A.col(i - 1);
    }
    settings.wa = 0.5 * (1 - wa0) / L;
    settings.wc0 = settings.wa0;
    settings.wc = settings.wa;
    return;
}

template <typename S>
void SPKF<S>::SigmaPointReconstruct(const std::vector<VecXT>& sigmapoints, SpkfSettings& setting, Eigen::Ref<VecXT> x,
                                    Eigen::Ref<MatXT> P) {
    int pts = 2 * setting.L + 1;
    x.setZero();
    P.setZero();
    for (size_t i = 0; i < pts; ++i) {
        if (0 == i) {
            x += (sigmapoints[i] * setting.wa0);
            continue;
        }
        x += (sigmapoints[i] * setting.wa);
    }
    for (size_t i = 0; i < pts; ++i) {
        VecXT devi = sigmapoints[i] - x;
        if (0 == i) {
            P += (devi * devi.transpose().eval()) * setting.wc0;
            continue;
        }
        P += (devi * devi.transpose().eval()) * setting.wc;
    }
}

template <typename S>
void SPKF<S>::ConputeCrossCovariance(const std::vector<VecXT>& x, const std::vector<VecXT>& y,
                                     const Eigen::Ref<VecXT> mean_x, const Eigen::Ref<VecXT> mean_y,
                                     const SpkfSettings& setting, Eigen::Ref<MatXT> cross) {
    assert(x.size() == y.size());
    int pts = 2 * setting.L + 1;
    cross.setZero();
    for (size_t i = 0; i < pts; ++i) {
        VecXT devi_x = x[i] - mean_x;
        VecXT devi_y = y[i] - mean_y;
        if (0 == i) {
            cross += (devi_x * devi_y.transpose().eval()) * setting.wc0;
            continue;
        }
        cross += (devi_x * devi_y.transpose().eval()) * setting.wc;
    }
}

}  // namespace sad

#endif /* !SLAM_in_AUTO_DRIVING_SPKF_HPP_*/
