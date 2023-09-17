#ifndef SLAM_in_AUTO_DRIVING_SPKF_HPP_
#define SLAM_in_AUTO_DRIVING_SPKF_HPP_

#include "common/eigen_types.h"
#include "common/gnss.h"
#include "common/imu.h"
#include "common/math_utils.h"
#include "common/nav_state.h"
#include "common/odom.h"

#include <glog/logging.h>
#include <Eigen/Cholesky>
#include <iomanip>
#include <vector>

namespace sad {

/**
 * 书本第3章作业无迹卡尔曼滤波器
 * 可以指定观测GNSS的读数，GNSS应该事先转换到车体坐标系
 *
 * 15使用18维的UKF（SPKF），标量类型可以由S指定，默认取double
 * 变量顺序：p, v, R, bg, ba
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class ESKF {
   public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec12T = Eigen::Matrix<S, 12, 1>;         // 12维向量类型
    using Vec15T = Eigen::Matrix<S, 15, 1>;         // 15维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 15, 15>;  // 运动噪声类型
    using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // 里程计噪声类型
    using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS噪声类型
    using Mat15T = Eigen::Matrix<S, 15, 15>;        // 18维方差类型
    using NavStateT = NavState<S>;                  // 整体名义状态变量类型

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
        double odom_var_ = 0.5;
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
        /// SPKF超参数
        double k_ = 1.0;
    };

    /**
     * 初始零偏取零
     */
    ESKF(Options option = Options()) : options_(option) { BuildNoise(option); }

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
    NavStateT GetNominalState() const {}

    /// 获取SE3 状态
    SE3 GetNominalSE3() const { return; }

    /// 设置状态X
    void SetX(const NavStated& x, const Vec3d& grav) {}

    /// 设置协方差
    void SetCov(const Mat18T& cov) { cov_ = cov; }

    /// 设置SPKF超参数k
    void SetK(const double& k) { options_.k_ = k; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }

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
        Q_.diagonal() << ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2,
            ea2；

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
    void UpdateAndReset() {}

    /// 将每个sigmapoint代入非线性运动模型精确求解
    void MotionModel(const std::vector nav_ps&, const std::vector noise_ps&, const IMU& imu) {}

    /// 成员变量
    /// 状态变量
    NavStateT nav_state_{0.0};

    VecT g_{0, 0, -9.8};

    /// 协方差阵
    Mat15T cov_ = Mat15T::Identity();

    /// 噪声阵
    MotionNoiseT Q_ = MotionNoiseT::Zero();
    OdomNoiseT odom_noise_ = OdomNoiseT::Zero();
    GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

    /// 标志位
    bool first_gnss_ = true;  // 是否为第一个gnss数据

    /// 配置项
    Options options_;
};

using ESKFD = ESKF<double>;
using ESKFF = ESKF<float>;

template <typename S>
bool ESKF<S>::Predict(const IMU& imu) {
    assert(imu.timestamp_ >= nav_state_.timestamp_);

    double dt = imu.timestamp_ - nav_state_.timestamp_;
    if (dt > (5 * options_.imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        LOG(INFO) << "skip this imu because dt_ = " << dt;
        nav_state_.timestamp_ = imu.timestamp_;
        return false;
    }

    // 状态变量+噪声 p v R ba bg nv nR ng na
    static constexpr size_t dim = 15 + 12;       // 总状态维度L
    static constexpr size_t n_ps = 2 * dim + 1;  // sigma点个数
    Eigen::Matrix<S, dim, dim> sigma_zz = Eigen::Matrix<S, 27, 27>::Zero();
    sigma_zz.template block<15, 15>(0, 0) = cov_;
    sigma_zz.template block<12, 12>(15, 15) = Q_;
    Eigen::Matrix<S, dim, dim> L = sigma_zz.llt().matrixL();
    std::vector<Vec15T> nav_ps;  // 状态变量sigma点
    nav_ps.resize(n_ps);
    std::vector<Vec12T> noise_ps;  // 噪声sigma点
    noise_ps.resize(n_ps);
    double coef = sqrtf64(options_.k_ + n_ps);
    Vec15T state_cur = nav_state_.Get15DVector();
    /// |     +     |       -      |  0 |
    /// |1 2 ... L  |-1 -2 ... -L  |  0 |
    /// |0 1 ... L-1| L L+1... 2L-1| 2L |
    for (size_t i = 0; i < dim; i++) {
        auto colL = L.col(i);
        Vec15T nav_part = colL.head(15);
        Vec12T noise_part = colL.tail(12);
        nav_ps.at(i) = state_cur + coef * nav_part;
        nav_ps.at(i + dim) = state_cur - coef * nav_part;
        noise_ps.at(i) = coef * noise_part;
        noise_ps.at(i + dim) = -noise_ps.at(i);  // 正的部分取负
    }
    nav_ps.at(n_ps - 1) = state_cur;
    noise_ps.at(n_ps - 1) = Vec12T::Zero();

    return true;
}

template <typename S>
bool ESKF<S>::ObserveWheelSpeed(const Odom& odom) {
    assert(odom.timestamp_ >= current_time_);
    // odom 修正以及雅可比
    // 使用三维的轮速观测，H为3x18，大部分为零
    Eigen::Matrix<S, 3, 18> H = Eigen::Matrix<S, 3, 18>::Zero();
    H.template block<3, 3>(0, 3) = Mat3T::Identity();

    // 卡尔曼增益
    Eigen::Matrix<S, 18, 3> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + odom_noise_).inverse();

    // velocity obs
    double velo_l = options_.wheel_radius_ * odom.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double velo_r =
        options_.wheel_radius_ * odom.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double average_vel = 0.5 * (velo_l + velo_r);

    VecT vel_odom(average_vel, 0.0, 0.0);
    VecT vel_world = R_ * vel_odom;

    dx_ = K * (vel_world - v_);

    // update cov
    cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

template <typename S>
bool ESKF<S>::ObserveGps(const GNSS& gnss) {
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
bool ESKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
    /// 既有旋转，也有平移
    /// 观测状态变量中的p, R，H为6x18，其余为零
    Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero();
    H.template block<3, 3>(0, 0) = Mat3T::Identity();  // P部分
    H.template block<3, 3>(3, 6) = Mat3T::Identity();  // R部分（3.66)

    // 卡尔曼增益和更新过程
    Vec6d noise_vec;
    noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise;

    Mat6d V = noise_vec.asDiagonal();
    Eigen::Matrix<S, 18, 6> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse();

    // 更新x和cov
    Vec6d innov = Vec6d::Zero();
    innov.template head<3>() = (pose.translation() - p_);          // 平移部分
    innov.template tail<3>() = (R_.inverse() * pose.so3()).log();  // 旋转部分(3.67)

    dx_ = K * innov;
    cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

}  // namespace sad

#endif /* !SLAM_in_AUTO_DRIVING_SPKF_HPP_*/
