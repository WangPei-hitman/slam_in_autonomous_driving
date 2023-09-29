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
#include <algorithm>
#include <iomanip>
#include <utility>
#include <vector>

namespace sad {

/**
 * 书本第3章作业无迹卡尔曼滤波器
 * 可以指定观测GNSS的读数，GNSS应该事先转换到车体坐标系
 *
 * 使用15维的UKF（SPKF），标量类型可以由S指定，默认取double
 * 变量顺序：p, v, R, ba, bg
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class SPKF {
   public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec6T = Eigen::Matrix<S, 6, 1>;           // 6维向量类型
    using Vec12T = Eigen::Matrix<S, 12, 1>;         // 12维向量类型
    using Vec15T = Eigen::Matrix<S, 15, 1>;         // 15维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 12, 12>;  // 运动噪声类型
    using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // 里程计噪声类型
    using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS噪声类型
    using Mat15T = Eigen::Matrix<S, 15, 15>;        // 15维方差类型
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
        double k_ = 2;
        double k_se3_ = 2;
        double k_odom_ = 2;
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
                              const VecT& gravity = VecT(0, 0, -9.8066)) {
        BuildNoise(options);
        options_ = options;
        nav_state_.bg_ = init_bg;
        nav_state_.ba_ = init_ba;
        g_ = gravity;
        cov_ = Mat15T::Identity() * 1e-4;
        LOG(INFO) << nav_state_ << std::endl;
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
    NavStateT GetNominalState() const { return nav_state_; }

    /// 获取SE3 状态
    // SE3 GetNominalSE3() const { return; }

    /// 设置状态X
    void SetX(const NavStated& x, const Vec3d& grav) {}

    /// 设置协方差
    void SetCov(const Mat15T& cov) { cov_ = cov; }

    /// 设置SPKF超参数k
    void SetK(const double& k) { options_.k_ = k; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }

   private:
    void BuildNoise(const Options& options) {
        double ev = options.acce_var_;
        double et = options.gyro_var_;
        double ea = options.bias_acce_var_;
        double eg = options.bias_gyro_var_;

        double ev2 = ev;  // * ev;
        double et2 = et;  // * et;
        double ea2 = ea;  // * ea;
        double eg2 = eg;  // * eg;

        // 设置过程噪声
        Q_.diagonal() << ev2, ev2, ev2, et2, et2, et2, ea2, ea2, ea2, eg2, eg2, eg2;
        // 设置里程计噪声
        double o2 = options_.odom_var_ * options_.odom_var_;
        odom_noise_.diagonal() << o2, o2, o2;

        // 设置GNSS状态
        double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
        double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
        double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
        gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
    }

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

using SPKFD = SPKF<double>;
using SPKFF = SPKF<float>;

template <typename S>
bool SPKF<S>::Predict(const IMU& imu) {
    assert(imu.timestamp_ >= nav_state_.timestamp_);

    double dt = imu.timestamp_ - nav_state_.timestamp_;
    double sqr_dt = sqrtf64(dt);
    double dt2 = dt * dt;
    if (dt > (5 * options_.imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        LOG(INFO) << "skip this imu because dt_ = " << dt;
        nav_state_.timestamp_ = imu.timestamp_;
        return false;
    }

    // 状态变量+噪声 p v R ba bg nv nR na ng
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
    double coef = sqrtf64(options_.k_ + dim);
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

    /// 将每个sigmapoint代入非线性运动模型精确求解
    for (size_t i = 0; i < n_ps; i++) {
        const VecT&& p_k_1 = std::move(nav_ps.at(i).template segment<3>(0));
        const VecT&& v_k_1 = std::move(nav_ps.at(i).template segment<3>(3));
        const VecT&& phi_k_1 = std::move(nav_ps.at(i).template segment<3>(6));
        SO3 R_k_1 = SO3::exp(phi_k_1);
        const VecT&& ba_k_1 = std::move(nav_ps.at(i).template segment<3>(9));
        const VecT&& bg_k_1 = std::move(nav_ps.at(i).template segment<3>(12));

        const VecT&& nv = noise_ps.at(i).template segment<3>(0);
        const VecT&& nt = noise_ps.at(i).template segment<3>(3);
        const VecT&& na = noise_ps.at(i).template segment<3>(6);
        const VecT&& ng = noise_ps.at(i).template segment<3>(9);
        // LOG(INFO) << nav_ps[i].transpose();
        //  递推公式
        Vec3d imu_a = imu.acce_ - ba_k_1 - nv;
        Vec3d imu_g = imu.gyro_ - bg_k_1 - nt;
        SO3 R_k = R_k_1 * SO3::exp(imu_g);
        nav_ps.at(i).template segment<3>(0) = p_k_1 + v_k_1 * dt + 0.5 * g_ * dt2 + R_k_1 * imu_a * dt2 * 0.5;  // pk
        nav_ps.at(i).template segment<3>(3) = v_k_1 + g_ * dt + R_k_1 * imu_a * dt;                             // vk
        nav_ps.at(i).template segment<3>(6) = R_k.log();                                                        // Rk
        nav_ps.at(i).template segment<3>(9) = ba_k_1 + na;                                                      // ba_k
        nav_ps.at(i).template segment<3>(9) = bg_k_1 + ng;                                                      // bg_k
    }

    // 将每个sigmapoint重新组合成预测置信度
    double alpha0 = options_.k_ / (options_.k_ + dim);
    double alpha1 = 0.5 / (options_.k_ + dim);
    Vec15T nav_state_prior = Vec15T::Zero();  // 预测状态
    Mat15T nav_cov_prior = Mat15T::Zero();    // 预测协方差
    for (auto nav_p : nav_ps) {
        if (nav_p == *nav_ps.rbegin()) {
            nav_state_prior += alpha0 * nav_p;
            continue;
        }
        nav_state_prior += alpha1 * nav_p;
    }
    for (auto nav_p : nav_ps) {
        Vec15T deviation = nav_p - nav_state_prior;
        if (nav_p == *nav_ps.rbegin()) {
            nav_cov_prior += alpha0 * deviation * deviation.transpose();
            continue;
        }
        nav_cov_prior += alpha1 * deviation * deviation.transpose();
    }
    nav_state_ = NavStateT(imu.timestamp_, nav_state_prior);
    cov_ = std::move(nav_cov_prior);
    return true;
}

template <typename S>
bool SPKF<S>::ObserveWheelSpeed(const Odom& odom) {
    assert(odom.timestamp_ >= odom.timestamp_);
    // odom 修正以及雅可比

    static constexpr size_t dim = 15 + 3;        // 总状态维度L
    static constexpr size_t n_ps = 2 * dim + 1;  // sigma点个数

    // 卡尔曼增益和更新过程
    Eigen::Matrix<S, dim, dim> sigma_zz = Eigen::Matrix<S, dim, dim>::Zero();
    sigma_zz.template block<15, 15>(0, 0) = cov_;
    sigma_zz.template block<3, 3>(15, 15) = odom_noise_;
    Eigen::Matrix<S, dim, dim> L = sigma_zz.llt().matrixL();
    std::vector<Vec15T> nav_ps;  // 状态变量sigma点
    nav_ps.resize(n_ps);
    std::vector<VecT> noise_ps;  // 噪声sigma点
    noise_ps.resize(n_ps);
    std::vector<Vec15T> deviation_x;  // x的残差，后面计算会用
    deviation_x.resize(n_ps);
    double coef = sqrtf64(options_.k_odom_ + dim);
    Vec15T state_cur = nav_state_.Get15DVector();
    /// |     +     |       -      |  0 |
    /// |1 2 ... L  |-1 -2 ... -L  |  0 |
    /// |0 1 ... L-1| L L+1... 2L-1| 2L |
    for (size_t i = 0; i < dim; i++) {
        auto colL = L.col(i);
        Vec15T nav_part = colL.head(15);
        VecT noise_part = colL.tail(3);
        nav_ps.at(i) = state_cur + coef * nav_part;
        nav_ps.at(i + dim) = state_cur - coef * nav_part;
        deviation_x.at(i) = coef * nav_part;
        deviation_x.at(i + dim) = -deviation_x.at(i);
        noise_ps.at(i) = coef * noise_part;
        noise_ps.at(i + dim) = -noise_ps.at(i);  // 正的部分取负
    }
    nav_ps.at(n_ps - 1) = state_cur;
    noise_ps.at(n_ps - 1) = VecT::Zero();

    // 噪声观测方程求解
    for (size_t i = 0; i < n_ps; i++) {
        const VecT&& p_k_1 = std::move(nav_ps.at(i).template segment<3>(0));
        const VecT&& v_k_1 = std::move(nav_ps.at(i).template segment<3>(3));
        const VecT&& phi_k_1 = std::move(nav_ps.at(i).template segment<3>(6));
        SO3 R_k_1 = SO3::exp(phi_k_1);
        const VecT&& ba_k_1 = std::move(nav_ps.at(i).template segment<3>(9));
        const VecT&& bg_k_1 = std::move(nav_ps.at(i).template segment<3>(12));

        const VecT&& nv = std::move(noise_ps.at(i));

        // 递推公式
        nav_ps.at(i).template segment<3>(0) = p_k_1;       // pk
        nav_ps.at(i).template segment<3>(3) = v_k_1 + nv;  // vk
        nav_ps.at(i).template segment<3>(6) = phi_k_1;     // Rk
        nav_ps.at(i).template segment<3>(9) = ba_k_1;      // ba_k
        nav_ps.at(i).template segment<3>(9) = bg_k_1;      // bg_k
    }

    // 将每个sigmapoint重新组合成
    double alpha0 = options_.k_odom_ / (options_.k_odom_ + dim);
    double alpha1 = 0.5 / (options_.k_odom_ + dim);
    Vec15T mu_yk = Vec15T::Zero();      //
    Mat15T sigma_yyk = Mat15T::Zero();  //
    Mat15T sigma_xyk = Mat15T::Zero();  //
    for (auto nav_p : nav_ps) {
        if (nav_p == *nav_ps.rbegin()) {
            mu_yk += alpha0 * nav_p;
            continue;
        }
        mu_yk += alpha1 * nav_p;
    }
    for (size_t i = 0; i < n_ps; i++) {
        Vec15T deviation_y = nav_ps.at(i) - mu_yk;
        if (i == n_ps - 1) {
            sigma_yyk += alpha0 * deviation_y * deviation_y.transpose();
            sigma_xyk += alpha0 * deviation_x.at(i) * deviation_y.transpose();
            continue;
        }
        sigma_yyk += alpha1 * deviation_y * deviation_y.transpose();
        sigma_xyk += alpha1 * deviation_x.at(i) * deviation_y.transpose();
    }

    // velocity obs
    double velo_l = options_.wheel_radius_ * odom.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double velo_r =
        options_.wheel_radius_ * odom.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double average_vel = 0.5 * (velo_l + velo_r);

    VecT vel_odom(average_vel, 0.0, 0.0);
    VecT vel_world = nav_state_.R_ * vel_odom;
    Vec15T odom_measure;
    odom_measure << Vec3d(0, 0, 0), vel_world, Vec3d(0, 0, 0), Vec3d(0, 0, 0), Vec3d(0, 0, 0);

    Mat15T K = sigma_xyk * sigma_yyk.inverse();          // 卡尔曼增益
    cov_ = cov_ - K * sigma_xyk.transpose();             // 更新协方差
    state_cur = state_cur + K * (odom_measure - mu_yk);  // 更新状态
    // TODO api修改，不要把时间设为0
    nav_state_ = NavStateT(odom.timestamp_, state_cur);  // 注意更新时间
    return true;
}

template <typename S>
bool SPKF<S>::ObserveGps(const GNSS& gnss) {
    /// GNSS 观测的修正
    assert(gnss.unix_time_ >= nav_state_.timestamp_);

    if (first_gnss_) {
        nav_state_.R_ = gnss.utm_pose_.so3();
        nav_state_.p_ = gnss.utm_pose_.translation();
        first_gnss_ = false;
        nav_state_.timestamp_ = gnss.unix_time_;
        return true;
    }

    assert(gnss.heading_valid_);
    // ObserveSE3(gnss.utm_pose_, options_.gnss_pos_noise_, options_.gnss_ang_noise_);
    // nav_state_.timestamp_ = gnss.unix_time_;
    return true;
}

template <typename S>
bool SPKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
    /// 既有旋转，也有平移
    // 状态变量+噪声 p v R ba bg np nR
    static constexpr size_t dim = 15 + 6;        // 总状态维度L
    static constexpr size_t n_ps = 2 * dim + 1;  // sigma点个数

    // 卡尔曼增益和更新过程
    Vec6d noise_vec;
    noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise;
    Mat6d V = noise_vec.asDiagonal();
    Eigen::Matrix<S, dim, dim> sigma_zz = Eigen::Matrix<S, 21, 21>::Zero();
    sigma_zz.template block<15, 15>(0, 0) = cov_;
    sigma_zz.template block<6, 6>(15, 15) = V;
    Eigen::Matrix<S, dim, dim> L = sigma_zz.llt().matrixL();
    std::vector<Vec15T> nav_ps;  // 状态变量sigma点
    nav_ps.resize(n_ps);
    std::vector<Vec6T> noise_ps;  // 噪声sigma点
    noise_ps.resize(n_ps);
    std::vector<Vec15T> deviation_x;  // x的残差，后面计算会用
    deviation_x.resize(n_ps);
    double coef = sqrtf64(options_.k_se3_ + dim);
    Vec15T state_cur = nav_state_.Get15DVector();
    /// |     +     |       -      |  0 |
    /// |1 2 ... L  |-1 -2 ... -L  |  0 |
    /// |0 1 ... L-1| L L+1... 2L-1| 2L |
    for (size_t i = 0; i < dim; i++) {
        auto colL = L.col(i);
        Vec15T nav_part = colL.head(15);
        Vec6T noise_part = colL.tail(6);
        nav_ps.at(i) = state_cur + coef * nav_part;
        nav_ps.at(i + dim) = state_cur - coef * nav_part;
        deviation_x.at(i) = coef * nav_part;
        deviation_x.at(i + dim) = -deviation_x.at(i);
        noise_ps.at(i) = coef * noise_part;
        noise_ps.at(i + dim) = -noise_ps.at(i);  // 正的部分取负
    }
    nav_ps.at(n_ps - 1) = state_cur;
    noise_ps.at(n_ps - 1) = Vec6T::Zero();
    // 噪声方程求解
    for (size_t i = 0; i < n_ps; i++) {
        const VecT&& p_k_1 = std::move(nav_ps.at(i).template segment<3>(0));
        const VecT&& v_k_1 = std::move(nav_ps.at(i).template segment<3>(3));
        const VecT&& phi_k_1 = std::move(nav_ps.at(i).template segment<3>(6));
        SO3 R_k_1 = SO3::exp(phi_k_1);
        const VecT&& ba_k_1 = std::move(nav_ps.at(i).template segment<3>(9));
        const VecT&& bg_k_1 = std::move(nav_ps.at(i).template segment<3>(12));

        const VecT&& nt = noise_ps.at(i).template segment<3>(0);
        const VecT&& nR = noise_ps.at(i).template segment<3>(3);

        // 递推公式
        SO3 R_k = R_k_1 * SO3::exp(nR);
        nav_ps.at(i).template segment<3>(0) = p_k_1 + nt;  // pk
        nav_ps.at(i).template segment<3>(3) = v_k_1;       // vk
        nav_ps.at(i).template segment<3>(6) = R_k.log();   // Rk
        nav_ps.at(i).template segment<3>(9) = ba_k_1;      // ba_k
        nav_ps.at(i).template segment<3>(9) = bg_k_1;      // bg_k
    }
    // 将每个sigmapoint重新组合成
    double alpha0 = options_.k_se3_ / (options_.k_se3_ + dim);
    double alpha1 = 0.5 / (options_.k_se3_ + dim);
    LOG(INFO) << "alpha0:" << alpha0 << '\n' << "alpha1:" << alpha1 << '\n';
    Vec15T mu_yk = Vec15T::Zero();      //
    Mat15d sigma_yyk = Mat15d::Zero();  //
    Mat15T sigma_xyk = Mat15T::Zero();  //
    for (auto nav_p : nav_ps) {
        if (nav_p == *nav_ps.rbegin()) {
            mu_yk += alpha0 * nav_p;
            continue;
        }
        mu_yk += alpha1 * nav_p;
    }
    for (size_t i = 0; i < n_ps; i++) {
        Vec15T deviation_y = nav_ps.at(i) - mu_yk;
        if (i == n_ps - 1) {
            sigma_xyk += alpha0 * deviation_x.at(i) * deviation_y.transpose();
            sigma_yyk += alpha0 * deviation_y * deviation_y.transpose();
            continue;
        }
        sigma_xyk += alpha1 * deviation_x.at(i) * deviation_y.transpose();
        sigma_yyk += alpha1 * deviation_y * deviation_y.transpose();
    }
    LOG(INFO) << sigma_xyk;
    LOG(INFO) << sigma_yyk;

    Vec3d position = pose.translation();
    Vec3d angle = pose.so3().log();
    Vec15d gnss_measure;
    gnss_measure << position, Vec3d(0, 0, 0), angle, Vec3d(0, 0, 0), Vec3d(0, 0, 0);

    Mat15T K = sigma_xyk * sigma_yyk.inverse();          //
    cov_ = cov_ - K * sigma_xyk.transpose();             // 更新协方差
    state_cur = state_cur + K * (gnss_measure - mu_yk);  // 更新状态
    LOG(INFO) << state_cur.transpose();
    // TODO api修改，不要把时间设为0
    nav_state_ = NavStateT(0, state_cur);  // 注意更新时间
    return true;
}

}  // namespace sad

#endif /* !SLAM_in_AUTO_DRIVING_SPKF_HPP_*/
