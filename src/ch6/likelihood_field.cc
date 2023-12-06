//
// Created by xiang on 2022/3/18.
//

#include "ch6/g2o_types.h"
#include "ch6/likelihood_filed.h"
#include "common/eigen_types.h"
#include "common/math_utils.h"
#include "sophus/ceres_manifold.hpp"

#include <ceres/cost_function.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/sized_cost_function.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

namespace sad {

void LikelihoodField::SetTargetScan(Scan2d::Ptr scan) {
    target_ = scan;

    // 在target点上生成场函数
    field_ = cv::Mat(1000, 1000, CV_32F, 30.0);

    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max) {
            continue;
        }

        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x = scan->ranges[i] * std::cos(real_angle) * resolution_ + 500;
        double y = scan->ranges[i] * std::sin(real_angle) * resolution_ + 500;

        // 在(x,y)附近填入场函数
        for (auto& model_pt : model_) {
            int xx = int(x + model_pt.dx_);
            int yy = int(y + model_pt.dy_);
            if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows &&
                field_.at<float>(yy, xx) > model_pt.residual_) {
                field_.at<float>(yy, xx) = model_pt.residual_;
            }
        }
    }
}

void LikelihoodField::BuildModel() {
    const int range = 20;  // 生成多少个像素的模板
    for (int x = -range; x <= range; ++x) {
        for (int y = -range; y <= range; ++y) {
            model_.emplace_back(x, y, std::sqrt((x * x) + (y * y)));
        }
    }
}

void LikelihoodField::SetSourceScan(Scan2d::Ptr scan) { source_ = scan; }

bool LikelihoodField::AlignGaussNewton(SE2& init_pose) {
    int iterations = 10;
    double cost = 0, lastCost = 0;
    SE2 current_pose = init_pose;
    const int min_effect_pts = 20;  // 最小有效点数
    const int image_boarder = 20;   // 预留图像边界

    has_outside_pts_ = false;
    for (int iter = 0; iter < iterations; ++iter) {
        Mat3d H = Mat3d::Zero();
        Vec3d b = Vec3d::Zero();
        cost = 0;
        int effective_num = 0;  // 有效点数
        // 遍历source
        for (size_t i = 0; i < source_->ranges.size(); ++i) {
            float r = source_->ranges[i];
            if (r < source_->range_min || r > source_->range_max) {
                continue;
            }

            float angle = source_->angle_min + i * source_->angle_increment;
            if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
                continue;
            }

            float theta = current_pose.so2().log();
            Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));

            // 在field中的图像坐标
            Vec2i pf = (pw * resolution_ + Vec2d(500, 500)).cast<int>();
            Vec2d pfd = pw * resolution_ + Vec2d(500, 500);

            if (pf[0] >= image_boarder && pf[0] < field_.cols - image_boarder && pf[1] >= image_boarder &&
                pf[1] < field_.rows - image_boarder) {
                effective_num++;

                // 图像梯度 场强=势差
                float dx = 0, dy = 0;
                if (linearize_field_) {
                    int x1 = static_cast<int>(pfd[0]);
                    int x2 = x1 + 1;
                    int y1 = static_cast<int>(pfd[1]);
                    int y2 = y1 + 1;
                    float delta_x = pfd[0] - x1;
                    float delta_y = pfd[1] - y1;
                    dx = delta_y * (field_.at<float>(y1, x2) - field_.at<float>(y1, x1)) +
                         (1 - delta_y) * (field_.at<float>(y2, x2) - field_.at<float>(y2, x1));
                    dy = delta_x * (field_.at<float>(y2, x1) - field_.at<float>(y1, x1)) +
                         (1 - delta_x) * (field_.at<float>(y2, x2) - field_.at<float>(y1, x2));
                } else {
                    dx = 0.5 * (field_.at<float>(pf[1], pf[0] + 1) -
                                field_.at<float>(pf[1], pf[0] - 1));  // 0.5 for cross 2 pixel,alpha=1
                    dy = 0.5 * (field_.at<float>(pf[1] + 1, pf[0]) - field_.at<float>(pf[1] - 1, pf[0]));
                }

                Vec3d J;
                J << resolution_ * dx, resolution_ * dy,
                    -resolution_ * dx * r * std::sin(angle + theta) + resolution_ * dy * r * std::cos(angle + theta);
                H += J * J.transpose();

                float e = field_.at<float>(pf[1], pf[0]);
                b += -J * e;

                cost += e * e;
            } else {
                has_outside_pts_ = true;
            }
        }

        if (effective_num < min_effect_pts) {
            return false;
        }

        // solve for dx
        Vec3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])) {
            break;
        }

        cost /= effective_num;
        if (iter > 0 && cost >= lastCost) {
            break;
        }

        LOG(INFO) << "iter " << iter << " cost = " << cost << ", effect num: " << effective_num;

        current_pose.translation() += dx.head<2>();
        current_pose.so2() = current_pose.so2() * SO2::exp(dx[2]);
        lastCost = cost;
    }

    init_pose = current_pose;
    return true;
}

cv::Mat LikelihoodField::GetFieldImage() {
    cv::Mat image(field_.rows, field_.cols, CV_8UC3);
    for (int x = 0; x < field_.cols; ++x) {
        for (int y = 0; y < field_.rows; ++y) {
            float r = field_.at<float>(y, x) * 255.0 / 30.0;
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(r), uchar(r), uchar(r));
        }
    }
    return image;
}

bool LikelihoodField::AlignG2O(SE2& init_pose) {
    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    auto* v = new VertexSE2();
    v->setId(0);
    v->setEstimate(init_pose);
    optimizer.addVertex(v);

    const double range_th = 15.0;  // 不考虑太远的scan，不准
    const double rk_delta = 0.8;

    has_outside_pts_ = false;
    // 遍历source
    for (size_t i = 0; i < source_->ranges.size(); ++i) {
        float r = source_->ranges[i];
        if (r < source_->range_min || r > source_->range_max) {
            continue;
        }

        if (r > range_th) {
            continue;
        }

        float angle = source_->angle_min + i * source_->angle_increment;
        if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }

        auto e = new EdgeSE2LikelihoodFiled(field_, r, angle, resolution_);
        e->setVertex(0, v);

        if (e->IsOutSide()) {
            has_outside_pts_ = true;
            delete e;
            continue;
        }

        e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(rk_delta);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    init_pose = v->estimate();
    return true;
}

class LikelihoodCostFunctor : public ceres::SizedCostFunction<1, 3> {
 public:
    LikelihoodCostFunctor(const cv::Mat& field, double range, double angle, float resolution = 10.0)
        : field_(field), range_(range), angle_(angle), resolution_(resolution) {}
    virtual ~LikelihoodCostFunctor() {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;

    const cv::Mat& field_;
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 20.0;
    inline static const int image_boarder_ = 20;
};

bool LikelihoodCostFunctor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    const double x = parameters[0][0];
    const double y = parameters[0][1];
    const double theta = parameters[0][2];
    SE2 current_pose;
    current_pose.translation() = Vec2d(x, y);
    current_pose.so2() = SO2::exp(theta);
    // LOG(INFO) << "x y theta:" << x << ' ' << y << ' ' << theta;
    const Vec2d pw = current_pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
    // 在field中的图像坐标
    const Vec2d pf = pw * resolution_ + Vec2d(500, 500);
    residuals[0] = math::GetPixelValue<float>(field_, pf[0], pf[1]);
    // LOG(INFO) << "residual:" << residuals[0];
    // LOG(INFO) << "Pf:" << pf.transpose();
    // Compute the Jacobian if asked for.
    if (jacobians != nullptr && jacobians[0] != nullptr) {
        // 图像梯度 场强=势差
        const int x1 = static_cast<int>(pf[0]);
        const int x2 = x1 + 1;
        const int y1 = static_cast<int>(pf[1]);
        const int y2 = y1 + 1;
        const float delta_x = pf[0] - x1;
        const float delta_y = pf[1] - y1;
        const float dx = delta_y * (field_.at<float>(y1, x2) - field_.at<float>(y1, x1)) +
                         (1 - delta_y) * (field_.at<float>(y2, x2) - field_.at<float>(y2, x1));
        const float dy = delta_x * (field_.at<float>(y2, x1) - field_.at<float>(y1, x1)) +
                         (1 - delta_x) * (field_.at<float>(y2, x2) - field_.at<float>(y1, x2));
        jacobians[0][0] = resolution_ * dx;
        jacobians[0][1] = resolution_ * dy;
        jacobians[0][2] = -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                          resolution_ * dy * range_ * std::cos(angle_ + theta);
        // LOG(INFO) << "----------------------\n"
        //<< "Jacobi:" << jacobians[0][0] << ' ' << jacobians[0][1] << ' ' << jacobians[0][2];
    }
    return true;
}

bool LikelihoodField::AlignCeres(SE2& init_pose) {
    int iterations = 10;
    const int min_effect_pts = 20;  // 最小有效点数
    const int image_boarder = 20;   // 预留图像边界
    const double range_th = 15.0;   // 不考虑太远的scan，不准
    const double rk_delta = 0.8;

    has_outside_pts_ = false;
    LOG(INFO) << "pose1: " << init_pose.translation().transpose() << ", " << init_pose.so2().log();

    SE2 current_pose = init_pose;
    double pose2D[] = {current_pose.translation()[0], current_pose.translation()[1], current_pose.so2().log()};
    ceres::Problem problem;
    ceres::Manifold* SE2Manifold = new Sophus::Manifold<Sophus::SE2>;
    problem.AddParameterBlock(pose2D, 3, SE2Manifold);
    // 遍历source
    for (size_t i = 0; i < source_->ranges.size(); ++i) {
        float r = source_->ranges[i];
        if (r < source_->range_min || r > source_->range_max) {
            continue;
        }

        if (r > range_th) {
            continue;
        }

        float angle = source_->angle_min + i * source_->angle_increment;
        if (angle < source_->angle_min + 30 * M_PI / 180.0 || angle > source_->angle_max - 30 * M_PI / 180.0) {
            continue;
        }
        const Vec2d pw = current_pose * Vec2d(r * std::cos(angle), r * std::sin(angle));
        // 在field中的图像坐标
        const Vec2d pf = pw * resolution_ + Vec2d(500, 500);
        if (pf[0] >= image_boarder && pf[0] < field_.cols - image_boarder && pf[1] >= image_boarder &&
            pf[1] < field_.rows - image_boarder) {
            ceres::CostFunction* cost_function = new LikelihoodCostFunctor(field_, r, angle, resolution_);
            problem.AddResidualBlock(cost_function, nullptr, pose2D);
        }
    }

    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.max_num_iterations = iterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    init_pose.translation() = Vec2d{pose2D[0], pose2D[1]};
    init_pose.so2() = SO2::exp(pose2D[2]);
    //    LOG(INFO) << summary.BriefReport() << std::endl;
    LOG(INFO) << "pose2: " << init_pose.translation().transpose() << ", " << init_pose.so2().log();
}

void LikelihoodField::SetFieldImageFromOccuMap(const cv::Mat& occu_map) {
    const int boarder = 25;
    field_ = cv::Mat(1000, 1000, CV_32F, 30.0);

    for (int x = boarder; x < occu_map.cols - boarder; ++x) {
        for (int y = boarder; y < occu_map.rows - boarder; ++y) {
            if (occu_map.at<uchar>(y, x) < 127) {
                // 在该点生成一个model
                for (auto& model_pt : model_) {
                    int xx = int(x + model_pt.dx_);
                    int yy = int(y + model_pt.dy_);
                    if (xx >= 0 && xx < field_.cols && yy >= 0 && yy < field_.rows &&
                        field_.at<float>(yy, xx) > model_pt.residual_) {
                        field_.at<float>(yy, xx) = model_pt.residual_;
                    }
                }
            }
        }
    }
}

}  // namespace sad