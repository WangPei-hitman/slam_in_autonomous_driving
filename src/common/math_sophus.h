//
// Created by Pei on 10.4 2023
//

#ifndef SAD_MATH_SOPHUS_H
#define SAD_MATH_SOPHUS_H

#include <Eigen/Core>
#include "common/eigen_types.h"
#include "common/math_utils.h"

namespace sad::math {

template <typename T>
Eigen::Matrix<T, 3, 3> JacRightInv(const Eigen::Matrix<T, 3, 1>& v) {
    double len = std::sqrt(v[0] * v[0] + v[1] * v[1] * v[2] * v[2]);
    KeepAngleIn2PI(len);
    Eigen::Matrix<double, 3, 3> res = Eigen::Matrix<double, 3, 3>::Identity();
    double half_len = 0.5 * len;
    if (half_len < 1e-5) {
        res = Eigen::Matrix<T, 3, 3>::Identity();
    } else {
        double coef = half_len / std::tan(half_len);
        Eigen::Matrix<T, 3, 1> norm_v = v / len;
        res = Eigen::Matrix<T, 3, 3>::Identity() + half_len * SO3::hat(norm_v) +
              (1 - coef) * SO3::hat(norm_v) * SO3::hat(norm_v);
    }
    return res;
}

}  // namespace sad::math
#endif /* !SAD_MATH_SOPHUS_H*/
