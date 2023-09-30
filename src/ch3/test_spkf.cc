//
// Created by Pei on 2023/9/30.
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ch3/spkf.hpp"
#include "common/eigen_types.h"
#include "common/io_utils.h"

#include <cstddef>
#include <iostream>
#include <vector>

TEST(SPKF_TEST, GENERATE_SIGMAPOINT) {
    sad::SPKFD spkf;
    sad::SPKFD::VecXT x1(3);
    x1 = sad::SPKFD::VecXT::Random(3);
    sad::SPKFD::MatXT P1(3, 3);
    P1 = sad::SPKFD::MatXT::Identity(3, 3);
    sad::SPKFD::SpkfSettings setting;
    std::vector<sad::SPKFD::VecXT> sigmapoints;
    //在不参与模型推倒的情况下，产生的sigmapoints重新组合应该与原来相差不大
    spkf.ConstructSigmaPoint(x1, P1, setting, &sigmapoints);
    sad::SPKFD::VecXT x2(3);
    sad::SPKFD::MatXT P2(3, 3);
    spkf.SigmaPointReconstruct(sigmapoints, setting, x2, P2);
    EXPECT_NEAR(x1[0], x2[0], 1e-2);
    EXPECT_NEAR(x1[1], x2[1], 1e-2);
    EXPECT_NEAR(x1[2], x2[2], 1e-2);
    EXPECT_NEAR(P1(0, 0), P2(0, 0), 1e-2);
    EXPECT_NEAR(P1(1, 1), P2(1, 1), 1e-2);
    EXPECT_NEAR(P1(2, 2), P2(2, 2), 1e-2);

    LOG(INFO) << "setting: L wa0 wc0 wa wc\n"
              << setting.L << ' ' << setting.wa0 << ' ' << setting.wc0 << ' ' << setting.wa << ' ' << setting.wc;
    LOG(INFO) << "x1:" << x1.transpose();
    LOG(INFO) << "x2:" << x2.transpose();
    LOG(INFO) << "P1:\n" << P1;
    LOG(INFO) << "P2:\n" << P2;
    size_t i = 0;
    for (auto const& sp : sigmapoints) {
        LOG(INFO) << "No." << ++i << " point:" << sp.transpose();
    }
    SUCCEED();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
