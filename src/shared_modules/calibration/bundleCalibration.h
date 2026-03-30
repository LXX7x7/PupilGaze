/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/


#include "common.h"
#include <vector>
#include <cstdio>
#include <limits>
#include "opencv2/opencv.hpp"

#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include "ceres/Fixed3DNormParametrization.h"
#include "ceres/EigenQuaternionParameterization.h"
#include "ceres/CeresUtils.h"
#include "math/distance.h"



using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


struct ReprojectionError {
	ReprojectionError(Eigen::Matrix<double, 3, 1> observed_point)
      : observed_point(observed_point) {}

  template <typename T>
  bool operator()(const T* const orientation,
                  const T* const translation,
                  const T* const point,
                  T* residuals) const {

        T p[3];


        // convetional order rot and then trans
        // ceres::AngleAxisRotatePoint(orientation, point, p);
        // // pose[3,4,5] are the translation.
        // p[0] += translation[0];
        // p[1] += translation[1];
        // p[2] += translation[2];


        // unconvetional reverse order trans then rot.
        T tp[3];
        // pose[3,4,5] are the translation.
        tp[0] = point[0] + translation[0];
        tp[1] = point[1] + translation[1];
        tp[2] = point[2] + translation[2];
        ceres::AngleAxisRotatePoint(orientation, tp, p);

        // Normalize / project back to unit sphere
        T s = sqrt( p[0]*p[0] + p[1]*p[1] + p[2]*p[2]  );
        p[0] /= s;
        p[1] /= s;
        p[2] /= s;


        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - T(observed_point[0]);
		residuals[1] = p[1] - T(observed_point[1]);
		residuals[2] = p[2] - T(observed_point[2]);
		//residuals[0] = 10. * (p[0] - T(observed_point[0]));
		//residuals[1] = 10. * (p[1] - T(observed_point[1]));
		//residuals[2] = 10. * (p[2] - T(observed_point[2]));

        return true;
  }

// Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Matrix<double, 3, 1> observed_point)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 3, 3, 3, 3>(new ReprojectionError(observed_point)));
  }

  Eigen::Matrix<double, 3, 1> observed_point;
};

double bundleAdjustCalibration(std::vector<Observer>& observers, std::vector<Eigen::Matrix<double, 3, 1>>& points, bool fix_points);

//A-->B
//*******************************************************
//功    能： 【刚体变换】
//输入参数： 
//    [in]    A                眼睛的视线向量，3*N
//    [in]    B                标定点的三维坐标，  3*N
//返 回 值：
//     3*4 矩阵，最后一列为平移量，前三列为旋转矩阵；   代表A-->B 的变换
//修改日志：
//*******************************************************
cv::Mat Find_rigid_transform(cv::Mat A, cv::Mat B);