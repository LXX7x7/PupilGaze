/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/


#include "bundleCalibration.h"


double bundleAdjustCalibration(std::vector<Observer>& observers, std::vector<Eigen::Matrix<double, 3, 1>>& points, bool fix_points)
{


    Problem problem;

    for( auto& observer : observers )
	{

        double* pose = observer.pose.data();

        int index = 0;
        for( auto& observation : observer.observations){

            // Each Residual block takes a point and a pose as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function =
                ReprojectionError::Create(observation);

            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     pose,
                                     pose+3,
                                     points[index].data() );
            index++;

        }
      if(observer.fix_rotation == 1){
          problem.SetParameterBlockConstant(pose);
        }
      if(observer.fix_translation == 1){
          problem.SetParameterBlockConstant(pose+3);
        }
    }

      if(fix_points == true){
        int index = 0;
        for(auto o : observers[0].observations){
            problem.SetParameterBlockConstant(points[index].data());
            index++;
            }
        }


    // Build and solve the problem.
    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    // options.parameter_tolerance = 1e-35;
    // options.function_tolerance = 1e-35;
    options.gradient_tolerance = 1e-35;
    // options.minimizer_progress_to_stdout = true;
    //options.logging_type = ceres::SILENT;
    // options.check_gradients = true;


    Solver::Summary summary;
    Solve(options, &problem, &summary);


    std::cout << summary.BriefReport() << "\n";
    // std::cout << summary.FullReport() << "\n";

    if( summary.termination_type != ceres::TerminationType::CONVERGENCE  )
	{
        std::cout << "Termination Error: " << ceres::TerminationTypeToString(summary.termination_type) << std::endl;
        return -1;
    }

    return summary.final_cost;

}

cv::Mat Find_rigid_transform(cv::Mat A, cv::Mat B)
{
	cv::Mat R;
	if (A.cols != B.cols)
	{
		return cv::Mat::zeros(1, 1, CV_32FC1);
	}
	int numPoint = A.cols;
	cv::Mat allone(numPoint, 1, CV_32FC1);
	allone = cv::Scalar(1);

	// 所有点的中心
	cv::Mat centroid_A = A*allone / (float)numPoint;
	cv::Mat centroid_B = B*allone / (float)numPoint;

	// 将所有点的中心移到原点
	cv::Mat AA = A - centroid_A * allone.t();
	cv::Mat BB = B - centroid_B * allone.t();

	cv::Mat H = AA * BB.t();

	cv::Mat w, u, v;
	cv::SVD::compute(H, w, u, v, cv::SVD::FULL_UV);

	R = v.t() * u.t();


	if (cv::determinant(R) < 0)
	{
		//return cv::Mat::eye(1, 1, CV_32FC1);
		v.rowRange(2, 3).colRange(0, 3) = v.rowRange(2, 3).colRange(0, 3) * -1;
		R = v.t() * u.t();
	}
	cv::Mat T = -R * centroid_A + centroid_B;

	cv::Mat TTAB(3, 4, CV_32FC1);

	R.copyTo(TTAB.rowRange(0, 3).colRange(0, 3));
	T.copyTo(TTAB.rowRange(0, 3).colRange(3, 4));

	return TTAB;
}