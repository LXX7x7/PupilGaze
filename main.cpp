#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "glog/logging.h"
#include "./src/api/gazeTracingM.h"
#include "./src/shared_modules/pupil_detectors/singleeyefitter/EyeModelFitter.h"
#include "./src/shared_modules/pupil_detectors/singleeyefitter/mathHelper.h"
#include "./src/shared_modules/calibration/testFunction.hpp"
#include "common/types.h"
#include "tool.hpp"
#include "geometry.h"
#include "state.h"
#include "ellseg_feature.h"

int main()
{
    // ========== 初始化参数 ==========
    std::string calibrate_file_name = "../calibration_file.yml";
    std::string ellseg_model_file = "../EllSeg/weights/ellseg_ritnet_v3.pt";
    cv::Mat gaze_point_sequence;  // 凝视点序列
    
    // ========== 设备对象 ==========
    Camera scene_camera;
    Camera eye_camera_right;
    Camera eye_camera_left;
    
    // ========== 状态控制标志 ==========
    bool calibration_flag = false;
    bool gaze_estimation_flag = false;
    
    // ========== 模型对象 ==========
    GeometryEyeModel eye_model_right;
    GeometryEyeModel eye_model_left;
    
    // ========== 跟踪器对象 ==========
    CGazeTrakingM gaze_tracker_right;
    CGazeTrakingM gaze_tracker_left;
    EllSegFeatureExtractor ellseg_feature_extractor;

    if(std::filesystem::exists(ellseg_model_file))
    {
        if(!ellseg_feature_extractor.Load(ellseg_model_file))
        {
            std::cerr<<"ELLSeg模型加载失败: "
                     <<ellseg_feature_extractor.LastError()<<std::endl;
        }
    }
    
    // ========== Kappa角存储 ==========
    KappaAngle kappa_angle;

    // ========== 主状态循环 ==========
    GazeState current_status = GazeState::INIT;
    while(true)
    {
        switch (current_status)
        {
        case GazeState::INIT:
            current_status = handleInit(calibrate_file_name,
                                      scene_camera,
                                      eye_camera_right,
                                      eye_camera_left,
                                      gaze_point_sequence);
            break;
            
        case GazeState::RUNNING:
            current_status = handleRunning(scene_camera,
                                         eye_camera_right,
                                         eye_camera_left,
                                         calibration_flag,
                                         gaze_estimation_flag);
            break;
            
        case GazeState::EYE_MODEL:
            current_status = handleEyeModel(scene_camera,
                                          eye_camera_right,
                                          eye_camera_left,
                                          gaze_tracker_right,
                                          gaze_tracker_left,
                                          eye_model_right,
                                          eye_model_left,
                                          &ellseg_feature_extractor);
            break;
            
        case GazeState::CALIBRATE_KAPPA:
            current_status = handleCalibrateKappa(scene_camera,
                                                eye_camera_right,
                                                eye_camera_left,
                                                gaze_point_sequence,
                                                eye_model_right,
                                                eye_model_left,
                                                calibration_flag);
            break;
            
        case GazeState::CALCULATE_KAPPA:
            current_status = handleCalculateKappa(kappa_angle);
            break;
            
        case GazeState::ESTIMATE_GAZE_POINT:
            current_status = handleEstimateGazePoint(scene_camera,
                                                   eye_camera_right,
                                                   eye_camera_left,
                                                   gaze_point_sequence,
                                                   eye_model_right,
                                                   eye_model_left,
                                                   gaze_estimation_flag);
            break;
            
        case GazeState::EXIT:
            current_status = handleExit(scene_camera,
                                      eye_camera_right,
                                      eye_camera_left);
            // 退出主循环
            return 0;
            
        default:
            std::cerr << "未知状态!" << std::endl;
            return -1;
        }
    }

    return 0;
}
