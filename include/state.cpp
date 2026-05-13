/**
 * @file state.cpp
 * @brief 状态机相关函数实现
 * @author LiuXin 
 * @date 2025-05-23
 */

#include "state.h"
#include <filesystem>

//相机类的实现
//构造函数
Camera::Camera(int deviceId, const cv::Mat& intrinsics, const cv::Mat& distCoeffs):
    deviceId_(deviceId), intrinsics_(intrinsics), distCoeffs_(distCoeffs){}

//眼球类的实现
// 默认构造函数
GeometryEyeModel::GeometryEyeModel() 
    : optical_axis_(Geometry_Vector(0.0, 0.0, 0.0)),
      eye_center_(Geometry_Point(0.0, 0.0, 0.0)), 
      pupil_center_(Geometry_Point(0.0, 0.0, 0.0)) {}

// 带参构造函数
GeometryEyeModel::GeometryEyeModel(const Geometry_Vector& optical_axis, 
                   const Geometry_Point& eye_center, 
                   const Geometry_Point& pupil_center)
    : optical_axis_(optical_axis), 
      eye_center_(eye_center), 
      pupil_center_(pupil_center) {}

// 从Detector3DResult初始化
void GeometryEyeModel::InitializeFromResult(const Detector3DResult& eye_result) {
    // 从Detector3DResult中获取数据
    // 眼球模型中心
    if(eye_result.sphere.center.size()==0||eye_result.sphere.center.size()!=3)
    {
        throw std::invalid_argument("Invalid detector result");
    }
    cv::Mat eye_center_mat = (cv::Mat_<double>(3,1)<<eye_result.sphere.center[0],
                                                    eye_result.sphere.center[1],
                                                    eye_result.sphere.center[2]);
    cv::Mat pupil_center_mat = (cv::Mat_<double>(3,1)<<eye_result.circle.center[0],
                                                        eye_result.circle.center[1],
                                                        eye_result.circle.center[2]);
    
    // 转换为Geometry_Point
    eye_center_= eye_center_.Mat2Point(eye_center_mat);
    pupil_center_=pupil_center_.Mat2Point(pupil_center_mat);
    
    // 计算光轴：眼球模型中心指向瞳孔中心的方向向量
    optical_axis_.Assign(
        pupil_center_.x - eye_center_.x,
        pupil_center_.y - eye_center_.y,
        pupil_center_.z - eye_center_.z
    );
    
    // 归一化光轴向量
    double length = optical_axis_.Magnitude();
    if (length > 1e-6) {  // 防止除以零
        optical_axis_.Assign(
            optical_axis_.x / length,
            optical_axis_.y / length,
            optical_axis_.z / length
        );
    }
}

// 获取光轴
Geometry_Vector GeometryEyeModel::GetOpticalAxis() const {
    return optical_axis_;
}

// 获取眼球中心
Geometry_Point GeometryEyeModel::GetEyeCenter() const {
    return eye_center_;
}

// 获取瞳孔中心
Geometry_Point GeometryEyeModel::GetPupilCenter() const {
    return pupil_center_;
}

// 计算视轴方向
void GeometryEyeModel::CalculateVisualAxis(KappaAngle kappa_angle)
{   
    //光轴方向
    Geometry_Vector optical_axis=GetOpticalAxis();
    double phi = AnglePhi(optical_axis);
    double theta = AngleTheta(optical_axis);

    //kappa角补偿
    phi -= kappa_angle.alpha;
    theta -= kappa_angle.beta;

    //计算视轴方向向量
    visual_axis_.Assign(
        std::cos(phi),
        std::sin(phi) * std::sin(theta),
        std::sin(phi) * std::cos(theta)
    );
    double length = visual_axis_.Magnitude();
    if(length>1e-6)
    {
        visual_axis_.x /= length;
        visual_axis_.y /= length;
        visual_axis_.z /= length;
    }
}

// 获取视轴方向向量
Geometry_Vector GeometryEyeModel::GetVisualAxis() const
{
    return visual_axis_;
}

GazeState handleInit(const std::string filename, 
        Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
        cv::Mat& gaze_point_sequence)
{

    // ==========参数加载模块==========
    cv::FileStorage calibration_file(filename, cv::FileStorage::READ);
    if (!calibration_file.isOpened())
    {
        std::cout << "读取标定文件失败！" << std::endl;
        return EXIT;
    }
    //场景相机:scene_camera
    cv::Mat scene_camera_intrinsic_matrix, scene_camera_distortion_coefficients;
    calibration_file["scene_camera_intrinsic_matrix"]>>scene_camera_intrinsic_matrix;
    calibration_file["scene_camera_distortion_coefficients"]>>scene_camera_distortion_coefficients;
    //右眼相机:eye_camera_right
    cv::Mat eye_camera_intrinsic_matrix_right, eye_camera_distortion_coefficients_right;
    calibration_file["eye_camera_intrinsic_matrix_right"]>>eye_camera_intrinsic_matrix_right;
    calibration_file["eye_camera_distortion_coefficients_right"]>>eye_camera_distortion_coefficients_right;
    //右眼相机到场景相机的变换矩阵
    cv::Mat eye_camera_right_to_scene_camera_rotation, eye_camera_right_to_scene_camera_translation;
    calibration_file["eye_camera_right_to_scene_camera_rotation"]>>eye_camera_right_to_scene_camera_rotation;
    calibration_file["eye_camera_right_to_scene_camera_translation"]>>eye_camera_right_to_scene_camera_translation;
    //左眼相机:eye_camera_left
    cv::Mat eye_camera_intrinsic_matrix_left, eye_camera_distortion_coefficients_left;
    calibration_file["eye_camera_intrinsic_matrix_left"]>>eye_camera_intrinsic_matrix_left;
    calibration_file["eye_camera_distortion_coefficients_left"]>>eye_camera_distortion_coefficients_left; 
    //左眼相机到场景相机的变换矩阵
    cv::Mat eye_camera_left_to_scene_camera_rotation, eye_camera_left_to_scene_camera_translation;
    calibration_file["eye_camera_left_to_scene_camera_rotation"]>>eye_camera_left_to_scene_camera_rotation;
    calibration_file["eye_camera_left_to_scene_camera_translation"]>>eye_camera_left_to_scene_camera_translation;      
    //相机索引号
    int scene_camera_id, eye_camera_right_id, eye_camera_left_id;
    calibration_file["scene_camera_id"]>>scene_camera_id;
    calibration_file["eye_camera_right_id"]>>eye_camera_right_id;
    calibration_file["eye_camera_left_id"]>>eye_camera_left_id;
    //初始化
    scene_camera.setDeviceId(scene_camera_id);
    scene_camera.setIntrinsics(scene_camera_intrinsic_matrix);
    scene_camera.setDistortion(scene_camera_distortion_coefficients);

    eye_camera_right.setDeviceId(eye_camera_right_id);
    eye_camera_right.setIntrinsics(eye_camera_intrinsic_matrix_right);
    eye_camera_right.setDistortion(eye_camera_distortion_coefficients_right);

    eye_camera_left.setDeviceId(eye_camera_left_id);
    eye_camera_left.setIntrinsics(eye_camera_intrinsic_matrix_left);
    eye_camera_left.setDistortion(eye_camera_distortion_coefficients_left);
    eye_camera_right.setExtrinsics(eye_camera_right_to_scene_camera_rotation, eye_camera_right_to_scene_camera_translation);
    eye_camera_left.setExtrinsics(eye_camera_left_to_scene_camera_rotation, eye_camera_left_to_scene_camera_translation); 
    eye_camera_right.createCamera();
    eye_camera_left.createCamera();
    calibration_file["gaze_point_sequence"]>>gaze_point_sequence;
    calibration_file.release();
    return GazeState::RUNNING;
}


GazeState handleRunning(
    Camera& scene_camera, 
    Camera& eye_camera_right,
    Camera& eye_camera_left,
    bool& calibration_flag,
    bool& gaze_estimation_flag) {

    // 打开摄像头，保证只进行一次初始化
    scene_camera.createCamera();
    eye_camera_right.createCamera();
    eye_camera_left.createCamera();

    // 合成图像
    cv::Mat result_img(480, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::resizeWindow("result", 1920, 480);

    while (true) {
        // 读取帧
        cv::Mat scene_frame,eye_frame_right, eye_frame_left;
        bool read_susccess = true;
        read_susccess &= scene_camera.readFrame(scene_frame);
        read_susccess &= eye_camera_right.readFrame(eye_frame_right);
        read_susccess &= eye_camera_left.readFrame(eye_frame_left);

        // ARUco 检测
        cv::Mat screen_to_scene_rot, screen_to_scene_trans;
        GeometryDetectAruco(scene_frame, 
                            scene_camera.getIntrinsics(), 
                            scene_camera.getDistortion(), 
                            screen_to_scene_rot, 
                            screen_to_scene_trans);
        DrawCameraAxes(scene_frame, scene_camera.getIntrinsics());

        // 合成画面
        cv::Mat resized_eye_right, resized_eye_left;
        cv::resize(eye_frame_right, resized_eye_right, cv::Size(640, 480));
        cv::resize(eye_frame_left, resized_eye_left, cv::Size(640, 480));

        resized_eye_right.copyTo(result_img(cv::Rect(0, 0, 640, 480)));
        resized_eye_left.copyTo(result_img(cv::Rect(640, 0, 640, 480)));
        cv::Mat resized_scene;
        cv::resize(scene_frame, resized_scene, cv::Size(640, 480));
        resized_scene.copyTo(result_img(cv::Rect(1280, 0, 640, 480)));
        // 显示合成画面
        cv::imshow("result", result_img);
        if (scene_camera.startRecording("../output/scene_video.avi"))
        {
            scene_camera.writeFrame(scene_frame);   
        }
        if (eye_camera_right.startRecording("../output/right_eye_video.avi"))
        {
            eye_camera_right.writeFrame(eye_frame_right);
        }
        if (eye_camera_left.startRecording("../output/left_eye_video.avi"))
        {
            eye_camera_left.writeFrame(eye_frame_left);
        }

        // 按键检测
        char key = cv::waitKey(30);
        switch (key)
        {
        case 'q':
            std::cout<<"退出程序"<<std::endl;
            return GazeState::EXIT;
        case 'e':
            gaze_estimation_flag = true;
            std::cout<<"开始凝视点估计"<<std::endl;
            return GazeState::ESTIMATE_GAZE_POINT;
        case 'r':
            std::cout<<"重建眼球模型"<<std::endl;
            return GazeState::EYE_MODEL;
        case 'c':
            calibration_flag = true;
            std::cout<<"开始kappa角标定"<<std::endl;
            return GazeState::CALIBRATE_KAPPA;
        case 'k':
            std::cout<<"开始计算kappa"<<std::endl;
            return GazeState::CALCULATE_KAPPA;
        default: ;
        }

    }

    // 资源释放（正常情况下不会执行到这里）
    return GazeState::EXIT;
}

GazeState handleEyeModel(    
    Camera& scene_camera, 
    Camera& eye_camera_right,
    Camera& eye_camera_left,
    CGazeTrakingM& gaze_tracker_right,
    CGazeTrakingM& gaze_tracker_left,
    GeometryEyeModel& eye_model_right,
    GeometryEyeModel& eye_model_left,
    EllSegFeatureExtractor* ellseg_feature_extractor)
{
    // 读取帧
    cv::Mat scene_frame,eye_frame_right, eye_frame_left;
    bool read_susccess = true;
    read_susccess &= scene_camera.readFrame(scene_frame);
    read_susccess &= eye_camera_right.readFrame(eye_frame_right);
    read_susccess &= eye_camera_left.readFrame(eye_frame_left);
    if(!read_susccess || scene_frame.empty() || eye_frame_right.empty() || eye_frame_left.empty())
    {
        std::cerr<<"眼球模型重建时读取图像失败"<<std::endl;
        return GazeState::RUNNING;
    }

    // ELLSeg 只负责从眼部图像中提取轻量化分割、latent 和椭圆特征。
    if(ellseg_feature_extractor != nullptr && ellseg_feature_extractor->IsLoaded())
    {
        EllSegFeature ellseg_right_feature;
        EllSegFeature ellseg_left_feature;
        const bool is_ellseg_right = ellseg_feature_extractor->Extract(eye_frame_right, ellseg_right_feature);
        const std::string ellseg_right_error = ellseg_feature_extractor->LastError();
        const bool is_ellseg_left = ellseg_feature_extractor->Extract(eye_frame_left, ellseg_left_feature);
        const std::string ellseg_left_error = ellseg_feature_extractor->LastError();

        if(is_ellseg_right && ellseg_right_feature.pupil.valid)
        {
            std::cout<<"ELLSeg右眼瞳孔中心: ("
                     <<ellseg_right_feature.pupil.center.x<<", "
                     <<ellseg_right_feature.pupil.center.y<<")"<<std::endl;
        }
        if(is_ellseg_left && ellseg_left_feature.pupil.valid)
        {
            std::cout<<"ELLSeg左眼瞳孔中心: ("
                     <<ellseg_left_feature.pupil.center.x<<", "
                     <<ellseg_left_feature.pupil.center.y<<")"<<std::endl;
        }
        if(!is_ellseg_right)
        {
            std::cerr<<"ELLSeg右眼特征提取失败: "<<ellseg_right_error<<std::endl;
        }
        if(!is_ellseg_left)
        {
            std::cerr<<"ELLSeg左眼特征提取失败: "<<ellseg_left_error<<std::endl;
        }
    }

    // 右眼处理
    cv::Mat eye_frame_right_gray;
    cv::cvtColor(eye_frame_right, eye_frame_right_gray, cv::COLOR_BGR2GRAY);
    gaze_tracker_right.ReadImage( eye_frame_right_gray, eye_frame_right_gray);
    Detector3DResult eye_result_right;
    bool is_pupil_detected_right = gaze_tracker_right.CalcGazePointPosition3D(eye_result_right, eye_frame_right, eye_frame_right);

    // 左眼处理
    cv::Mat eye_frame_left_gray;
    cv::cvtColor(eye_frame_left, eye_frame_left_gray, cv::COLOR_BGR2GRAY);
    gaze_tracker_left.ReadImage(eye_frame_left_gray, eye_frame_left_gray);
    Detector3DResult eye_result_left;
    bool is_pupil_detected_left = gaze_tracker_left.CalcGazePointPosition3D(eye_result_left, eye_frame_left, eye_frame_left);

    //眼球模型初始化
    eye_model_right.InitializeFromResult(eye_result_right);
    eye_model_left.InitializeFromResult(eye_result_left);

    return GazeState::RUNNING;
}

GazeState handleCalibrateKappa(
    Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
    cv::Mat& gaze_point_sequence,
    GeometryEyeModel& eye_model_right,
    GeometryEyeModel& eye_model_left,
    bool& calibration_flag
)
{
    while(calibration_flag)
    {
                // 读取帧
        cv::Mat scene_frame,eye_frame_right, eye_frame_left;
        bool read_susccess = true;
        read_susccess &= scene_camera.readFrame(scene_frame);
        read_susccess &= eye_camera_right.readFrame(eye_frame_right);
        read_susccess &= eye_camera_left.readFrame(eye_frame_left);

        //初始化
        static int  gaze_point_index = 0;
        static bool is_gaze_point_selected = false;
        static DataList data_list;

        //Aruco检测
        cv::Mat scene_camera_intrinsic_matrix=scene_camera.getIntrinsics();
        cv::Mat scene_camera_distortion_coefficients=scene_camera.getDistortion();
        cv::Mat screen_to_scene_camera_rotation, screen_to_scene_camera_translation;
        GeometryDetectAruco(scene_frame,scene_camera_intrinsic_matrix,
                        scene_camera_distortion_coefficients,screen_to_scene_camera_rotation, 
                        screen_to_scene_camera_translation); 
        DrawCameraAxes(scene_frame,scene_camera_intrinsic_matrix);

        //数据采集
        if(!screen_to_scene_camera_rotation.empty()&&!screen_to_scene_camera_translation.empty())
        {
            //只有当选择好凝视点后开始进行标定
            if(is_gaze_point_selected)
            {
                //凝视点            
                cv::Mat gaze_point_in_screen=gaze_point_sequence.row(gaze_point_index).clone();
                cv::Mat gaze_point_in_scene_camera=screen_to_scene_camera_rotation*gaze_point_in_screen+screen_to_scene_camera_translation;
                Geometry_Point geometry_gaze_point;
                geometry_gaze_point=geometry_gaze_point.Mat2Point(gaze_point_in_scene_camera);
                //眼球中心
                Geometry_Point geometry_right_eye_center=eye_model_right.GetEyeCenter();
                Geometry_Point geometry_left_eye_center=eye_model_left.GetEyeCenter();
                //瞳孔中心
                Geometry_Point geometry_left_pupil_center=eye_model_left.GetPupilCenter();
                Geometry_Point geometry_right_pupil_center=eye_model_right.GetPupilCenter();

                //保存数据到csv文件中
                Geometry_Point geometry_E;//默认的E，满足Data初始化参数
                data_list.push_back(Data(geometry_left_eye_center,geometry_right_eye_center,
                                            geometry_left_pupil_center,geometry_right_pupil_center,
                                            geometry_gaze_point,geometry_E));
                std::cout<<"已采集第"<<data_list.size()<<"组数据,当前凝视点为"<<gaze_point_index<<std::endl;
            }
        }    
        // 按键处理
        char key = cv::waitKey(30);
        switch (key) {
            case '1': case '2': case '3': case '4':
                gaze_point_index = key - '1';
                is_gaze_point_selected = true;
                std::cout<<"已选择凝视点"<<gaze_point_index<<std::endl;
                break;
            case 's':
                if(!data_list.empty()){
                    std::string data_file = "../data/gaze_point_"+std::to_string(gaze_point_index)+".csv";
                    SaveToCSV(data_file, data_list);
                    std::cout << "已保存 " << data_list.size() 
                                << " 组数据到 " << data_file << std::endl;
                    data_list.clear();     // 清空数据
                    is_gaze_point_selected = false; // 重置标志
                }
                break;
            case 'f':
                calibration_flag = false;
                return GazeState::RUNNING;
            default:
                break;
        }
    }


    return GazeState::CALIBRATE_KAPPA; //保持当前状态
}

GazeState handleCalculateKappa(KappaAngle& kappa_angle) {
    // ===== 数据准备 =====
    std::vector<DataList> all_data;  // 存储所有数据集
    const std::string result_file = "../results/kappa_result.csv";
    
    const int num_data_files = 4;
    for (int i=1; i<=num_data_files; ++i)
    {
        std::string data_file="../data/kappaAngle/gaze_point_"+std::to_string(i)+".csv";
        DataList data = LoadFromCSV(data_file);
        if(!data.empty())
        {
            all_data.push_back(data);
            std::cout << "已加载文件: " << data_file << " (" << data.size() << " 组数据)" << std::endl;
        }else{
            std::cout << "未找到文件: " << data_file << std::endl;
        }
    }

    if(all_data.empty())
    {
        std::cout << "未找到任何数据文件" << std::endl;
        return GazeState::EXIT;
    }
    kappa_angle = CalculateKappaAngle(all_data);
    SaveKappaAngleToCSV(result_file, kappa_angle);
    std::cout << "已保存结果到 " << result_file << std::endl;
    std::cout << "计算结果（角度值）:\n"
                << "右眼 alpha: " << RadianToAngle(kappa_angle.alpha) << "°\n"
                << "右眼 beta:  " << RadianToAngle(kappa_angle.beta) << "°\n"
                << "右眼 angle: " << RadianToAngle(kappa_angle.angle) << "°\n"
                << "左眼 alpha: " << RadianToAngle(kappa_angle.alpha_left) << "°\n"
                << "左眼 beta:  " << RadianToAngle(kappa_angle.beta_left) << "°\n"
                << "左眼 angle: " << RadianToAngle(kappa_angle.angle_left) << "°\n";    

    return GazeState::RUNNING; // 返回主运行状态
}

GazeState handleEstimateGazePoint(
        Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
        cv::Mat& gaze_point_sequence,
        GeometryEyeModel& eye_model_right,
        GeometryEyeModel& eye_model_left,
        bool& gaze_estimation_flag       
)
{
    while (gaze_estimation_flag)
    {
        // 读取帧
        cv::Mat scene_frame,eye_frame_right, eye_frame_left;
        bool read_susccess = true;
        read_susccess &= scene_camera.readFrame(scene_frame);
        read_susccess &= eye_camera_right.readFrame(eye_frame_right);
        read_susccess &= eye_camera_left.readFrame(eye_frame_left);

        //初始化
        static int  gaze_point_index = 0;
        static bool is_gaze_point_selected = false;
        static DataList data_list;
        std::string kappaFile="../results/kappa_result.csv";
        KappaAngle kappa_angle=LoadKappaAngleFromCSV(kappaFile);

        //Aruco检测
        cv::Mat scene_camera_intrinsic_matrix=scene_camera.getIntrinsics();
        cv::Mat scene_camera_distortion_coefficients=scene_camera.getDistortion();
        cv::Mat screen_to_scene_camera_rotation, screen_to_scene_camera_translation;
        GeometryDetectAruco(scene_frame,scene_camera_intrinsic_matrix,
                        scene_camera_distortion_coefficients,screen_to_scene_camera_rotation, 
                        screen_to_scene_camera_translation); 
        DrawCameraAxes(scene_frame,scene_camera_intrinsic_matrix);

        //数据采集
        if(!screen_to_scene_camera_rotation.empty()&&!screen_to_scene_camera_translation.empty())
        {
            //只有当选择好凝视点后开始进行标定
            if(is_gaze_point_selected)
            {
                //凝视点            
                cv::Mat gaze_point_in_screen=gaze_point_sequence.row(gaze_point_index).clone();
                cv::Mat gaze_point_in_scene_camera=screen_to_scene_camera_rotation*gaze_point_in_screen+screen_to_scene_camera_translation;
                Geometry_Point geometry_gaze_point;
                geometry_gaze_point=geometry_gaze_point.Mat2Point(gaze_point_in_scene_camera);
                //眼球中心
                Geometry_Point geometry_right_eye_center=eye_model_right.GetEyeCenter();
                Geometry_Point geometry_left_eye_center=eye_model_left.GetEyeCenter();
                //瞳孔中心
                Geometry_Point geometry_left_pupil_center=eye_model_left.GetPupilCenter();
                Geometry_Point geometry_right_pupil_center=eye_model_right.GetPupilCenter();

                //视线
                Geometry_Line visual_line_in_cam_left;
                eye_model_left.CalculateVisualAxis(kappa_angle);
                visual_line_in_cam_left.Assign(geometry_left_pupil_center,eye_model_left.GetVisualAxis());
                Geometry_Line visual_line_in_cam_right;
                eye_model_right.CalculateVisualAxis(kappa_angle);
                visual_line_in_cam_right.Assign(geometry_right_pupil_center,eye_model_right.GetVisualAxis());

                //估计凝视点
                Geometry_Point estimate_gaze_point = EstimateGazePoint(scene_frame,
                                                            visual_line_in_cam_left,
                                                            visual_line_in_cam_right,
                                                            scene_camera.getIntrinsics(),
                                                            scene_camera.getDistortion());

                //保存数据到csv文件中
                data_list.push_back(Data(geometry_left_eye_center,geometry_right_eye_center,
                                            geometry_left_pupil_center,geometry_right_pupil_center,
                                            geometry_gaze_point,estimate_gaze_point));
                std::cout<<"已采集第"<<data_list.size()<<"组数据,当前凝视点为"<<gaze_point_index<<std::endl;
            }
        }    
        // 按键处理
        char key = cv::waitKey(30);
        switch (key) {
            case '1': case '2': case '3': case '4':
                gaze_point_index = key - '1';
                is_gaze_point_selected = true;
                std::cout<<"已选择凝视点"<<gaze_point_index<<std::endl;
                break;
            case 's':
                if(!data_list.empty()){
                    std::string data_file = "../data/gazePoint/gaze_point_"+std::to_string(gaze_point_index)+".csv";
                    SaveToCSV(data_file, data_list);
                    std::cout << "已保存 " << data_list.size() 
                            << " 组数据到 " << data_file << std::endl;
                    data_list.clear();     // 清空数据
                    is_gaze_point_selected = false; // 重置标志
                }
                break;
            case 'f':
                gaze_estimation_flag  = false;
                return GazeState::RUNNING;
            default:
                break;
        } 
    }

    return GazeState::RUNNING;

}

GazeState handleExit(Camera& scene_camera, 
                    Camera& eye_camera_right,
                    Camera& eye_camera_left) 
{
    // 释放相机资源
    scene_camera.release();
    eye_camera_right.release();
    eye_camera_left.release();

    // 关闭所有OpenCV窗口
    cv::destroyAllWindows();

    // 打印退出信息
    std::cout << "\n=== 资源释放报告 ===" << std::endl;
    std::cout << "1. 已释放场景相机（ID:" << scene_camera.getDeviceId() << "）" << std::endl;
    std::cout << "2. 已释放右眼相机（ID:" << eye_camera_right.getDeviceId() << "）" << std::endl;
    std::cout << "3. 已释放左眼相机（ID:" << eye_camera_left.getDeviceId() << "）" << std::endl;
    std::cout << "4. 已关闭所有图像窗口" << std::endl;
    std::cout << "=== 程序安全退出 ===" << std::endl;

    return GazeState::EXIT;
}
