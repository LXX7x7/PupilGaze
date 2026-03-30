/**
 * @file state.h
 * @brief 状态机相关函数定义
 * @author LiuXin 
 * @date 2025-05-23
 */
// #include "common/types.h"
// #include "./src/api/gazeTracingM.h"
#include "geometry.h"
#include <filesystem>

#ifndef STATE_H
#define STATE_H




//相机类，包含内参，畸变系数，相机索引号等
class Camera 
{
public:
    Camera(int deviceId = -1, 
            const cv::Mat& intrinsics = cv::Mat::eye(3,3,CV_64F), 
            const cv::Mat& distortion = cv::Mat::zeros(5,1,CV_64F));
    Camera(const Camera&) = delete;            // 禁用拷贝构造
    Camera& operator=(const Camera&) = delete; // 禁用拷贝赋值
    //  获取相机内参矩阵
    void setIntrinsics(const cv::Mat& intrinsics)
    {
        intrinsics.copyTo(intrinsics_);
    };
    //  设置相机畸变系数，不允许修改
    const cv::Mat& getIntrinsics() const
    {
        return intrinsics_;
    };
    //  获取相机畸变系数
    void setDistortion(const cv::Mat& distortion)
    {
        distortion.copyTo(distCoeffs_);
    };
    const cv::Mat& getDistortion() const
    {
        return distCoeffs_;
    };
    //  获取相机ID
    void setDeviceId(int deviceId)
    {
        deviceId_ = deviceId;
    };
    int getDeviceId() const
    {
        return deviceId_;
    };
    //  获取相机外参
    void setExtrinsics(const cv::Mat& rotation,const cv::Mat& translation)
    {
        rotation.copyTo(rotation_);
        translation.copyTo(translation_);
    };
    const cv::Mat& getRotation() const
    {
        return rotation_;
    };
    const cv::Mat& getTranslation() const
    {
        return translation_;
    };
    // 读取帧
    bool readFrame(cv::Mat& frame)
    {
        if (!camera_.isOpened())
        {
            std::cerr <<"相机无法打开"<<std::endl;
            return false;
        }
        return camera_.read(frame);
    };
    // 帧写入接口
    bool writeFrame(const cv::Mat& frame)
    {
        if (!writer_.isOpened())
        {
            std::cerr <<"视频写入器未初始化"<<std::endl;
            return false;
        }
        writer_.write(frame);
        return true;
    };
    // 返回自身引用，使得相机对象可以连续调用，保证多次调用时使用同一个视频流
    Camera& createCamera()
    {
        if (!camera_.isOpened())
        {
            camera_.open(deviceId_);
            if(!camera_.isOpened())
            {
                std::cerr <<"无法打开相机设备："<<deviceId_<<std::endl;
            }
        }
        return *this;
    }
    // 视频写入配置，初始化writer_
    bool startRecording(const std::string& outputPath)
    {
        if(!camera_.isOpened()) return false;

        cv::Size frameSize(
            static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
        writer_.open(outputPath, cv::VideoWriter::fourcc('M','J','P','G'), 30, frameSize);
        return writer_.isOpened();
    }

    // 资源释放
    void release()
    {
        if(camera_.isOpened())
        {
            camera_.release();
            std::cout<<"释放相机"<<deviceId_<<std::endl;
        }
        if(writer_.isOpened())
        {
            writer_.release();
            std::cout<<"关闭相机"<<deviceId_<<"的写入器"<<std::endl;
        }
        
    }


private:
    int deviceId_;                  // 相机ID
    cv::VideoCapture camera_;       // 相机对象
    cv::VideoWriter writer_;        // 视频写入对象
    cv::Mat intrinsics_;            // 相机内参矩阵
    cv::Mat distCoeffs_;            // 畸变系数
    cv::Mat rotation_;              // 相机外参的旋转矩阵
    cv::Mat translation_;           // 相机外参的平移向量
    
};

class GeometryEyeModel {
public:
    // 构造函数
    GeometryEyeModel();
    
    // 带参数构造函数，直接设置光轴、眼球中心和瞳孔中心
    GeometryEyeModel(const Geometry_Vector& optical_axis, 
             const Geometry_Point& eye_center, 
             const Geometry_Point& pupil_center);
    
    // 初始化方法，从Detector3DResult中提取关键信息
    void InitializeFromResult(const Detector3DResult& eye_result);
    
    // 获取光轴向量
    Geometry_Vector GetOpticalAxis() const;
    
    // 获取眼球模型中心点
    Geometry_Point GetEyeCenter() const;
    
    // 获取瞳孔中心点
    Geometry_Point GetPupilCenter() const;

    // 计算视轴方向
    void  CalculateVisualAxis(KappaAngle kappa_angle);

    // 获取视轴方向向量
    Geometry_Vector GetVisualAxis() const;

private:
    Geometry_Vector optical_axis_;  // 光轴方向向量
    Geometry_Vector visual_axis_;
    Geometry_Point eye_center_;    // 眼球模型中心点
    Geometry_Point pupil_center_;  // 瞳孔中心点
};

//状态机的全局变量
enum GazeState{
    INIT,                   //初始化状态
    RUNNING,                //运行状态
    EYE_MODEL,              //构建眼球模型状态
    CALIBRATE_KAPPA,        //标定kappa角状态
    CALCULATE_KAPPA,        //计算kappa角状态
    ESTIMATE_GAZE_POINT,    //估计凝视点位置状态
    EXIT                    //退出状态
};
/**
 * @brief 读入参数，相机初始化，打开帧，打开视频读写
 * 
 * @param filename 
 * @param scene_camera 
 * @param eye_camera_right 
 * @param eye_camera_left 
 * @param gaze_point_sequence 
 * @return GazeState 
 */
GazeState handleInit(const std::string filename, 
        Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
        cv::Mat& gaze_point_sequence);
/**
 * @brief 运行状态，按键选择各种状态，同时保持显示画面
 * 
 * @param scene_camera 
 * @param eye_camera_right 
 * @param eye_camera_left 
 * @param calibration_flag 
 * @param gaze_estimation_flag 
 * @return GazeState 
 */
GazeState handleRunning(Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
                        bool& calibration_flag,bool& gaze_estimation_flag);
/**
 * @brief 重建眼球模型
 * 
 * @param scene_camera        场景相机
 * @param eye_camera_right    右眼相机
 * @param eye_camera_left     左眼相机
 * @param gaze_tracker_right  右眼追踪器
 * @param gaze_tracker_left   左眼追踪器
 * @param eye_model_right     右眼球模型
 * @param eye_model_left      左眼球模型
 * @return * GazeState 
 */
GazeState handleEyeModel(    
    Camera& scene_camera, 
    Camera& eye_camera_right,
    Camera& eye_camera_left,
    CGazeTrakingM& gaze_tracker_right,
    CGazeTrakingM& gaze_tracker_left,
    GeometryEyeModel& eye_model_right,
    GeometryEyeModel& eye_model_left);
/**
 * @brief 标定kappa角
 * 
 * @param scene_camera        场景相机
 * @param eye_camera_right    右眼相机
 * @param eye_camera_left     左眼相机
 * @param gaze_point_sequence 凝视点坐标
 * @param eye_model_right     右眼球模型
 * @param eye_model_left      左眼球模型
 * @param calibration_flag    开始标定标志符
 * @return GazeState 
 */
GazeState handleCalibrateKappa(
    Camera& scene_camera, Camera& eye_camera_right,Camera& eye_camera_left,
    cv::Mat& gaze_point_sequence,
    GeometryEyeModel& eye_model_right,
    GeometryEyeModel& eye_model_left,
    bool& calibration_flag
);
//计算kappa角
GazeState handleCalculateKappa(KappaAngle& kappa_angle);
//估计凝视点位置
GazeState handleEstimateGazePoint(
        Camera& scene_camera,Camera& eye_camera_right,Camera& eye_camera_left,
        cv::Mat& gaze_point_sequence,
        GeometryEyeModel& eye_model_right,
        GeometryEyeModel& eye_model_left,
        bool& gaze_estimation_flag       
);
//退出程序
GazeState handleExit(Camera& scene_camera, 
                    Camera& eye_camera_right,
                    Camera& eye_camera_left) ;
#endif