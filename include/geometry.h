/**
 * @file geometry.h
 * @brief 眼动监测项目的相关函数定义
 * @author LiuXin 
 * @date 2025-05-16
 */
#include <cmath>
#include <functional>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include "common/types.h"
#include "./src/api/gazeTracingM.h"



#ifndef GEOMETRY_H
#define GEOMETRY_H

/** @struct Geometry_Point 
 * @brief 定义一个点结构体
 */
struct Geometry_Point {
    double x;
    double y;
    double z;
    
    Geometry_Point();
    Geometry_Point(double x, double y, double z);
    void Assign(double x, double y, double z);
    Geometry_Point Mat2Point(const cv::Mat& mat);
    cv::Mat Point2Mat(const Geometry_Point& p);
};

/** @struct Geometry_Vector 
 * @brief 一个向量结构体和向量相关运算
 */
struct Geometry_Vector {
    double x;
    double y;
    double z;
    
    Geometry_Vector();
    Geometry_Vector(double x, double y, double z);
    void Assign(double x, double y, double z);
    //void Assign(const Geometry_Point& p1, const Geometry_Point& p2);
    double Magnitude() const;
    double DotProduct(const Geometry_Vector& v) const;
    Geometry_Vector CrossProduct(const Geometry_Vector& v) const;
    Geometry_Vector operator+(const Geometry_Vector& v) const;
    Geometry_Vector operator-(const Geometry_Vector& v) const;
    Geometry_Vector operator*(double s) const;
     // 补充标量乘法友元函数（支持标量在前）
     friend Geometry_Vector operator*(double scalar, const Geometry_Vector& v);
};

struct Geometry_Line {
    Geometry_Point start; //直线起点
    Geometry_Point end; //直线终点
    Geometry_Vector direction; //直线方向向量
    
    Geometry_Line();
    Geometry_Line(const Geometry_Point& s, const Geometry_Point& e);
    void Assign(const Geometry_Point& s, const Geometry_Point& e);
    void Assign(const Geometry_Point& s, const Geometry_Vector& v);
};


//使用外部声明定义坐标系
extern const Geometry_Vector X_AXIS;
extern const Geometry_Vector Y_AXIS;
extern const Geometry_Vector Z_AXIS;

/** @brief 角度计算相关函数 */
//向量与x轴之间的夹角, 单位：弧度
double AnglePhi(const Geometry_Vector& v);
//向量在yz平面上的投影向量与z轴的夹角，单位弧度
double AngleTheta(const Geometry_Vector& v);
double AngleAlpha(const Geometry_Vector& v1, const Geometry_Vector& v2);
double AngleBeta(const Geometry_Vector& v1, const Geometry_Vector& v2);
/** @brief 弧度转角度 
 * @param radian 弧度
 * @param decimals 保留小数位数，默认为5
*/
double RadianToAngle(double radian,int decimals=5);

/** @struct KappaAngle 
 * @brief Kappa角结构体
 * @details 描述光轴与视轴之间的夹角，支持单目/双目两种计算模式
 * @param alpha      光轴和视轴分别与x轴夹角之差(单眼)，单位：弧度
 * @param beta       光轴和视轴在yz平面与z轴夹角之差(单眼)，单位：弧度
 * @param angle      光轴与视轴的夹角(单眼)，单位：弧度
 * @param alpha_left 左眼光轴和视轴x轴夹角差(双目)，单位：弧度
 * @param beta_left  左眼光轴和视轴yz平面夹角差(双目)，单位：弧度
 * @param angle_left 左眼光轴与视轴夹角(双目)，单位：弧度
 * @note 所有角度参数均以弧度为单位，可用[RadianToAngle()](file:///home/lx/kappa_calibration_opencv4/geometry.h#L85-L85)转换为角度值
 */
struct KappaAngle{
    double alpha; //光轴和视轴分别与x轴夹角之差，默认为单眼，单位：弧度
    double beta; //光轴和视轴分别在yz平面上与z轴的夹角之差，默认为单眼，单位：弧度
    double angle; //光轴和视轴的夹角，单位：弧度
    double alpha_left; //左眼光轴和视轴分别与x轴夹角之差，双目估计时用，单位：弧度
    double beta_left; //左眼光轴和视轴分别在yz平面上与z轴的夹角之差，双目估计时用，单位：弧度
    double angle_left; //左眼光轴和视轴分别与x轴夹角之差，双目估计时用，单位：弧度
    KappaAngle();
    KappaAngle(double alpha, double beta,double angle);
    void Assign(double alpha, double beta,double angle);
    //左眼
    KappaAngle(double alpha, double beta,double angle,double alpha_left, double beta_left,double angle_left);
    void Assign(double alpha, double beta,double angle,double alpha_left, double beta_left,double angle_left);
};

/** @struct GazePoint
 * @brief  凝视点结构体
 * @details 凝视点在标定板坐标系下的物理坐标
 */
struct GazePoint{
    double gaze_point_x_physical;
    double gaze_point_y_physical;
    double gaze_point_z_physical;
    GazePoint();
    GazePoint(double gaze_point_x_physical,double gaze_point_y_physical,double gaze_point_z_physical):
        gaze_point_x_physical(gaze_point_x_physical),
        gaze_point_y_physical(gaze_point_y_physical),
        gaze_point_z_physical(gaze_point_z_physical){};
    void Assign(double gaze_point_x_physical,double gaze_point_y_physical,double gaze_point_z_physical);
};


// 单份数据结构
/** @struct Data
 * @brief 单份数据结构
 * @param o 眼球模型中心
 * @param p 瞳孔中心
 * @param ol 左眼眼球模型中心
 * @param pl 左眼瞳孔中心
 * @param F 凝视点
 * @param E 估计凝视点
 */
struct Data {
    Geometry_Point o; //眼球模型中心
    Geometry_Point p; //瞳孔中心
    Geometry_Point ol; //左眼眼球模型中心
    Geometry_Point pl; //左眼瞳孔中心
    Geometry_Point F;   //凝视点
    Geometry_Point E;   //估计凝视点
    Data();
    Data(const Geometry_Point& o, const Geometry_Point& p, const Geometry_Point& F, const Geometry_Point& E);
    Data(const Geometry_Point& o, const Geometry_Point& p,
         const Geometry_Point& ol, const Geometry_Point& pl, 
         const Geometry_Point& F,   const Geometry_Point& E);
    void Assign(const Geometry_Point& o, const Geometry_Point& p, const Geometry_Point& F, const Geometry_Point& E);
    void Assign(const Geometry_Point& o, const Geometry_Point& p, 
                const Geometry_Point& ol, const Geometry_Point& pl, 
                const Geometry_Point& F, const Geometry_Point& E);
};

//定义数据容器用于存储多组数据
typedef std::vector<Data> DataList;


//单个data计算Kappa角
KappaAngle CalculateKappaAngle(const Data& data);
//一组数据计算kappa角并取均值
KappaAngle CalculateKappaAngle(const DataList& data);
//多组数据计算kappa角并取均值
KappaAngle CalculateKappaAngle(const std::vector<DataList>& data);
//将Kappa角保存到CSV文件中
void SaveKappaAngleToCSV(const std::string& filename, const KappaAngle& kappa_angle);
//从CSV文件中读取kappa角
KappaAngle LoadKappaAngleFromCSV(const std::string& filename);
/////////////////////////////CSV文件保存与读写///////////////////////////////

/** @brief 保存单组数据到CSV文件
 * @param filename 文件名
 * @param data_list 单组数据
 * @param include_left 是否保存左眼数据
 */
void SaveToCSV(const std::string& filename, const std::vector<Data>& data_list,bool include_left=false);

/** @brief 从CSV文件读取数据
 * @param filename 文件名
 * @return 单组数据列表
 */
DataList LoadFromCSV(const std::string& filename);

/////////////////////////////Aruco二维码检测////////////////////////////////

/** @struct PoseFilter 
 * @brief 姿态滤波器结构体
 * @param prevR 上一帧旋转矩阵
 * @param prevT 上一帧平移向量
 * @param alpha  滤波系数，值越小越平滑
 * @param is_initialized 是否初始化
*/
struct PoseFilter {
    cv::Mat prevR;    // 上一帧旋转矩阵
    cv::Mat prevT;    // 上一帧平移向量
    double alpha = 0.2; // 滤波系数(0-1), 值越小越平滑
    bool is_initialized = false;
};

/** @brief 标定aruco标定板
 * @param img 场景相机图像
 * @param K 场景相机内参矩阵
 * @param d 场景相机畸变参数
 * @param R 标定板到场景的旋转矩阵
 * @param T 标定板到场景的平移向量
 */
void GeometryDetectAruco(cv::Mat& img,const cv::Mat& K,const cv::Mat& d,cv::Mat& R,cv::Mat& T);
/** @brief 在画面上绘制坐标系辅助线，验证主点位置
 * @param frame 相机帧
 * @param cameraMatrix 相机内参矩阵
 */
void DrawCameraAxes(cv::Mat &frame,const cv::Mat &cameraMatrix);

///////////////////////////相机重连///////////////////////////////
/** @brief 重新初始化相机
 * @param scene_cam 场景相机对象
 * @param scene_id 场景相机ID
 * @param eye_cam 右眼眼部相机对象
 * @param eye_id 右眼眼部相机ID
 * @param eye_cam_left 左眼眼部相机对象
 * @param eye_id_left 左眼眼部相机ID
 */
void ReinitCameras(cv::VideoCapture& scene_cam, int scene_id,
                cv::VideoCapture& eye_cam, int eye_id,
                 cv::VideoCapture& eye_cam_left, int eye_left_id);
/** @brief 重新初始化相机
 * @param scene_cam 场景相机对象
 * @param scene_id 场景相机ID
 * @param eye_cam 眼部相机对象
 * @param eye_id 眼部相机ID
 */
void ReinitCameras(cv::VideoCapture& scene_cam, int scene_id,
                cv::VideoCapture& eye_cam, int eye_id);

//////////////////////////视线估计//////////////////////////////
/** @brief 单目凝视点估计，并绘制凝视点位置
 * @details 计算视线与标定板平面的交点，并投影到成像平面中进行显示
 * @param frame 场景相机图像
 * @param R_board_to_cam 标定板坐标系到相机坐标系的旋转矩阵
 * @param t_board_to_cam 标定板坐标系到相机坐标系的平移向量
 * @param line_in_cam  相机坐标系下的视线向量
 * @param cameraMatrix 场景相机内参矩阵
 * @param distCoeffs 场景相机畸变系数
 */
Geometry_Point EstimateGazePoint(
    const cv::Mat& frame, //场景相机图像
    const cv::Mat& R_board_to_cam, //场景相机到标定板相机的旋转矩阵
    const cv::Mat& t_board_to_cam,  //场景相机到标定板相机的平移向量
    const Geometry_Line& line_in_cam,   //在场景相机坐标系下的视轴
    const cv::Mat& cameraMatrix,    //场景相机内参矩阵
    const cv::Mat& distCoeffs); //场景相机畸变参数

/** @brief 双目凝视点估计，并绘制出凝视点位置
 * @details 计算左右视线的最近点坐标，并投影到成像平面中进行显示
 * @param frame 场景相机图像
 * @param line_in_cam_left 左眼视线
 * @param line_in_cam_right 右眼视线
 * @param cameraMatrix 场景相机内参矩阵
 * @param distCoeffs 场景相机畸变系数
 */
Geometry_Point EstimateGazePoint(
    const cv::Mat& frame, 
    const Geometry_Line& line_in_cam_left, //左眼视线
    const Geometry_Line& line_in_cam_right, //右眼视线
    const cv::Mat& cameraMatrix, //场景相机内参矩阵
    const cv::Mat& distCoeffs); //场景相机畸变系数




#endif // GEOMETRY_H