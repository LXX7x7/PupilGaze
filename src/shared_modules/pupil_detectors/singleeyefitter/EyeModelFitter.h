#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <opencv2/video/tracking.hpp> // Kalaman Filter

#include <vector>
#include <memory>
#include <Eigen/Core>

#include "common/types.h"
#include "ImageProcessing/cvx.h"
#include "geometry/Circle.h"
#include "geometry/Ellipse.h"
#include "geometry/Sphere.h"
#include "EyeModel.h"

//#include "logger/pycpplogger.h"


namespace singleeyefitter {


    class EyeModelFitter {
        public:

            typedef singleeyefitter::Sphere<double> Sphere;
            typedef std::unique_ptr<EyeModel> EyeModelPtr;

            // Constructors
            EyeModelFitter(double focalLength, Vector3 cameraCenter = Vector3::Zero() );
            EyeModelFitter() = delete;

			int m_cx;
			int m_cy;



			//*******************************************************
			//功    能： 【获得相机的焦距】
			//            
			//输入参数： 
			//
			//返 回 值：
			//     返回焦距
			//修改日志：
			//*******************************************************
            double getFocalLength(){ return mFocalLength; };
			//*******************************************************
			//功    能： 【重置模型】
			//            
			//输入参数： 
			//
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
            void reset();

            // this is called with new observations from the 2D detector
            // it decides what happens ,since not all observations are added
			//*******************************************************
			//功    能： 【由2D瞳孔信息获得眼球模型和视线向量】
			//            根据获得的瞳孔信息重建眼球模型，获得视线向量
			//输入参数： 
			//    [in]     observation                     2D瞳孔检测的所有信息
			//    [in]     props                           3D瞳孔还原中眼球模型 performance的最小值
			//    [in]     debug                           输出debug信息
			//返 回 值：
			//     3D瞳孔检测的所有信息
			//修改日志：
			//*******************************************************
            Detector3DResult updateAndDetect( std::shared_ptr<Detector2DResult>& observation,const Detector3DProperties& props, bool debug = false );


        private:

            const Vector3 mCameraCenter;                                                           //  相机光心 ？
            const double mFocalLength;                                                             //  焦距长度？

            bool mDebug;

            Clock::time_point mLastTimeModelAdded, mLastTimePerformancePenalty;                    // 最新模型产生的时间，2、模型的表现不OK或模型的表现下滑幅度过大，则记录时间   

            int mNextModelID;                                                                      // 新模型的ID号
            std::unique_ptr<EyeModel> mActiveModelPtr;                                             // 当前最优的眼球模型
            std::list<EyeModelPtr> mAlternativeModelsPtrs;                                         // 目前存在的眼球模型

            Sphere mCurrentSphere;                                                                 // 当前已经优化的模型                     
            Sphere mCurrentInitialSphere;                                                          // 当前未优化的模型

            cv::KalmanFilter mPupilState;                                                          // Kalman 滤波模型，主要用来预测眼球瞳孔的下一个位置

            double mLastFrameTimestamp; //needed to calculate framerate                            // 上一幅图像来的时间，此成员变量用来计算帧率
            int mApproximatedFramerate;                                                            // 当前帧率的近似值（两幅图像）
            math::SMA<double> mAverageFramerate;                                                   // 一段时间内的帧率均值

           // pupillabs::PyCppLogger mLogger;                                                        // 已经被干掉





	    public:
			//*******************************************************
			//功    能： 【对眼球的模型进行操作，选择或产生或预备最好的眼球模型，主要的操作成员变量是mActiveModelPtr、mAlternativeModelsPtrs】
			//            
			//输入参数： 
			//    [in]     sensitivity                               眼球模型的置信度最小值
			//    [in]     frame_timestamp                           标记眼球模型的出生时间
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
            void checkModels( float sensitivity,double frame_timestamp);

            //Contours3D unprojectContours( const Contours_2D& contours) const;
			//*******************************************************
			//功    能： 【2D瞳孔边界点 反向投影到眼球（当前的球）上的3D点】
			//            
			//输入参数： 
			//    [in]     edges                                    2D瞳孔中的边界点
			//返 回 值：
			//     返回3D瞳孔在球上的边界点位置
			//修改日志：
			//*******************************************************
            Edges3D unprojectEdges(const Edges2D& edges) const;

            // whenever the 2D fit is bad we wanna call this and predict an new circle to use for findCircle
			//*******************************************************
			//功    能： 【当用眼球模型检测出的3D瞳孔置信度较低，则是用Kalman Filter 预测瞳孔的先验估计】
			//            
			//输入参数： 
			//    [in]     deltaTime                               在转移矩阵中使用，因为瞳孔的状态向量中包含有vx，xy，ax，ay速度，加速，用时间来更好预测
			//返 回 值：
			//     返回3D瞳孔的先验估计
			//修改日志：
			//*******************************************************
            Circle predictPupilState( double deltaTime );

			//*******************************************************
			//功    能： 【矫正3D瞳孔的先验估计，获得预测的真值】
			//            
			//输入参数： 
			//    [in]     circle                                 测量值，即当前时刻3D瞳孔的观测值；
			//返 回 值：
			//     返回3D瞳孔的矫正值
			//修改日志：
			//*******************************************************
            Circle correctPupilState( const Circle& circle );

			//*******************************************************
			//功    能： 【3D瞳孔角度（球坐标）预测的偏差】
			//            
			//输入参数： 
			//    
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
            double getPupilPositionErrorVar () const;

			//*******************************************************
			//功    能： 【3D瞳孔半径预测的偏差】
			//            
			//输入参数： 
			//    
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
            double getPupilSizeErrorVar() const;

			//*******************************************************
			//功    能： 【获得当前使用眼球的模型】
			//            
			//输入参数： 
			//    
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
			singleeyefitter::Sphere<double> GetCurrentSphere() const;



            //void fitCircle(const Contours_2D& contours2D , const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle(const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;
            void filterCircle2( const Circle& predictedCircle, const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;

			//*******************************************************
			//功    能： 【根据预测的3D瞳孔位置，使用2D瞳孔边缘点优化更合理的瞳孔位置】
			// 
			//输入参数： 
			//    [in]        predictedCircle                   Kalman滤波预测的3D瞳孔位置
			//    [in]        rawEdge                           2D图像中瞳孔的原始边界
			//    [in]        props                             3D检测的的属性，置信度，此参数目前在函数内部没有使用；
			//    [in/out]    result                            主要是跟新3D瞳孔圆的参数     
			//返 回 值：
			//     
			//修改日志：
			//*******************************************************
            void filterCircle3( const Circle& predictedCircle, const Edges2D& rawEdge, const Detector3DProperties& props,  Detector3DResult& result) const;

    };

}

#endif // SingleEyeFitter_h__
