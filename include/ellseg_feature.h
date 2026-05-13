/**
 * @file ellseg_feature.h
 * @brief ELLSeg眼部特征提取模块定义
 * @details 通过ELLSeg导出的TorchScript/LibTorch模型提取眼部分割、latent特征和瞳孔/虹膜椭圆信息
 * @author LiuXin
 * @date 2025-03-10
 */
#ifndef ELLSEG_FEATURE_H
#define ELLSEG_FEATURE_H

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

/** @enum EllSegLabel
 * @brief ELLSeg分割结果中的语义标签
 */
enum class EllSegLabel : unsigned char {
    Background = 0,    // 背景区域
    Iris = 1,          // 虹膜区域
    Pupil = 2          // 瞳孔区域
};

/** @struct EllSegEllipse
 * @brief ELLSeg提取出的二维椭圆参数
 * @param valid          椭圆参数是否有效
 * @param center         椭圆中心点，单位为原始眼图像素
 * @param axes           椭圆半长轴和半短轴，单位为像素
 * @param angle_radians  椭圆旋转角，单位为弧度
 */
struct EllSegEllipse {
    bool valid = false;
    cv::Point2f center = cv::Point2f(-1.0f, -1.0f);
    cv::Size2f axes = cv::Size2f(-1.0f, -1.0f);
    float angle_radians = 0.0f;

    /** @brief 转换为OpenCV绘图和几何计算常用的RotatedRect格式 */
    cv::RotatedRect ToRotatedRect() const;
};

/** @struct EllSegFeature
 * @brief 单帧眼图像的ELLSeg特征结果
 * @param valid               特征提取是否成功
 * @param segmentation        恢复到原始图像尺寸的分割标签图
 * @param overlay             分割和椭圆叠加后的可视化图像
 * @param pupil               瞳孔椭圆参数
 * @param iris                虹膜椭圆参数
 * @param latent_feature      ELLSeg编码器输出的轻量化latent特征
 * @param ellipse_regression  ELLSeg椭圆回归头输出的原始参数
 */
struct EllSegFeature {
    bool valid = false;
    cv::Mat segmentation;
    cv::Mat overlay;
    EllSegEllipse pupil;
    EllSegEllipse iris;
    std::vector<float> latent_feature;
    std::vector<float> ellipse_regression;
};

/** @struct EllSegExtractorOptions
 * @brief ELLSeg特征提取器配置
 * @param model_input_size        TorchScript模型输入尺寸，默认宽320高240
 * @param prefer_cuda             是否优先使用CUDA推理
 * @param min_points_per_ellipse  拟合椭圆所需的最少分割点数量
 */
struct EllSegExtractorOptions {
    cv::Size model_input_size = cv::Size(320, 240);
    bool prefer_cuda = false;
    int min_points_per_ellipse = 50;
};

/** @class EllSegFeatureExtractor
 * @brief 使用ELLSeg导出的LibTorch模型进行眼部特征提取
 * @details 当工程未启用LibTorch时，该类保留同名接口但不执行实际推理，便于项目先完成接口接入
 */
class EllSegFeatureExtractor {
public:
    /** @brief 构造函数，传入ELLSeg提取器配置 */
    explicit EllSegFeatureExtractor(const EllSegExtractorOptions& options = EllSegExtractorOptions());

    /** @brief 加载ELLSeg导出的TorchScript模型 */
    bool Load(const std::string& model_path);
    /** @brief 判断模型是否已经成功加载 */
    bool IsLoaded() const;
    /** @brief 获取最近一次加载或推理失败的错误信息 */
    const std::string& LastError() const;

    /** @brief 对单帧眼图像执行ELLSeg特征提取 */
    bool Extract(const cv::Mat& eye_frame, EllSegFeature& feature) const;
    /** @brief 绘制ELLSeg分割图和瞳孔/虹膜椭圆叠加结果 */
    cv::Mat DrawOverlay(const cv::Mat& eye_frame,
                        const EllSegFeature& feature,
                        double alpha = 0.35) const;

private:
    /** @struct PreprocessMeta
     * @brief 预处理阶段记录的尺寸变换信息，用于将分割结果恢复到原图尺寸
     */
    struct PreprocessMeta {
        cv::Size original_size;    // 原始眼图尺寸
        cv::Size resized_size;     // 按宽度缩放后的尺寸
        double scale = 1.0;        // 从原始宽度到模型输入宽度的缩放系数
        int vertical_delta = 0;    // 高度方向补边或裁剪的像素差
    };

    /** @brief 将输入眼图转换为ELLSeg模型需要的归一化灰度图 */
    bool PreprocessFrame(const cv::Mat& eye_frame,
                         cv::Mat& normalized_frame,
                         PreprocessMeta& meta) const;
    /** @brief 将模型输出的分割图恢复到原始眼图尺寸 */
    cv::Mat RestoreSegmentation(const cv::Mat& model_segmentation,
                                const PreprocessMeta& meta) const;
    /** @brief 根据指定标签区域拟合瞳孔或虹膜椭圆 */
    EllSegEllipse FitEllipseFromLabel(const cv::Mat& segmentation,
                                      EllSegLabel label) const;

    EllSegExtractorOptions options_;       // 提取器配置
    bool loaded_ = false;                  // 模型是否加载成功
    mutable std::string last_error_;       // 最近一次错误信息

#ifdef PUPILGAZE_WITH_LIBTORCH
    class TorchState;
    std::shared_ptr<TorchState> torch_state_; // LibTorch模型和设备上下文
#endif
};

#endif
