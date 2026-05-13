/**
 * @file ellseg_feature.cpp
 * @brief ELLSeg眼部特征提取模块实现
 * @author LiuXin
 * @date 2025-03-10
 */
#include "ellseg_feature.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

#ifdef PUPILGAZE_WITH_LIBTORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace {

//角度转换常量，用于OpenCV椭圆角度和弧度之间转换
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRadToDeg = 180.0f / kPi;
constexpr float kDegToRad = kPi / 180.0f;

/**
 * @brief 获取不同ELLSeg语义标签对应的可视化颜色
 * @param label ELLSeg分割标签
 * @return OpenCV BGR颜色
 */
cv::Scalar LabelColor(EllSegLabel label) {
    switch (label) {
    case EllSegLabel::Pupil:
        return cv::Scalar(0, 255, 255);
    case EllSegLabel::Iris:
        return cv::Scalar(0, 180, 0);
    default:
        return cv::Scalar(0, 0, 0);
    }
}

/**
 * @brief 判断OpenCV拟合出的椭圆是否可用
 * @param rect OpenCV椭圆矩形
 * @return 有限且长短轴为正时返回true
 */
bool IsFiniteEllipse(const cv::RotatedRect& rect) {
    return std::isfinite(rect.center.x) &&
           std::isfinite(rect.center.y) &&
           std::isfinite(rect.size.width) &&
           std::isfinite(rect.size.height) &&
           rect.size.width > 0.0f &&
           rect.size.height > 0.0f;
}

#ifdef PUPILGAZE_WITH_LIBTORCH
/**
 * @brief 将LibTorch Tensor转换为float数组
 * @param tensor 待转换的Tensor
 * @return 一维float数组
 */
std::vector<float> TensorToVector(const at::Tensor& tensor) {
    if (!tensor.defined()) {
        return {};
    }

    at::Tensor flat = tensor.detach()
                          .to(torch::kCPU)
                          .to(torch::kFloat)
                          .contiguous()
                          .view({-1});
    std::vector<float> values(static_cast<size_t>(flat.numel()));
    if (!values.empty()) {
        std::memcpy(values.data(),
                    flat.data_ptr<float>(),
                    values.size() * sizeof(float));
    }
    return values;
}

/**
 * @brief 将ELLSeg分割logits转换为标签图Tensor
 * @param seg_logits 模型输出的分割logits
 * @return 单通道标签图Tensor
 */
at::Tensor LabelTensorFromSegmentationLogits(const at::Tensor& seg_logits) {
    if (seg_logits.dim() == 4) {
        return seg_logits.argmax(1).squeeze(0);
    }
    if (seg_logits.dim() == 3 && seg_logits.size(0) > 1) {
        return seg_logits.argmax(0);
    }
    return seg_logits.squeeze();
}
#endif

} // namespace

//将内部椭圆参数转换为OpenCV RotatedRect，便于绘制和后续几何计算
cv::RotatedRect EllSegEllipse::ToRotatedRect() const {
    if (!valid) {
        return cv::RotatedRect();
    }

    return cv::RotatedRect(center,
                           cv::Size2f(axes.width * 2.0f, axes.height * 2.0f),
                           angle_radians * kRadToDeg);
}

#ifdef PUPILGAZE_WITH_LIBTORCH
//LibTorch状态，仅在工程启用PUPILGAZE_WITH_LIBTORCH时编译
class EllSegFeatureExtractor::TorchState {
public:
    torch::jit::script::Module module;                 // TorchScript模型
    torch::Device device = torch::Device(torch::kCPU); // 推理设备
};
#endif

//构造函数，保存配置并在启用LibTorch时初始化Torch状态对象
EllSegFeatureExtractor::EllSegFeatureExtractor(const EllSegExtractorOptions& options)
    : options_(options) {
#ifdef PUPILGAZE_WITH_LIBTORCH
    torch_state_ = std::make_shared<TorchState>();
#endif
}

/**
 * @brief 加载ELLSeg导出的TorchScript模型
 * @param model_path TorchScript模型路径
 * @return 加载成功返回true，否则返回false并记录错误信息
 */
bool EllSegFeatureExtractor::Load(const std::string& model_path) {
    loaded_ = false;
    last_error_.clear();

#ifndef PUPILGAZE_WITH_LIBTORCH
    last_error_ = "LibTorch is not enabled. Configure CMake with Torch_DIR to enable EllSeg inference.";
    (void)model_path;
    return false;
#else
    if (!std::filesystem::exists(model_path)) {
        last_error_ = "EllSeg TorchScript model does not exist: " + model_path;
        return false;
    }

    try {
        torch_state_->device = torch::Device(torch::kCPU);
        if (options_.prefer_cuda && torch::cuda::is_available()) {
            torch_state_->device = torch::Device(torch::kCUDA);
        }

        torch_state_->module = torch::jit::load(model_path, torch_state_->device);
        torch_state_->module.to(torch_state_->device);
        torch_state_->module.eval();
        loaded_ = true;
        return true;
    } catch (const c10::Error& error) {
        last_error_ = error.what();
    } catch (const std::exception& error) {
        last_error_ = error.what();
    }

    return false;
#endif
}

//返回模型是否已经成功加载
bool EllSegFeatureExtractor::IsLoaded() const {
    return loaded_;
}

//返回最近一次错误信息
const std::string& EllSegFeatureExtractor::LastError() const {
    return last_error_;
}

/**
 * @brief 对单帧眼图像执行ELLSeg推理
 * @param eye_frame 输入眼部图像，可为灰度图或BGR/BGRA图
 * @param feature 输出的眼部特征
 * @return 特征提取成功返回true
 */
bool EllSegFeatureExtractor::Extract(const cv::Mat& eye_frame, EllSegFeature& feature) const {
    feature = EllSegFeature();
    last_error_.clear();

    if (!loaded_) {
        last_error_ = "EllSeg model is not loaded.";
        return false;
    }

#ifndef PUPILGAZE_WITH_LIBTORCH
    last_error_ = "LibTorch is not enabled. Configure CMake with Torch_DIR to enable EllSeg inference.";
    (void)eye_frame;
    return false;
#else
    cv::Mat normalized_frame;
    PreprocessMeta meta;
    //按照ELLSeg Python推理脚本的尺寸对齐和归一化方式预处理图像
    if (!PreprocessFrame(eye_frame, normalized_frame, meta)) {
        return false;
    }

    try {
        torch::NoGradGuard no_grad;
        at::Tensor input_tensor = torch::from_blob(
                                      normalized_frame.ptr<float>(),
                                      {1, 1, normalized_frame.rows, normalized_frame.cols},
                                      torch::TensorOptions().dtype(torch::kFloat))
                                      .clone()
                                      .to(torch_state_->device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);
        torch::jit::IValue raw_output = torch_state_->module.forward(inputs);

        at::Tensor seg_logits;
        at::Tensor latent_feature;
        at::Tensor ellipse_regression;

        //导出脚本中的包装模型返回(seg_out, latent, el_out)
        if (raw_output.isTuple()) {
            const auto tuple = raw_output.toTuple();
            const auto& elements = tuple->elements();
            if (elements.size() < 3) {
                last_error_ = "EllSeg model output tuple must contain seg_out, latent, and el_out.";
                return false;
            }
            seg_logits = elements[0].toTensor();
            latent_feature = elements[1].toTensor();
            ellipse_regression = elements[2].toTensor();
        } else if (raw_output.isTensor()) {
            seg_logits = raw_output.toTensor();
        } else {
            last_error_ = "Unsupported EllSeg model output type.";
            return false;
        }

        at::Tensor label_tensor = LabelTensorFromSegmentationLogits(seg_logits)
                                      .to(torch::kCPU)
                                      .to(torch::kByte)
                                      .contiguous();
        if (label_tensor.dim() != 2) {
            last_error_ = "EllSeg segmentation output is not a 2D label map.";
            return false;
        }

        cv::Mat model_segmentation(static_cast<int>(label_tensor.size(0)),
                                   static_cast<int>(label_tensor.size(1)),
                                   CV_8UC1,
                                   label_tensor.data_ptr<unsigned char>());

        //将模型输出整理成项目侧可直接使用的结构体
        feature.segmentation = RestoreSegmentation(model_segmentation.clone(), meta);
        feature.latent_feature = TensorToVector(latent_feature);
        feature.ellipse_regression = TensorToVector(ellipse_regression);
        feature.pupil = FitEllipseFromLabel(feature.segmentation, EllSegLabel::Pupil);
        feature.iris = FitEllipseFromLabel(feature.segmentation, EllSegLabel::Iris);
        feature.overlay = DrawOverlay(eye_frame, feature);
        feature.valid = !feature.segmentation.empty();
        return feature.valid;
    } catch (const c10::Error& error) {
        last_error_ = error.what();
    } catch (const cv::Exception& error) {
        last_error_ = error.what();
    } catch (const std::exception& error) {
        last_error_ = error.what();
    }

    return false;
#endif
}

/**
 * @brief 生成ELLSeg分割和椭圆的叠加显示图
 * @param eye_frame 原始眼部图像
 * @param feature ELLSeg提取结果
 * @param alpha 分割颜色叠加权重
 * @return BGR可视化图像
 */
cv::Mat EllSegFeatureExtractor::DrawOverlay(const cv::Mat& eye_frame,
                                            const EllSegFeature& feature,
                                            double alpha) const {
    cv::Mat base;
    if (eye_frame.empty()) {
        return base;
    }

    //将灰度或BGRA图统一转换为BGR，方便OpenCV绘制彩色叠加结果
    if (eye_frame.channels() == 1) {
        cv::cvtColor(eye_frame, base, cv::COLOR_GRAY2BGR);
    } else if (eye_frame.channels() == 4) {
        cv::cvtColor(eye_frame, base, cv::COLOR_BGRA2BGR);
    } else {
        base = eye_frame.clone();
    }

    if (feature.segmentation.empty() || feature.segmentation.size() != base.size()) {
        return base;
    }

    //根据分割标签生成彩色mask，再与原图叠加
    cv::Mat color_mask(base.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    color_mask.setTo(LabelColor(EllSegLabel::Iris),
                     feature.segmentation == static_cast<unsigned char>(EllSegLabel::Iris));
    color_mask.setTo(LabelColor(EllSegLabel::Pupil),
                     feature.segmentation == static_cast<unsigned char>(EllSegLabel::Pupil));

    cv::Mat overlay;
    cv::addWeighted(color_mask, alpha, base, 1.0, 0.0, overlay);

    //在叠加图上绘制拟合后的虹膜和瞳孔椭圆
    if (feature.iris.valid) {
        cv::ellipse(overlay, feature.iris.ToRotatedRect(), LabelColor(EllSegLabel::Iris), 2);
    }
    if (feature.pupil.valid) {
        cv::ellipse(overlay, feature.pupil.ToRotatedRect(), LabelColor(EllSegLabel::Pupil), 2);
    }

    return overlay;
}

/**
 * @brief 对眼图进行ELLSeg模型输入预处理
 * @details 先转灰度，再按模型输入宽度缩放，并在高度方向居中补边或裁剪，最后做均值方差归一化
 * @param eye_frame 输入眼部图像
 * @param normalized_frame 输出的CV_32F归一化图像
 * @param meta 输出的尺寸变换信息
 * @return 预处理成功返回true
 */
bool EllSegFeatureExtractor::PreprocessFrame(const cv::Mat& eye_frame,
                                             cv::Mat& normalized_frame,
                                             PreprocessMeta& meta) const {
    if (eye_frame.empty()) {
        last_error_ = "Empty eye frame.";
        return false;
    }
    if (options_.model_input_size.width <= 0 || options_.model_input_size.height <= 0) {
        last_error_ = "Invalid EllSeg model input size.";
        return false;
    }

    //统一转换为灰度图，匹配ELLSeg导出模型的单通道输入
    cv::Mat gray;
    if (eye_frame.channels() == 1) {
        gray = eye_frame;
    } else if (eye_frame.channels() == 3) {
        cv::cvtColor(eye_frame, gray, cv::COLOR_BGR2GRAY);
    } else if (eye_frame.channels() == 4) {
        cv::cvtColor(eye_frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        last_error_ = "Unsupported eye frame channel count.";
        return false;
    }

    //按照宽度对齐到模型输入尺寸，保持原始宽高比例
    meta.original_size = gray.size();
    meta.scale = static_cast<double>(options_.model_input_size.width) /
                 static_cast<double>(gray.cols);

    const int resized_height = std::max(1, cvRound(gray.rows * meta.scale));
    meta.resized_size = cv::Size(options_.model_input_size.width, resized_height);

    cv::Mat resized;
    cv::resize(gray, resized, meta.resized_size, 0.0, 0.0, cv::INTER_LANCZOS4);

    //高度不足时居中补边，高度过大时居中裁剪
    meta.vertical_delta = options_.model_input_size.height - resized.rows;

    cv::Mat aligned;
    if (meta.vertical_delta > 0) {
        const int top = meta.vertical_delta / 2;
        const int bottom = meta.vertical_delta - top;
        cv::copyMakeBorder(resized,
                           aligned,
                           top,
                           bottom,
                           0,
                           0,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0));
    } else if (meta.vertical_delta < 0) {
        const int crop_top = std::min((-meta.vertical_delta) / 2,
                                      std::max(0, resized.rows - options_.model_input_size.height));
        aligned = resized(cv::Rect(0,
                                   crop_top,
                                   options_.model_input_size.width,
                                   options_.model_input_size.height))
                      .clone();
    } else {
        aligned = resized;
    }

    if (aligned.size() != options_.model_input_size) {
        cv::Mat aligned_resized;
        cv::resize(aligned, aligned_resized, options_.model_input_size, 0.0, 0.0, cv::INTER_LANCZOS4);
        aligned = aligned_resized;
    }

    //执行(x - mean) / std归一化，和ELLSeg Python脚本保持一致
    aligned.convertTo(normalized_frame, CV_32F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(normalized_frame, mean, stddev);
    const double denom = std::max(stddev[0], 1e-6);
    normalized_frame = (normalized_frame - mean[0]) / denom;
    return true;
}

/**
 * @brief 将ELLSeg模型输出的分割图还原到原始眼图尺寸
 * @param model_segmentation 模型输入尺寸下的分割标签图
 * @param meta 预处理阶段保存的尺寸变换信息
 * @return 原始眼图尺寸下的分割标签图
 */
cv::Mat EllSegFeatureExtractor::RestoreSegmentation(const cv::Mat& model_segmentation,
                                                    const PreprocessMeta& meta) const {
    if (model_segmentation.empty()) {
        return cv::Mat();
    }

    //根据预处理时的补边或裁剪操作做反向恢复
    cv::Mat restored;
    if (meta.vertical_delta > 0) {
        const int top = meta.vertical_delta / 2;
        const int height = std::max(1, model_segmentation.rows - meta.vertical_delta);
        restored = model_segmentation(cv::Rect(0, top, model_segmentation.cols, height)).clone();
    } else if (meta.vertical_delta < 0) {
        const int missing_height = -meta.vertical_delta;
        const int top = missing_height / 2;
        const int bottom = missing_height - top;
        cv::copyMakeBorder(model_segmentation,
                           restored,
                           top,
                           bottom,
                           0,
                           0,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(static_cast<unsigned char>(EllSegLabel::Background)));
    } else {
        restored = model_segmentation.clone();
    }

    cv::Mat original_size_segmentation;
    cv::resize(restored,
               original_size_segmentation,
               meta.original_size,
               0.0,
               0.0,
               cv::INTER_NEAREST);
    return original_size_segmentation;
}

/**
 * @brief 根据分割标签区域拟合二维椭圆
 * @param segmentation 原始眼图尺寸下的分割标签图
 * @param label 需要拟合的语义标签
 * @return 拟合出的椭圆参数，点数不足或拟合失败时valid为false
 */
EllSegEllipse EllSegFeatureExtractor::FitEllipseFromLabel(const cv::Mat& segmentation,
                                                          EllSegLabel label) const {
    EllSegEllipse ellipse;
    if (segmentation.empty()) {
        return ellipse;
    }

    cv::Mat mask = segmentation == static_cast<unsigned char>(label);
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    //点数过少时不进行椭圆拟合，避免产生不稳定结果
    if (static_cast<int>(points.size()) < options_.min_points_per_ellipse) {
        return ellipse;
    }

    try {
        cv::RotatedRect rect = cv::fitEllipse(points);
        if (!IsFiniteEllipse(rect)) {
            return ellipse;
        }

        ellipse.valid = true;
        ellipse.center = rect.center;
        ellipse.axes = cv::Size2f(rect.size.width * 0.5f, rect.size.height * 0.5f);
        ellipse.angle_radians = rect.angle * kDegToRad;
    } catch (const cv::Exception&) {
        return EllSegEllipse();
    }

    return ellipse;
}
