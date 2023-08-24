#pragma once

#include <opencv2/core/cuda.hpp>
#include <cudnn.h>
#include <vector>

struct BoundingBox {
  float x1, y1, x2, y2, score;
  int label;
};

// 比较函数，用于排序
struct BBoxCompare {
  __host__ __device__
  bool operator()(const BoundingBox& a, const BoundingBox& b) const {
    return a.score > b.score;
  }
};

// 生成多尺度特征图
void generate_multiscale_feature_maps(cudnnHandle_t cudnn_handle, const std::vector<float> &weights, const cv::cuda::GpuMat &input_features, std::vector<cv::cuda::GpuMat> &output_feature_maps);

// 类别预测和边界框回归
void predict_classes_and_bboxes(cudnnHandle_t cudnn_handle, const std::vector<float> &weights, const std::vector<cv::cuda::GpuMat> &feature_maps, cv::cuda::GpuMat &class_predictions, cv::cuda::GpuMat &bbox_predictions);

// 解码预测结果
void decode_predictions(const cv::cuda::GpuMat &class_predictions, const cv::cuda::GpuMat &bbox_predictions, std::vector<cv::Rect> &decoded_bboxes, std::vector<int> &decoded_class_ids, std::vector<float> &decoded_scores);

// 置信度阈值过滤
void filter_predictions_by_confidence(const std::vector<cv::Rect> &decoded_bboxes, const std::vector<int> &decoded_class_ids, const std::vector<float> &decoded_scores, float confidence_threshold, std::vector<cv::Rect> &filtered_bboxes, std::vector<int> &filtered_class_ids, std::vector<float> &filtered_scores);

// 非极大值抑制
non_max_suppression(const std::vector<BoundingBox>& bboxes, float threshold, int top_k);
// SSD模型的完整检测过程
void ssd_detect(cudnnHandle_t cudnn_handle, const std::vector<float> &weights, const cv::cuda::GpuMat &input_image, std::vector<cv::Rect> &final_bboxes, std::vector<int> &final_class_ids, std::vector<float> &final_scores, float confidence_threshold = 0.5, float iou_threshold = 0.5);
