#pragma once

#include <opencv2/core/cuda.hpp>
#include <cudnn.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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
void generate_multiscale_feature_maps(cudnnHandle_t cudnn_handle, const std::vector<float*> &extra_conv_weights, const std::vector<float*> &extra_conv_biases, const std::vector<float*>t &vgg16_output, std::vector<cv::cuda::GpuMat> &feature_maps);
void apply_softmax(cudnnHandle_t cudnn_handle, cv::cuda::GpuMat &data);
void predict_classes_and_bboxes(cudnnHandle_t cudnn_handle, const std::vector<cv::cuda::GpuMat> &feature_maps, const std::vector<float*> &class_weights,  const std::vector<float*> &class_biases,  const std::vector<float*> &box_weights, const std::vector<std::vector<float>> &box_biases, std::vector<cv::cuda::GpuMat> &class_scores, std::vector<cv::cuda::GpuMat> &box_deltas);

// 解码预测结果
void decode_predictions(const cv::cuda::GpuMat &class_predictions, const cv::cuda::GpuMat &bbox_predictions, std::vector<cv::Rect> &decoded_bboxes, std::vector<int> &decoded_class_ids, std::vector<float> &decoded_scores);

// 非极大值抑制
std::vector<BoundingBox> non_max_suppression(const std::vector<cv::Rect2f>& decoded_boxes, const std::vector<int>& decoded_labels, const std::vector<float>& decoded_scores, float threshold, int top_k);

void ssd_detect(cudnnHandle_t cudnn_handle, const std::vector<float*> &weights,const std::vector<float*> &biases const cv::cuda::GpuMat &input_image, std::vector<cv::Rect> &final_bboxes, std::vector<int> &final_class_ids, std::vector<float> &final_scores, float confidence_threshold, float iou_threshold); 
