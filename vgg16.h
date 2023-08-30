#pragma once


#include <cudnn.h>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include "image_preprocessing.h"

void perform_convolution(cudnnHandle_t cudnn_handle, int kernel_height, int kernel_width, int pad_height, int pad_width, cudnnTensorDescriptor_t input_descriptor, float* d_input_data, const std::vector<float> &weights, const std::vector<float> &biases, cudnnTensorDescriptor_t output_descriptor, float* d_output_data);
void perform_pooling(cudnnHandle_t cudnn_handle, int window_height, int window_width, int pooling_stride, cudnnTensorDescriptor_t input_descriptor, float* d_input_data, cudnnTensorDescriptor_t output_descriptor, float* d_output_data);
void extract_features(cudnnHandle_t cudnn_handle, const std::vector<std::vector<float>> &conv_weights, const std::vector<std::vector<float>> &conv_biases, const cv::cuda::GpuMat &input_image, cv::cuda::GpuMat &output_features);
