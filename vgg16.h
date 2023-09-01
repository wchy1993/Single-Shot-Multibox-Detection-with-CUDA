#ifndef VGG16_H
#define VGG16_H

#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <vector>

void perform_convolution(
    cudnnHandle_t& cudnn_handle,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int input_channels,
    int output_channels,
    int kernel_size,
    int stride,
    cudnnTensorDescriptor_t& input_descriptor,
    cudnnTensorDescriptor_t& output_descriptor,
    float*& d_input_data,
    float*& d_output_data
);

void perform_pooling(
    cudnnHandle_t &cudnn_handle,
    cudnnTensorDescriptor_t &input_descriptor, cudnnTensorDescriptor_t &output_descriptor,
    float *d_input_data, float *&d_output_data);

void extract_features(
    cudnnHandle_t &cudnn_handle, const std::vector<float> &conv_weights, const std::vector<float> &conv_biases, const cv::cuda::GpuMat &input_image, cv::cuda::GpuMat &output_features);

#endif
