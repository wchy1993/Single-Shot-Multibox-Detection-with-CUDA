#include "vgg16.h"
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

class CudnnHandle {
public:
    CudnnHandle() {
        cudnnStatus_t status = cudnnCreate(&_handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuDNN handle");
        }
    }

    ~CudnnHandle() {
        if (_handle) {
            cudnnDestroy(_handle);
        }
    }

    cudnnHandle_t get() const {
        return _handle;
    }

private:
    cudnnHandle_t _handle = nullptr;
};

void perform_convolution(
    cudnnHandle_t& cudnn_handle,
    int input_channels,
    int output_channels,
    int kernel_size,
    int stride,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    cudnnTensorDescriptor_t& input_descriptor,
    cudnnTensorDescriptor_t& output_descriptor,
    float*& d_input_data,
    float*& d_output_data) 
{
    // 创建卷积层描述符、卷积权重描述符和卷积偏置描述符
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnTensorDescriptor_t bias_descriptor;

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnCreateTensorDescriptor(&bias_descriptor);

    int num_filters = output_channels;
    int filter_height = kernel_size;
    int filter_width = kernel_size;

    cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, num_filters, input_channels, filter_height, filter_width);
    cudnnSetTensor4dDescriptor(bias_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_filters, 1, 1);

    // 为卷积层的权重和偏置分配GPU内存
    float *d_filter_data;
    float *d_bias_data;
    cudaMalloc(&d_filter_data, num_filters * input_channels * filter_height * filter_width * sizeof(float));
    cudaMalloc(&d_bias_data, num_filters * sizeof(float));

    // 将权重数据复制到GPU内存
    cudaMemcpy(d_filter_data, weights.data(), num_filters * input_channels * filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_data, biases.data(), num_filters * sizeof(float), cudaMemcpyHostToDevice);

    // 计算输出尺寸
    int output_height, output_width;
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, filter_descriptor, &output_height, &output_width);

    // 设置输出张量描述符
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, num_filters, output_height, output_width);

    // 执行卷积操作
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaMalloc(&d_output_data, num_filters * output_height * output_width * sizeof(float));

    cudnnConvolutionForward(cudnn_handle, &alpha, input_descriptor, d_input_data, filter_descriptor, d_filter_data, convolution_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0, &beta, output_descriptor, d_output_data);

    // 添加偏置
    cudnnAddTensor(cudnn_handle, &alpha, bias_descriptor, d_bias_data, &alpha, output_descriptor, d_output_data);

    // 激活函数（ReLU）
    cudnnActivationDescriptor_t activation_descriptor;
    cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);

    cudnnActivationForward(cudnn_handle, activation_descriptor, &alpha, output_descriptor, d_output_data, &beta, output_descriptor, d_output_data);

    // 销毁创建的描述符和释放分配的内存
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyTensorDescriptor(bias_descriptor);
    cudnnDestroyActivationDescriptor(activation_descriptor);

    cudaFree(d_filter_data);
    cudaFree(d_bias_data);
}


void perform_pooling(
    cudnnHandle_t &cudnn_handle,
    int pool_height, int pool_width, int pool_stride,
    cudnnTensorDescriptor_t &input_descriptor, cudnnTensorDescriptor_t &output_descriptor,
    float *d_input_data, float *&d_output_data) {

    // 创建池化层描述符
    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnCreatePoolingDescriptor(&pooling_descriptor);

    // 设置池化层参数
    cudnnSetPooling2dDescriptor(pooling_descriptor, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, pool_height, pool_width, 0, 0, pool_stride, pool_stride);

    // 计算输出尺寸
    int n, c, output_height, output_width;
    cudnnGetPooling2dForwardOutputDim(pooling_descriptor, input_descriptor, &n, &c, &output_height, &output_width);

    // 设置输出张量描述符
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, output_height, output_width);

    // 执行池化操作
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaMalloc(&d_output_data, n * c * output_height * output_width * sizeof(float));
    cudnnPoolingForward(cudnn_handle, pooling_descriptor, &alpha, input_descriptor, d_input_data, &beta, output_descriptor, d_output_data);

    // 销毁创建的描述符和释放分配的内存
    cudnnDestroyPoolingDescriptor(pooling_descriptor);
}


void extract_features(
    cudnnHandle_t &cudnn_handle, const std::vector<float> &conv_weights, const std::vector<float> &conv_biases, const cv::cuda::GpuMat &input_image, cv::cuda::GpuMat &output_features) {
    
    // 数据预处理
    cv::cuda::GpuMat preprocessed_image;
    preprocess_image_batch(input_image, preprocessed_image);

    // 为输入和输出创建张量描述符
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);

    // 设置输入张量描述符
    int input_channels = 3;
    int input_height = preprocessed_image.rows; // 512
    int input_width = preprocessed_image.cols; // 512
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, input_channels, input_height, input_width);

    // 分配GPU内存并将输入数据复制到GPU内存
    float *d_input_data, *d_output_data;
    size_t input_data_size = input_channels * input_height * input_width * sizeof(float);
    cudaMalloc(&d_input_data, input_data_size);
    cudaMemcpy(d_input_data, preprocessed_image.ptr<float>(), input_data_size, cudaMemcpyHostToDevice);

    // 第一个卷积块（2个卷积层）
    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[0], conv_biases[0], output_descriptor, d_output_data);
    input_descriptor = output_descriptor; // 更新输入描述符为上一层的输出
    d_input_data = d_output_data; // 更新输入数据为上一层的输出

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[1], conv_biases[1], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_pooling(cudnn_handle, 2, 2, 2, input_descriptor, d_input_data, output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    
    // 第二个卷积块（2个卷积层）
    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[2], conv_biases[2], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[3], conv_biases[3], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_pooling(cudnn_handle, 2, 2, 2, input_descriptor, d_input_data, output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    // 第三个卷积块（3个卷积层）
    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[4], conv_biases[4], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[5], conv_biases[5], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[6], conv_biases[6], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_pooling(cudnn_handle, 2, 2, 2, input_descriptor, d_input_data, output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    // 第四个卷积块（3个卷积层）
    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[7], conv_biases[7], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[8], conv_biases[8], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[9], conv_biases[9], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_pooling(cudnn_handle, 2, 2, 2, input_descriptor, d_input_data, output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    // 第五个卷积块（3个卷积层）
    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[10], conv_biases[10], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[11], conv_biases[11], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_convolution(cudnn_handle, 3, 3, 1, 1, input_descriptor, d_input_data, conv_weights[12], conv_biases[12], output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;

    perform_pooling(cudnn_handle, 2, 2, 2, input_descriptor, d_input_data, output_descriptor, d_output_data);
    input_descriptor = output_descriptor;
    d_input_data = d_output_data;


    int output_channels, output_height, output_width;
    cudnnGetTensor4dDescriptor(output_descriptor, CUDNN_DATA_FLOAT, 1, &output_channels, &output_height, &output_width);

    output_features.create(output_height, output_width, CV_32FC(output_channels));
    cudaMemcpy(output_features.ptr<float>(), d_output_data, output_channels * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(d_input_data);
    cudaFree(d_output_data);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
}

 




