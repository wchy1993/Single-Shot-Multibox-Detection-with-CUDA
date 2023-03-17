// preprocessing.cpp

#include "preprocessing.h"

void preprocess_image_batch(const std::vector<cv::Mat> &input_images, std::vector<cv::cuda::GpuMat> &output_images, int output_width, int output_height) {
    // 图像归一化参数
    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std(0.229, 0.224, 0.225);

    output_images.resize(input_images.size());
    
    for (size_t i = 0; i < input_images.size(); ++i) {
        const cv::Mat& input_image = input_images[i];
        cv::cuda::GpuMat& output_image = output_images[i];

        // 调整图像大小
        cv::cuda::GpuMat resized_image;
        cv::cuda::resize(input_image, resized_image, cv::Size(output_width, output_height));

        // 将输入图像转换为32位浮点数
        cv::cuda::GpuMat float_image;
        resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

        // 将图像从BGR转换为RGB
        cv::cuda::GpuMat rgb_image;
        cv::cuda::cvtColor(float_image, rgb_image, cv::COLOR_BGR2RGB);

        // 减去均值并除以标准差
        cv::cuda::subtract(rgb_image, mean, rgb_image, cv::noArray(), -1);
        cv::cuda::divide(rgb_image, std, output_image, 1, -1);
    }
}



