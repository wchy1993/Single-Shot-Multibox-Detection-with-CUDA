// preprocessing.h

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

void preprocess_image_batch(const std::vector<cv::Mat> &input_images, std::vector<cv::cuda::GpuMat> &output_images, int output_width, int output_height);

#endif // PREPROCESSING_H
