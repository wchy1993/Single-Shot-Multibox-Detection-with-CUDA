#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "image_preprocessing.h"
#include "utils.h"
#include "ssd.h"
#include "visualization.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <path_to_image> <path_to_binary_weights>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string binary_weights_path = argv[2];

    // Load the input image
    cv::Mat input_image = cv::imread(image_path);
    if (input_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    // Load weights and biases from binary file
    std::vector<float*> weights, biases, class_weights, class_biases, box_weights;
    std::vector<std::vector<float>> box_biases;
    if (!load_weights_and_biases_from_binary(binary_weights_path, weights, biases, class_weights, class_biases, box_weights, box_biases)) {
        return 1;
    }

    // Preprocess the image
    cv::cuda::GpuMat gpu_input_image;
    gpu_input_image.upload(input_image);

    // Run SSD detection
    std::vector<cv::Rect> final_bboxes;
    std::vector<int> final_class_ids;
    std::vector<float> final_scores;
    float confidence_threshold = 0.5;
    float iou_threshold = 0.5;

    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // Start the timer
    int64 start = cv::getTickCount();
    
    ssd_detect(cudnn_handle, weights, biases, class_weights, class_biases, box_weights, box_biases, gpu_input_image, final_bboxes, final_class_ids, final_scores, confidence_threshold, iou_threshold);
    
    // Stop the timer
    int64 end = cv::getTickCount();
    double duration = (end - start) / cv::getTickFrequency() * 1000;  // Convert to milliseconds
    
    std::cout << "SSD Detection took " << duration << " ms." << std::endl;

    cudnnDestroy(cudnn_handle);

    // Visualize the detection results
    cv::Mat detection_result = input_image.clone();
    visualize_detections(detection_result, final_bboxes, final_class_ids, final_scores);

    // Display the detection result
    cv::imshow("Detection Result", detection_result);
    cv::waitKey(0);

    return 0;
}






