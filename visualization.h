#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <vector>

// 绘制单个检测结果
void draw_detection(cv::Mat &image, const cv::Rect &bbox, int class_id, float score, const std::vector<std::string> &class_names, const std::vector<cv::Scalar> &colors);

// 在图像上绘制所有检测结果
void visualize_detections(cv::Mat &image, const std::vector<cv::Rect> &bboxes, const std::vector<int> &class_ids, const std::vector<float> &scores, const std::vector<std::string> &class_names, const std::vector<cv::Scalar> &colors) {
    for (size_t i = 0; i < bboxes.size(); ++i) {
        draw_detection(image, bboxes[i], class_ids[i], scores[i], class_names, colors);
    }
}

#endif // VISUALIZATION_H
