#include "visualization.h"

void draw_detection(cv::Mat &image, const cv::Rect &bbox, int class_id, float score, const std::vector<std::string> &class_names, const std::vector<cv::Scalar> &colors) {
    // Draw bounding box
    cv::rectangle(image, bbox, colors[class_id], 2);

    // Prepare text overlay
    std::string label = class_names[class_id] + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    // Draw filled rectangle to put text on
    cv::Rect text_rect(bbox.x, bbox.y - label_size.height - 5, label_size.width, label_size.height + 5);
    cv::rectangle(image, text_rect, colors[class_id], -1);

    // Put text on the rectangle
    cv::putText(image, label, cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

