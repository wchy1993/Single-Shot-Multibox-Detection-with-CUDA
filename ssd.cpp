#include "ssd.h"
#include "image_preprocessing.h"
#include "vgg16.h"


void generate_multiscale_feature_maps(cudnnHandle_t cudnn_handle, const std::vector<std::vector<float>> &extra_conv_weights, const std::vector<std::vector<float>> &extra_conv_biases, const cv::cuda::GpuMat &vgg16_output, std::vector<cv::cuda::GpuMat> &feature_maps) {
    // 根据SSD论文中的额外卷积层结构，设置卷积层参数
    // 每个元素表示 {kernel_size, kernel_size, stride, padding}
    std::vector<std::vector<int>> extra_conv_params = {
        {1, 1, 1, 0},
        {3, 3, 2, 1},
        {1, 1, 1, 0},
        {3, 3, 2, 1},
        {1, 1, 1, 0},
        {3, 3, 2, 1},
        {1, 1, 1, 0}
    };

    cv::cuda::GpuMat input_features = vgg16_output;
    for (size_t i = 0; i < extra_conv_weights.size(); ++i) {
        cv::cuda::GpuMat output_features;
        perform_convolution(cudnn_handle, extra_conv_params[i][0], extra_conv_params[i][1], extra_conv_params[i][2], extra_conv_params[i][3], input_features, extra_conv_weights[i], extra_conv_biases[i], output_features);

        // 保存生成的特征图
        feature_maps.push_back(output_features);

        // 将当前输出作为下一个卷积层的输入
        input_features = output_features;
    }
}

void predict_boxes_and_classes(cudnnHandle_t cudnn_handle, const std::vector<cv::cuda::GpuMat> &feature_maps, const std::vector<std::vector<float>> &class_weights, const std::vector<std::vector<float>> &class_biases, const std::vector<std::vector<float>> &box_weights, const std::vector<std::vector<float>> &box_biases, std::vector<cv::cuda::GpuMat> &class_scores, std::vector<cv::cuda::GpuMat> &box_deltas) {
    int num_classes = class_weights.size();  // 类别数量
    int num_boxes = box_weights.size();      // 边界框数量

    for (size_t i = 0; i < feature_maps.size(); ++i) {
        cv::cuda::GpuMat class_scores_map, box_deltas_map;

        // 执行类别预测卷积
        perform_convolution(cudnn_handle, 3, 3, 1, 1, feature_maps[i], class_weights[i], class_biases[i], class_scores_map);
        
        // 执行边界框回归卷积
        perform_convolution(cudnn_handle, 3, 3, 1, 1, feature_maps[i], box_weights[i], box_biases[i], box_deltas_map);

        // 保存类别得分和边界框坐标调整值
        class_scores.push_back(class_scores_map);
        box_deltas.push_back(box_deltas_map);
    }
}

void decode_predictions(const std::vector<cv::cuda::GpuMat> &class_scores, const std::vector<cv::cuda::GpuMat> &box_deltas, const std::vector<cv::Rect2f> &prior_boxes, float score_threshold, std::vector<cv::Rect2f> &decoded_boxes, std::vector<int> &decoded_labels, std::vector<float> &decoded_scores) {
    for (size_t i = 0; i < class_scores.size(); ++i) {
        for (int y = 0; y < class_scores[i].rows; ++y) {
            for (int x = 0; x < class_scores[i].cols; ++x) {
                // 获取类别得分
                float max_score = -1.0f;
                int max_class_idx = -1;
                for (int c = 0; c < class_scores[i].channels(); ++c) {
                    float score = class_scores[i].at<float>(y, x, c);
                    if (score > max_score) {
                        max_score = score;
                        max_class_idx = c;
                    }
                }

                // 检查得分阈值
                if (max_score < score_threshold) {
                    continue;
                }

                // 获取边界框调整值
                cv::Vec4f delta = box_deltas[i].at<cv::Vec4f>(y, x);

                // 应用边界框调整值
                cv::Rect2f prior_box = prior_boxes[y * class_scores[i].cols + x];
                cv::Rect2f decoded_box;
                decoded_box.x = prior_box.x + prior_box.width * delta[0];
                decoded_box.y = prior_box.y + prior_box.height * delta[1];
                decoded_box.width = prior_box.width * std::exp(delta[2]);
                decoded_box.height = prior_box.height * std::exp(delta[3]);

                // 保存解码后的边界框、类别和得分
                decoded_boxes.push_back(decoded_box);
                decoded_labels.push_back(max_class_idx);
                decoded_scores.push_back(max_score);
            }
        }
    }
}


// void apply_nms(float nms_threshold, std::vector<cv::Rect2f> &decoded_boxes, std::vector<int> &decoded_labels, std::vector<float> &decoded_scores) {
//     std::vector<cv::dnn::DetectionOutputLayer<float>::Box> boxes;
//     for (const auto &rect : decoded_boxes) {
//         boxes.push_back({rect.x, rect.y, rect.x + rect.width, rect.y + rect.height});
//     }

//     cv::dnn::DetectionOutputLayer<float>::nms(boxes, decoded_scores, decoded_labels, nms_threshold);

//     // 将检测结果更新为执行NMS后的结果
//     decoded_boxes.resize(boxes.size());
//     for (size_t i = 0; i < boxes.size(); ++i) {
//         decoded_boxes[i] = cv::Rect2f(boxes[i].x, boxes[i].y, boxes[i].width - boxes[i].x, boxes[i].height - boxes[i].y);
//     }
// }

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

__device__ float IoU(BoundingBox a, BoundingBox b) {
  float x1 = max(a.x1, b.x1);
  float y1 = max(a.y1, b.y1);
  float x2 = min(a.x2, b.x2);
  float y2 = min(a.y2, b.y2);

  float intersection = max(0.0f, x2 - x1 + 1) * max(0.0f, y2 - y1 + 1);
  float areaA = (a.x2 - a.x1 + 1) * (a.y2 - a.y1 + 1);
  float areaB = (b.x2 - b.x1 + 1) * (b.y2 - b.y1 + 1);

  return intersection / (areaA + areaB - intersection);
}



__global__ void nms_kernel(BoundingBox* d_bboxes, int* d_nms, int num_bboxes, float threshold) {
  extern __shared__ float shared_mem[];
  float* shared_coords = shared_mem;
  float* shared_scores = (float*)&shared_coords[4 * blockDim.x];

  cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_bboxes) return;

  BoundingBox my_bbox = d_bboxes[idx];
  shared_coords[threadIdx.x * 4] = my_bbox.x1;
  shared_coords[threadIdx.x * 4 + 1] = my_bbox.y1;
  shared_coords[threadIdx.x * 4 + 2] = my_bbox.x2;
  shared_coords[threadIdx.x * 4 + 3] = my_bbox.y2;
  shared_scores[threadIdx.x] = my_bbox.score;
  block.sync();

  for (int i = threadIdx.x + 1; i < blockDim.x && i + blockIdx.x * blockDim.x < num_bboxes; ++i) {
    BoundingBox other_bbox;
    other_bbox.x1 = shared_coords[i * 4];
    other_bbox.y1 = shared_coords[i * 4 + 1];
    other_bbox.x2 = shared_coords[i * 4 + 2];
    other_bbox.y2 = shared_coords[i * 4 + 3];
    float iou = IoU(my_bbox, other_bbox);

    uint32_t mask = __ballot_sync(0xFFFFFFFF, iou > threshold);
    if (threadIdx.x % 32 == 0) {
      d_nms[i + blockIdx.x * blockDim.x] = (mask == 0);
    }

    block.sync();
  }
}



std::vector<BoundingBox>  apply_nms(std::vector<BoundingBox>& bboxes, float threshold, int top_k) {
  int num_bboxes = bboxes.size();
  std::vector<int> nms_flags(num_bboxes, 1);


  for (int i = 0; i < num_bboxes; ++i) {
    bboxes[i].label = i % 20; 
  }

  thrust::device_vector<BoundingBox> d_bboxes = bboxes;
  thrust::sort(thrust::device, d_bboxes.begin(), d_bboxes.end(), BBoxCompare());
  num_bboxes = min(top_k, num_bboxes);
  d_bboxes.resize(num_bboxes);

  thrust::device_vector<int> d_nms = nms_flags;
  d_nms.resize(num_bboxes);
 
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_bboxes + threadsPerBlock - 1) / threadsPerBlock;
 
  nms_kernel<<<blocksPerGrid, threadsPerBlock, 5 * threadsPerBlock * sizeof(float)>>>
  (thrust::raw_pointer_cast(d_bboxes.data()), thrust::raw_pointer_cast(d_nms.data()), num_bboxes, threshold);
  cudaDeviceSynchronize();

  // 复制结果回主机内存
  thrust::host_vector<int> h_nms = d_nms;
  std::vector<BoundingBox> result;
  for (int i = 0; i < num_bboxes; ++i) {
    if (h_nms[i]) {
      result.push_back(d_bboxes[i]);
    }
  }

  return result;
}


void ssd_detect(cudnnHandle_t cudnn_handle, const std::vector<float> &weights, const cv::cuda::GpuMat &input_image, std::vector<cv::Rect> &final_bboxes, std::vector<int> &final_class_ids, std::vector<float> &final_scores, float confidence_threshold, float iou_threshold) {
    // 数据预处理
    cv::cuda::GpuMat preprocessed_image;
    preprocess_image_batch(input_image, preprocessed_image);

    // 特征提取
    cv::cuda::GpuMat extracted_features;
    extract_features(cudnn_handle, weights, preprocessed_image, extracted_features);

    // 生成多尺度特征图
    std::vector<cv::cuda::GpuMat> multiscale_feature_maps;
    generate_multiscale_feature_maps(cudnn_handle, weights, extracted_features, multiscale_feature_maps);

    // 类别预测和边界框回归
    cv::cuda::GpuMat class_predictions, bbox_predictions;
    predict_classes_and_bboxes(cudnn_handle, weights, multiscale_feature_maps, class_predictions, bbox_predictions);

    // 解码预测结果
    std::vector<cv::Rect> decoded_bboxes;
    std::vector<int> decoded_class_ids;
    std::vector<float> decoded_scores;
    decode_predictions(class_predictions, bbox_predictions, decoded_bboxes, decoded_class_ids, decoded_scores);

    // 置信度阈值过滤
    std::vector<cv::Rect> filtered_bboxes;
    std::vector<int> filtered_class_ids;
    std::vector<float> filtered_scores;
    filter_predictions_by_confidence(decoded_bboxes, decoded_class_ids, decoded_scores, confidence_threshold, filtered_bboxes, filtered_class_ids, filtered_scores);

    // 非极大值抑制
    non_max_suppression(filtered_bboxes, filtered_class_ids, filtered_scores, iou_threshold, final_bboxes, final_class_ids, final_scores);
}
