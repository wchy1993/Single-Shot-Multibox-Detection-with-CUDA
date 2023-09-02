#include "ssd.h"
#include "image_preprocessing.h"
#include "vgg16.h"


void generate_multiscale_feature_maps(cudnnHandle_t cudnn_handle, const std::vector<std::vector<float>> &extra_conv_weights, const std::vector<std::vector<float>> &extra_conv_biases, const cv::cuda::GpuMat &vgg16_output, std::vector<cv::cuda::GpuMat> &feature_maps) {
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
    
        feature_maps.push_back(output_features);
       
        input_features = output_features;
    }
}

void apply_softmax(cudnnHandle_t cudnn_handle, cv::cuda::GpuMat &data) {
    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, data.channels(), data.rows, data.cols);

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, data_desc, data.ptr<float>(), &beta, data_desc, data.ptr<float>());

    cudnnDestroyTensorDescriptor(data_desc);
}


void predict_classes_and_bboxes(cudnnHandle_t cudnn_handle, const std::vector<cv::cuda::GpuMat> &feature_maps, const std::vector<std::vector<float>> &class_weights, const std::vector<std::vector<float>> &class_biases, const std::vector<std::vector<float>> &box_weights, const std::vector<std::vector<float>> &box_biases, std::vector<cv::cuda::GpuMat> &class_scores, std::vector<cv::cuda::GpuMat> &box_deltas) {
    int num_classes = class_weights.size();  // 类别数量
    int num_boxes = box_weights.size();      // 边界框数量

    for (size_t i = 0; i < feature_maps.size(); ++i) {
        cv::cuda::GpuMat class_scores_map, box_deltas_map;

        // 执行类别预测卷积
        perform_convolution(cudnn_handle, 3, 3, 1, 1, feature_maps[i], class_weights[i], class_biases[i], class_scores_map);
        
        // 应用softmax
        apply_softmax(cudnn_handle, class_scores_map);

        // 执行边界框回归卷积
        perform_convolution(cudnn_handle, 3, 3, 1, 1, feature_maps[i], box_weights[i], box_biases[i], box_deltas_map);

        // 保存类别得分和边界框坐标调整值
        class_scores.push_back(class_scores_map);
        box_deltas.push_back(box_deltas_map);
    }
}


void decode_predictions(const std::vector<cv::cuda::GpuMat> &class_scores, const std::vector<cv::cuda::GpuMat> &box_deltas, const std::vector<cv::Rect2f> &prior_boxes, float score_threshold = 0.5, std::vector<cv::Rect2f> &decoded_boxes, std::vector<int> &decoded_labels, std::vector<float> &decoded_scores) {
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

                if (max_score < score_threshold) {
                    continue;
                }

                cv::Vec4f delta = box_deltas[i].at<cv::Vec4f>(y, x);

                cv::Rect2f prior_box = prior_boxes[y * class_scores[i].cols + x];
                cv::Rect2f decoded_box;
                decoded_box.x = prior_box.x + prior_box.width * delta[0];
                decoded_box.y = prior_box.y + prior_box.height * delta[1];
                decoded_box.width = prior_box.width * std::exp(delta[2]);
                decoded_box.height = prior_box.height * std::exp(delta[3]);

                decoded_boxes.push_back(decoded_box);
                decoded_labels.push_back(max_class_idx);
                decoded_scores.push_back(max_score);
            }
        }
    }
}





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



std::vector<BoundingBox> non_max_suppression(const std::vector<cv::Rect2f>& decoded_boxes, const std::vector<int>& decoded_labels, const std::vector<float>& decoded_scores, float threshold, int top_k) {
    int num_bboxes = decoded_boxes.size();
    std::vector<BoundingBox> bboxes(num_bboxes);
    std::vector<int> nms_flags(num_bboxes, 1);

    for (int i = 0; i < num_bboxes; ++i) {
        bboxes[i].x1 = decoded_boxes[i].x;
        bboxes[i].y1 = decoded_boxes[i].y;
        bboxes[i].x2 = decoded_boxes[i].x + decoded_boxes[i].width;
        bboxes[i].y2 = decoded_boxes[i].y + decoded_boxes[i].height;
        bboxes[i].score = decoded_scores[i];
        bboxes[i].label = decoded_labels[i];
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

    thrust::host_vector<int> h_nms = d_nms;
    std::vector<BoundingBox> result;
    for (int i = 0; i < num_bboxes; ++i) {
        if (h_nms[i]) {
            result.push_back(d_bboxes[i]);
        }
    }

    return result;
}


void ssd_detect(cudnnHandle_t cudnn_handle, const std::vector<float*> &weights,const std::vector<float*> &biases const cv::cuda::GpuMat &input_image, std::vector<cv::Rect> &final_bboxes, std::vector<int> &final_class_ids, std::vector<float> &final_scores, float confidence_threshold, float iou_threshold) {
    // Data preprocessing
    cv::cuda::GpuMat preprocessed_image;
    preprocess_image_batch(input_image, preprocessed_image);

    // Feature extraction
    cv::cuda::GpuMat extracted_features;
    extract_features(cudnn_handle, weights, biases, input_image, extracted_features);
    // Generate multiscale feature maps
    std::vector<cv::cuda::GpuMat> multiscale_feature_maps;
    generate_multiscale_feature_maps(cudnn_handle, weights, extracted_features, multiscale_feature_maps);

    // Class prediction and bounding box regression
    cv::cuda::GpuMat class_predictions, bbox_predictions;
    predict_classes_and_bboxes(cudnn_handle, weights, multiscale_feature_maps, class_predictions, bbox_predictions);

    // Decode predictions
    std::vector<cv::Rect> decoded_bboxes;
    std::vector<int> decoded_class_ids;
    std::vector<float> decoded_scores;
    decode_predictions(class_predictions, bbox_predictions, decoded_bboxes, decoded_class_ids, decoded_scores);


    // Non-maximum suppression 
    std::vector<BoundingBox> suppressed_bboxes = non_max_suppression(decoded_bboxes, decoded_class_ids, decoded_scores, iou_threshold, top_k);

    for (const BoundingBox& bbox : suppressed_bboxes) {
        final_bboxes.push_back(cv::Rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1));
        final_class_ids.push_back(bbox.label);
        final_scores.push_back(bbox.score);
    }

}
