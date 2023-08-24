# SSD Detection with CUDA

This project implements a Single Shot MultiBox Detector (SSD) for object detection. It utilizes CUDA for GPU-accelerated computation, ensuring efficient and speedy performance.

## Features

- **Image Preprocessing**: The code contains utilities to preprocess images to make them suitable for object detection.
  
- **VGG16 Backbone**: Uses the VGG16 architecture for feature extraction.
  
- **Multi-scale Feature Maps**: Generates multiple feature maps to detect objects of varying sizes.
  
- **Object Detection**: Predicts bounding boxes and class scores for each object in the image.
  
- **Non-Maximum Suppression (NMS)**: Implements NMS to filter out redundant bounding boxes based on their scores and Intersection over Union (IoU) values.
  
- **CUDA and CUDNN Integration**: The implementation leverages CUDA and CUDNN to perform parallelized operations on the GPU, drastically improving the detection speed.

## Dependencies

- OpenCV: For image processing and manipulation.
- CUDA: For GPU-accelerated operations.
- CUDNN: Deep neural network library to boost the training and inference process with NVIDIA GPUs.


## Reference

@inproceedings{wang2022speed,
  title={Speed-up Single Shot Detector on GPU with CUDA},
  author={Wang, Chenyu and Endo, Toshio and Hirofuchi, Takahiro and Ikegami, Tsutomu},
  booktitle={International Conference on Software Engineering, Artificial Intelligence, Networking and Parallel/Distributed Computing},
  pages={89--106},
  year={2022},
  organization={Springer}
}


