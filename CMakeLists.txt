cmake_minimum_required(VERSION 3.10)
project(SSD_Detection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(SSD_Detection
    main.cpp
    image_preprocessing.cpp
    utils.cpp
    ssd.cpp
    visualization.cpp
)

target_link_libraries(SSD_Detection
    ${OpenCV_LIBS}
    ${CUDAToolkit_LIBRARIES}
    cudnn
)
