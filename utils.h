#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>


// Load weights from a .pth file using PyTorch C++ API
bool load_weights_from_binary(const std::string& binary_file, std::vector<std::vector<float>>& weights)

#endif // UTILS_H
