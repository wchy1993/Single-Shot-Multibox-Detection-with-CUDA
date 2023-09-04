#include "utils.h"

bool load_weights_and_biases_from_binary(const std::string& binary_file, 
                                         std::vector<std::vector<float>>& weights, 
                                         std::vector<std::vector<float>>& biases) {
    std::ifstream file(binary_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary file: " << binary_file << std::endl;
        return false;
    }

    bool isWeight = true; 

    while (!file.eof()) {
        uint64_t tensor_size;
        file.read(reinterpret_cast<char*>(&tensor_size), sizeof(tensor_size));
        if (file.eof()) {
            break;
        }

        // Read the tensor data
        std::vector<float> tensor_data(tensor_size);
        for (size_t i = 0; i < tensor_size; ++i) {
            float value;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            tensor_data[i] = value;
        }

        if (isWeight) {
            weights.push_back(tensor_data);
        } else {
            biases.push_back(tensor_data);
        }

        isWeight = !isWeight;  // Toggle the flag
    }

    file.close();
    return true;
}
