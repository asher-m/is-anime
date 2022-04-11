#include <torch/torch.h>

#include "model.hpp"

VGG7Like::VGG7Like(){
    conv1(torch::nn::Conv2dOptions(3, 32, 3).stride(1).bias(false));
    conv2(torch::nn::Conv2dOptions(32, 32, 3).stride(2).bias(false));
    conv3(torch::nn::Conv2dOptions(32, 64, 3).stride(1).bias(false));
    conv4(torch::nn::Conv2dOptions(64, 64, 3).stride(2).bias(false));
    conv5(torch::nn::Conv2dOptions(64, 64, 3).stride(2).bias(false));
    conv6(torch::nn::Conv2dOptions(64, 32, 3).stride(2).bias(false));
    fc(torch::nn::LinearOptions(800, 1));
};
