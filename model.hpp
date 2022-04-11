#include <torch/torch.h>

#ifndef MODEL_HPP
#define MODEL_HPP

class VGG7Like : torch::nn::Module
{
public:
    // init method for this nn
    VGG7Like();

    // forward method for this nn
    torch::Tensor forward(torch::Tensor);

    // fully connected layer
    torch::nn::Linear fc;
    // convolutional layers
    torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5, conv6;
};

#endif
