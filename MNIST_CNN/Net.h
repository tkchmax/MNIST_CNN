#pragma once
#include "Layer.h"
#include "Layer2d.h"
#include "MNIST.h"
#include <memory>

class Net
{
public:
    Net(const std::vector<Layer2d::Topology>& topology2d, const std::vector<Layer::Topology>& topology);
    void train(const MNIST::LabeledSamples& train, double alpha);
    void test(const MNIST::LabeledSamples& test);
    Mat predict(const Tensor& input);
private:
    Mat forward(const Tensor& input);
    void backprop(const Mat& y, double alpha);
private:
    std::vector<std::unique_ptr<Layer2d>> layers2d;
    std::vector<std::unique_ptr<Layer>> layers;
};

