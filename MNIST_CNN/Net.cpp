#include "Net.h"
#include <cassert>

namespace {
    void ShowImg(const Mat2& num) {
        for (int h = 0; h < 28; ++h) {
            for (int w = 0; w < 28; ++w) {
                if (num[h][w] != 0) {
                    std::cout << "@ ";
                }
                else {
                    std::cout << "- ";
                }
            }
            std::cout << std::endl;
        }
    }
}

Net::Net(const std::vector<Layer2d::Topology>& topology2d, const std::vector<Layer::Topology>& topology)
{
    for (const auto& t : topology2d) {
        if (t.layer_name == "Conv2d") {
            layers2d.push_back(std::make_unique<Conv2d>(Conv2d(t.input_size, t.activation_func, t.kernel_num, t.kernel_stride, t.padding, t.kernel_dim)));
        }
        else if (t.layer_name == "Maxpool") {
            layers2d.push_back(std::make_unique<Maxpool2d>(Maxpool2d(t.input_size, t.kernel_dim)));
        }
    }

    for (const auto& t : topology) {
        if (t.layer_name == "Dense") {
            layers.push_back(std::make_unique<DenseLayer>(DenseLayer(t.input_size, t.output_size, t.activation_func)));
        }
        else if (t.layer_name == "Softmax") {
            layers.push_back(std::make_unique<SoftmaxLayer>(SoftmaxLayer(t.input_size, t.output_size)));
        }
    }
}

void Net::train(const MNIST::LabeledSamples& train, double alpha)
{
    int corrects = 0;
    for (int i = 0; i < train.size(); ++i) {
        Mat out = forward(train[i].second);

        corrects += (ArgMax(out) == ArgMax(train[i].first)) ? 1 : 0;
        if (i % 100 == 0) {
            std::cout << "#" << i << " " << double(corrects) / 100 << std::endl;
            corrects = 0;
        }

        backprop(train[i].first, alpha);
    }
}

void Net::test(const MNIST::LabeledSamples& test)
{
    double corrects = 0;
    for (int i = 0; i < test.size(); ++i) {
        Mat out = predict(test[i].second);
        corrects += (ArgMax(out) == ArgMax(test[i].first)) ? 1 : 0;
    }
    std::cout << "correct/total = " << corrects / test.size() << std::endl;
}

Mat Net::predict(const Tensor& input)
{
    Conv2d in;
    in.setOutput(input);
    if (!layers2d.empty()) {
        layers2d[0]->feedForward(in);
    }
    for (int d = 1; d < layers2d.size(); ++d) {
        layers2d[d]->feedForward(*layers2d[d - 1]);
    }
    Mat flatted = (layers2d.empty()) ? Flatten(in.getOut()) : Flatten(layers2d.back()->getOut());
    layers[0]->feedForward(flatted);
    for (int f = 1; f < layers.size(); ++f) {
        layers[f]->feedForward(layers[f - 1]->getOut());
    }
    return layers.back()->getOut();
}

Mat Net::forward(const Tensor& input)
{
    assert(!layers.empty());

    if (!layers2d.empty()) {
        Conv2d in;
        in.setOutput(input);
        layers2d[0]->feedForward(in);
        for (int i = 1; i < layers2d.size(); ++i) {
            layers2d[i]->feedForward(*layers2d[i - 1]);
        }

    }

    Mat flatted = (layers2d.empty()) ? Flatten(input) : Flatten(layers2d.back()->getOut());
    layers[0]->feedForward(flatted);
    for (int i = 1; i < layers.size(); ++i) {
        layers[i]->feedForward(layers[i - 1]->getOut());
    }

    return layers.back()->getOut();
}

void Net::backprop(const Mat& y, double alpha)
{
    assert(!layers.empty());

    layers.back()->backProp(y, alpha);
    for (int i = layers.size() - 2; i >= 0; --i) {
        layers[i]->backProp(layers[i + 1]->getDlDx(), alpha);
    }

    if (!layers2d.empty()) {
        Tensor tensored_dL_dX = layers[0]->getTensorDlDx(layers2d.back()->getOutputSize());
        layers2d.back()->backProp(tensored_dL_dX, alpha);
        for (int i = layers2d.size() - 2; i >= 0; --i) {
            layers2d[i]->backProp(layers2d[i + 1]->getDlDx(), alpha);
        }
    }
}
