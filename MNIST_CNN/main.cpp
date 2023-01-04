#include <iostream>
#include "MNIST.h"
#include "Net.h"

int main()
{
    srand(time(0));

    auto labeled_2d = MNIST::Get().GetLabeledImages();
    std::pair<MNIST::LabeledSamples, MNIST::LabeledSamples> train_test = MNIST::Get().GetTrainTestSamples();

    MNIST::LabeledSamples train = train_test.first;
    MNIST::LabeledSamples test = train_test.second;

    Net net (
        {
            { "Conv2d", {28,28,1}, 5, 1, 16, 0, EActivation::ReLU},
            { "Maxpool", {24,24,16}, 2},
            { "Conv2d", {12,12,16}, 5, 1, 32, 0, EActivation::ReLU},
            { "Maxpool", {8,8,32}, 2}

        },
        {
            {"Softmax", 4*4*32, 10}
        }
    );

    for (int epoch = 0; epoch < 15; ++epoch) {
        net.train(train, 0.05);
    }
    net.test(test);
}

