#pragma once
#include <vector>
#include "Math.h"

class MNIST
{
public:
    static const int IMG_HEIGHT = 28;
    static const int IMG_WIDTH = 28;
    using Images = std::vector<std::pair<int, std::vector<double>>>;
    using LabeledSamples = std::vector<std::pair<Mat, Mat2>>;


    Mat2 GetImg(int label, int n) const;
    LabeledSamples GetLabeledImages() const;
    std::pair<LabeledSamples, LabeledSamples> GetTrainTestSamples() const;
    static const MNIST& Get();
    MNIST(const MNIST&) = delete;
    void operator=(const MNIST&) = delete;
private:
    MNIST();
    Images ReadMNIST();

private:
    Images images;
};

