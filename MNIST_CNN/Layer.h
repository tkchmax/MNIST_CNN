#pragma once
#include <iostream>
#include <functional>
#include "Math.h"

class Layer {
public:
    struct Topology {
        std::string layer_name;
        int input_size;
        int output_size;
        EActivation activation_func;
    };
public:
    virtual void feedForward(const Mat& X) = 0;
    virtual void backProp(const Mat& dL_dA, double alpha) = 0;

    const Mat& getOut() const { return out; }
    Tensor getTensorDlDx(const Tensor::Size& tensor_size) const;
    const Mat& getDlDx() const { return dL_dX; }
protected:
    Mat2 weights;
    Mat out;
    Mat dL_dX;
    Mat X;
    Mat bias;
};

class DenseLayer : public Layer{
public:
    DenseLayer(int input_size, int output_size, EActivation activation_func);

    void feedForward(const Mat& input) override;
    void backProp(const Mat& dL_dA, double alpha) override;
    void feedForward(const DenseLayer& prevLayer);
private:
    std::function<double(double)> activation;
    std::function<double(double)> derivActivation;
};

class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(int input_num, int output_num);
    void feedForward(const DenseLayer& prevLayer);
    void feedForward(const Mat& input) override;
    void backProp(const Mat& ground_truth, double alpha) override;
private:
};
