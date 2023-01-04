#pragma once
#include "Tensor.h"
#include "Math.h"
#include <functional>

class Layer2d {
public:
    struct Topology {
        std::string layer_name;
        Tensor::Size input_size;
        int kernel_dim;
        int kernel_stride;
        int kernel_num;
        int padding;
        EActivation activation_func;
    };
public:
    Layer2d() {}
    Layer2d(const Tensor::Size& input_size, int kernel_dim, int kernel_stride, int padding, int kernel_num) :
        inputSize(input_size),
        kernel_dim(kernel_dim),
        kernel_stride(kernel_stride),
        kernel_num(kernel_num),
        kernel_padding(padding) {}

    virtual void feedForward(const Layer2d& prevLayer) = 0;
    virtual void backProp(const Tensor& dL_dA, double alpha) = 0;

    int getKernelDim() const { return kernel_dim; }
    int getKernelStride() const { return kernel_stride; }
    int getKernelPadding() const { return kernel_padding; }
    int getKernelNum() const { return kernel_num; }
    int getOutHeight() const { return out.height(); }
    int getOutWidth() const { return out.width(); }
    const Tensor& getOut() const { return out; }
    Tensor::Size getOutputSize() const { return outputSize; }
    Tensor::Size getInputSize() const { return inputSize; }
    const Tensor& getDlDx() const { return dL_dX; }
protected:
    Tensor::Size inputSize;
    Tensor::Size outputSize;

    Tensor out;
    Tensor X;
    Tensor dL_dX;
    int kernel_dim;
    int kernel_stride;
    int kernel_padding;
    int kernel_num;
};


class Conv2d : public Layer2d
{
public:
    Conv2d(const Tensor::Size& inSize, EActivation activation_func, int kernel_num, int stride, int padding, int kernel_dim);
    Conv2d() {}
    virtual void feedForward(const Layer2d& prevLayer) override;
    void setOutput(const Tensor& tensor);
    void backProp(const Tensor& dL_dA, double alpha=0.05) override;


private:
    std::vector<Tensor> kernels;
    std::vector<double> bias;
    std::vector<Tensor> dL_dK;
    std::vector<double> dL_db;
    std::function<double(double)> activation;
    std::function<double(double)> derivActivation;
};

class Maxpool2d : public Layer2d
{
public:
    Maxpool2d(Tensor::Size inputSize, int kernel_dim);
    void feedForward(const Layer2d& prevLayer) override;
    void backProp(const Tensor& dL_dA, double alpha) override;
private:
    Tensor mask;
};
