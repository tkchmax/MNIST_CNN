#include "Layer2d.h"
#include <cassert>

Conv2d::Conv2d(const Tensor::Size& inSize, EActivation activation_func, int kernel_num, int stride, int padding, int kernel_dim)
{
    this->kernel_dim = kernel_dim;
    this->kernel_num = kernel_num;
    this->kernel_stride = stride;
    this->kernel_padding = padding;

    inputSize = inSize;
    outputSize.depth = kernel_num;
    outputSize.height = (inputSize.height + 2 * padding - kernel_dim) / stride + 1;
    outputSize.width = (inputSize.width + 2 * padding - kernel_dim) / stride + 1;

    kernels.resize(kernel_num, Tensor(kernel_dim, kernel_dim, inSize.depth));
    dL_dK.resize(kernel_num, Tensor(kernel_dim, kernel_dim, inSize.depth));

    bias.resize(kernel_num);
    dL_db.resize(kernel_num);

    dL_dX = Tensor(inputSize);
    out = Tensor(outputSize);

    switch (activation_func) {
    case EActivation::ReLU:
        activation = ReLU;
        derivActivation = ReluDeriv;
        break;
    case EActivation::SIGMOID:
        activation = Sigmoid;
        derivActivation = SigmoidDeriv;
        break;
    }

    for (int n = 0; n < kernels.size(); ++n) {
        bias[n] = 0.01;
        for (int d = 0; d < kernels[0].depth(); ++d) {
            for (int i = 0; i < kernels[0].height(); ++i) {
                for (int j = 0; j < kernels[0].width(); ++j) {
                    kernels[n](i, j, d) = NRand(0, 2.f/(kernel_dim * kernel_dim));
                }
            }
        }
    }

}

void Conv2d::feedForward(const Layer2d& prevLayer)
{
    X = prevLayer.getOut();

    for (int k = 0; k < kernel_num; ++k) {
        Tensor convolved = Conv(X, kernels[k], kernel_stride, kernel_padding) + bias[k];
        out.copy(convolved, 0, k);
    }

    for (int i = 0; i < out.getRawSize(); ++i) {
        out[i] = activation(out[i]);
    }
}


void Conv2d::setOutput(const Tensor& tensor)
{
    out = tensor;
    outputSize = tensor.getSize();
    inputSize = tensor.getSize();
}


void Conv2d::backProp(const Tensor& dL_dA, double alpha)
{
    assert(dL_dA.depth() == kernel_num);
    assert(Z.getRawSize() == dL_dA.getRawSize());

    Tensor dL_dZ = dL_dA;
    for (int i = 0; i < dL_dA.getRawSize(); ++i) {
        dL_dZ[i] = dL_dA[i] * derivActivation(out[i]);
    }

    for (int k = 0; k < kernel_num; ++k) {
        for (int c = 0; c < inputSize.depth; ++c) {
            Tensor convolved(inputSize);
            convolved = Conv(X(c), dL_dZ(k), kernel_stride, kernel_padding);
            dL_dK[k].set(convolved, c);
        }
    }


    assert(dL_dK[0].depth() == inputSize.depth);

    int pad = kernel_dim - 1 - kernel_padding;

    for (int c = 0; c < inputSize.depth; ++c) {
        Tensor dL_dXc(inputSize.height, inputSize.width, 1);
        for (int k = 0; k < kernel_num; ++k) {
            dL_dXc += Conv(dL_dZ(k), kernels[k](c).turn180(), kernel_stride, pad);
        }
        dL_dX.set(dL_dXc, c);
    }

    assert(dL_dZ.depth() == dL_db.size());
    for (int k = 0; k < kernel_num; ++k) {
        for (int i = 0; i < dL_dZ.height(); ++i) {
            for (int j = 0; j < dL_dZ.width(); ++j) {
                dL_db[k] += dL_dZ(i, j, k);
            }
        }
    }

    //update weights
    for (int k = 0; k < kernel_num; ++k) {
        bias[k] -= alpha * dL_db[k];
        dL_db[k] = 0;
        for (int c = 0; c < inputSize.depth; ++c) {
            for (int n = 0; n < kernel_dim; ++n) {
                for (int m = 0; m < kernel_dim; ++m) {
                    kernels[k](n, m, c) -= alpha * dL_dK[k](n, m, c);
                    dL_dK[k](n, m, c) = 0;
                }
            }
        }
    }
}

Maxpool2d::Maxpool2d(Tensor::Size inputSize, int kernel_dim)
{

    this->kernel_dim = kernel_dim;
    this->kernel_stride = kernel_dim;
    this->kernel_padding = 0;
    this->kernel_num = inputSize.depth;

    this->inputSize = inputSize;
    outputSize.depth = inputSize.depth;
    outputSize.height = (inputSize.height - kernel_dim) / kernel_stride + 1;
    outputSize.width = (inputSize.width - kernel_dim) / kernel_stride + 1;

    dL_dX = Tensor(inputSize);
    X = Tensor(inputSize);
    mask = Tensor(inputSize);
    out = Tensor(outputSize);
}

void Maxpool2d::feedForward(const Layer2d& prevLayer)
{
    const auto& prevOut = prevLayer.getOut();
    assert(out.depth() == prevOut.depth());
    X = prevOut;

    for (int c = 0; c < inputSize.depth; ++c) {
        for (int y = 0; y < inputSize.height; y += kernel_dim) {
            for (int x = 0; x < inputSize.width; x += kernel_dim) {
                int yMax = y;
                int xMax = x;
                double max = X(y, x, c);

                for (int i = y; i < y + kernel_dim; ++i) {
                    for (int j = x; j < x + kernel_dim; ++j) {
                        double val = X(i, j, c);
                        mask(i, j, c) = 0;

                        if (val > max) {
                            yMax = i;
                            xMax = j;
                            max = val;
                        }
                    }
                }
                out(y / kernel_dim, x / kernel_dim, c) = max;
                mask(yMax, xMax, c) = 1;
            }
        }
    }

}

void Maxpool2d::backProp(const Tensor& dL_dA, double alpha)
{
    for (int c = 0; c < kernel_num; ++c) {
        for (int i = 0; i < inputSize.height; ++i) {
            for (int j = 0; j < inputSize.width; ++j) {
                dL_dX(i, j, c) = dL_dA(i / kernel_dim, j / kernel_dim, c) * mask(i, j, c);
            }
        }
    }
}
