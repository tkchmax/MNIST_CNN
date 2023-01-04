#include "Layer.h"
#include <cassert>
#include <random>

SoftmaxLayer::SoftmaxLayer(int input_num, int output_num)
{
    out.resize(output_num);
    dL_dX.resize(input_num);
    bias.resize(output_num);

    weights.resize(output_num, Mat(input_num));
    for (int i = 0; i < weights.size(); ++i) {
        bias[i] = 0;
        for (int j = 0; j < weights[0].size(); ++j) {
            weights[i][j] = NRand(0,2.f/input_num);
        }
    }
}

void SoftmaxLayer::feedForward(const DenseLayer& prevLayer)
{
    feedForward(prevLayer.getOut());
}

void SoftmaxLayer::feedForward(const Mat& input)
{
    X = input;
    const int INPUT_SIZE = weights[0].size();
    const int OUTPUT_SIZE = weights.size();

    assert(input.size() == INPUT_SIZE);

    out = Mat(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out[i] += bias[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            out[i] += weights[i][j] * X[j];
        }
    }

    double sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out[i] = exp(out[i]);
        sum += out[i];
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        out[i] /= sum;
    }
}

void SoftmaxLayer::backProp(const std::vector<double>& y, double alpha)
{
    assert(out.size() == y.size());
    const int OUTPUT_SIZE = weights.size();
    const int INPUT_SIZE = weights[0].size();

    Mat2 dL_dW(OUTPUT_SIZE, Mat(INPUT_SIZE));
    Mat dL_dX(INPUT_SIZE);
    Mat dL_dZ(OUTPUT_SIZE);
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        dL_dZ[j] = out[j] - y[j];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            dL_dX[i] += weights[j][i] * dL_dZ[j];
            dL_dW[j][i] = X[i] * dL_dZ[j];
        }
    }

    //update weights 
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        bias[i] -= alpha * dL_dZ[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            weights[i][j] -= alpha * dL_dW[i][j];

        }
    }

}

DenseLayer::DenseLayer(int input_size, int output_size, EActivation activation_func)
{
    out.resize(output_size);
    X.resize(input_size);
    dL_dX.resize(input_size);
    bias.resize(output_size);

    weights.resize(output_size, Mat(input_size));
    for (int i = 0; i < output_size; ++i) {
        bias[i] = 0;
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = NRand(0, 2.f/input_size);
        }
    }

    switch (activation_func) {
    case EActivation::SIGMOID:
        activation = Sigmoid;
        derivActivation = SigmoidDeriv;
        break;
    case EActivation::ReLU:
        activation = ReLU;
        derivActivation = ReluDeriv;
        break;
    }
}

void DenseLayer::feedForward(const Mat& input)
{
    X = input;

    assert(input.size() == weights[0].size());
    assert(out.size() == weights.size());
    out = Mat(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
        out[i] += bias[i];
        for (int j = 0; j < weights[0].size(); ++j) {
            out[i] += weights[i][j] * input[j];
        }
    }

    for (int i = 0; i < out.size(); ++i) {
        out[i] = activation(out[i]);
    }
}

void DenseLayer::feedForward(const DenseLayer& prevLayer)
{
    feedForward(prevLayer.getOut());
}

void DenseLayer::backProp(const Mat& dL_dA, double alpha)
{
    const int INPUT_SIZE = weights[0].size();
    const int OUTPUT_SIZE = weights.size();

    assert(dL_dA.size() == OUTPUT_SIZE);

    Mat2 dL_dW(OUTPUT_SIZE, Mat(INPUT_SIZE));
    Mat dL_dX(INPUT_SIZE);
    Mat dL_dZ(dL_dA.size());
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        dL_dZ[j] = dL_dA[j] * derivActivation(out[j]);
        for (int i = 0; i < INPUT_SIZE; ++i) {
            dL_dX[i] += weights[j][i] * dL_dZ[j];
            dL_dW[j][i] = X[i] * dL_dZ[j];
        }
    }

    //update weights
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        bias[i] -= alpha * dL_dZ[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            weights[i][j] -= alpha * dL_dW[i][j];
        }
    }
}

Tensor Layer::getTensorDlDx(const Tensor::Size& tensor_size) const
{
    Tensor tensor(tensor_size);
    int hw = tensor_size.height * tensor_size.width;
    for (int d = 0; d < tensor_size.depth; ++d) {
        for (int i = 0; i < tensor_size.height; ++i) {
            for (int j = 0; j < tensor_size.width; ++j) {
                tensor(i, j, d) = dL_dX[d * hw + tensor_size.width * i + j];
            }
        }
    }
    return tensor;
}
