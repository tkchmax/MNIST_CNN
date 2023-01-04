#include "Math.h"
#include <cassert>
#include <random>

Tensor Conv(const Tensor& img, const Tensor& kernel, int stride, int padding)
{
    assert(img.depth() == kernel.depth());

    Tensor::Size outSize;
    outSize.width = (img.width() + 2 * padding - kernel.width()) / stride + 1;
    outSize.height = (img.height() + 2 * padding - kernel.height()) / stride + 1;
    outSize.depth = 1;
    Tensor out(outSize);

    for (int y = 0; y < outSize.height; ++y) {
        for (int x = 0; x < outSize.width; ++x) {
            double sum = 0;

            for (int i = 0; i < kernel.height(); ++i) {
                for (int j = 0; j < kernel.width(); ++j) {
                    int i0 = stride * y + i - padding;
                    int j0 = stride * x + j - padding;

                    if (i0 < 0 || i0 >= img.height() || j0 < 0 || j0 >= img.width()) {
                        continue;
                    }

                    for (int c = 0; c < img.depth(); ++c) {
                        sum += kernel(i, j, c) * img(i0, j0, c);
                    }
                }
            }
            out(y, x, 0) = sum;
        }
    }

    return out;
}

Mat Flatten(const Tensor& tensor)
{
    Mat flatted;
    for (int d = 0; d < tensor.depth(); ++d) {
        for (int i = 0; i < tensor.height(); ++i) {
            for (int j = 0; j < tensor.width(); ++j) {
                flatted.push_back(tensor(i, j, d));
            }
        }
    }
    return flatted;
}

double Sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double SigmoidDeriv(double x) {
    return Sigmoid(x) * (1 - Sigmoid(x));
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

double ReluDeriv(double x) {
    return x > 0 ? 1 : 0;
}

double Rand(int minus) {
    double r = rand() / double(RAND_MAX);
    //double r = rand();;

    return (minus && rand() % 2 == 0) ? -r : r;
}

double NRand(double mean, double stddev)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> nd(mean, stddev);
    return nd(gen);
}

int ArgMax(const Mat& arr) {
    assert(!arr.empty());
    int iMax = 0;
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] > arr[iMax]) {
            iMax = i;
        }
    }
    return iMax;
}

Mat operator+(const Mat& m1, const Mat& m2)
{
    assert(m1.size() == m2.size());
    Mat res(m1);
    for (int i = 0; i < m1.size(); ++i) {
        res[i] += m2[i];
    }
    return res;
}

Mat2 operator+(const Mat2& m2, double n)
{
    Mat2 res(m2);
    for (int i = 0; i < m2.size(); ++i) {
        for (int j = 0; j < m2[0].size(); ++j) {
            res[i][j] += n;
        }
    }
    return res;
}

void operator+=(Mat2& m1, const Mat2& m2)
{
    assert(m1.size() == m2.size());
    assert(m1[0].size() == m2[0].size());

    for (int i = 0; i < m1.size(); ++i) {
        for (int j = 0; j < m2.size(); ++j) {
            m1[i][j] += m2[i][j];
        }
    }
}

std::ostream& operator<<(std::ostream& out, const Mat& mat)
{
    out << "{";
    for (int i = 0; i < mat.size(); ++i) {
        out << mat[i];
        if (i != mat.size() - 1) {
            out << " ";
        }
    }
    out << "}";
    return out;
}

std::ostream& operator<<(std::ostream& out, const Mat2& mat2)
{
    out << "{";
    for (int i = 0; i < mat2.size(); ++i) {
        out << mat2[i];
        if (i != mat2.size() - 1) {
            out << std::endl;
        }
    }
    out << "}\n";
    return out << std::endl;
}

