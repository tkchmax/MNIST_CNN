#include "Tensor.h"
#include <cassert>

Tensor::Tensor(int height, int width, int depth)
{
    size.height = height;
    size.width = width;
    size.depth = depth;
    values.resize(height * width * depth);
    hw = height * width;
}

Tensor::Tensor(const Mat2& mat)
{
    size.depth = 1;
    size.height = mat.size();
    size.width = mat[0].size();
    hw = size.height * size.width;

    values.resize(size.height * size.width);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            values[i * size.width + j] = mat[i][j];
        }
    }
}

Tensor::Tensor(const Mat3& mat)
{
    size.depth = mat.size();
    size.height = mat[0].size();
    size.width = mat[0][0].size();
    hw = size.height * size.width;

    values.resize(size.depth * size.height * size.width);
    for (int d = 0; d < size.depth; ++d) {
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                values[d * hw + i * size.width + j] = mat[d][i][j];
            }
        }
    }
}

Tensor Tensor::turn180()
{
    Tensor turned(size);
    for (int d = 0; d < size.depth; ++d) {
        for (int i = size.height - 1; i >= 0; --i) {
            for (int j = size.width - 1; j >= 0; --j) {
                turned(size.height - i - 1, size.width - j -1, d) = (*this)(i, j, d);
            }
        }
    }
    return turned;
}

void Tensor::set(const Mat2& mat, int d)
{
    assert(mat.size() == size.height);
    assert(mat[0].size() == size.width);

    for (int i = 0; i < mat.size(); ++i) {
        for (int j = 0; j < mat[0].size(); ++j) {
            operator()(i, j, d) = mat[i][j];
        }
    }
}

void Tensor::set(const Tensor& tensor, int d)
{
    assert(tensor.depth() == 1);
    assert(tensor.height() == size.height);
    assert(tensor.width() == size.width);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            (*this)(i, j, d) = tensor(i, j, 0);
        }
    }
}

void Tensor::copy(const Tensor& toCopy, int from_depth, int to_depth)
{
    assert(size.height == toCopy.size.height);
    assert(size.width == toCopy.size.width);

    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            (*this)(i, j, to_depth) = toCopy(i, j, from_depth);
        }
    }
}

double& Tensor::operator()(int i, int j, int d)
{
    return values[d * hw + i * size.width + j];
}

double Tensor::operator()(int i, int j, int d) const
{
    return values[d * hw + i * size.width + j];
}

Tensor Tensor::operator()(int d)
{
    Tensor res(size.height, size.width, 1);
    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            res(i, j, 0) = (*this)(i, j, d);
        }
    }
    return res;
}

Tensor Tensor::operator+(const Mat& mat)
{
    assert(mat.size() == size.depth);
    Tensor res(*this);
    for (int d = 0; d < size.depth; ++d) {
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                res(i, j, d) += mat[d];
            }
        }
    }
    return res;
}

Tensor Tensor::operator+(double num)
{
    Tensor res(*this);
    for (int d = 0; d < size.depth; ++d) {
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                res(i, j, d) += num;
            }
        }
    }
    return res;
}

void Tensor::operator+=(const Tensor& other)
{
    assert(size.depth == other.depth());
    assert(size.height == other.height());
    assert(size.width == other.width());

    for (int d = 0; d < size.depth; ++d) {
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                (*this)(i, j, d) += other(i, j, d);
            }
        }
    }
}

double& Tensor::operator[](int index)
{
    return values[index];
}

const double& Tensor::operator[](int index) const
{
    return values[index];
}

std::ostream& operator<<(std::ostream& out, const Tensor& t)
{
    for (int d = 0; d < t.size.depth; ++d) {
        for (int i = 0; i < t.size.height; ++i) {
            for (int j = 0; j < t.size.width; ++j) {
                out << t(i, j, d) << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }
    return out;
}
