#pragma once
#include <vector>
#include <iostream>

typedef std::vector<double> Mat;
typedef std::vector<std::vector<double>> Mat2;
typedef std::vector<std::vector<std::vector<double>>> Mat3;

class Tensor
{
public:
    struct Size {
        int height;
        int width;
        int depth;
    };
public:
    friend std::ostream& operator<<(std::ostream& out, const Tensor& t);

    Tensor() : Tensor(0,0,0) {}
    Tensor(int height, int width, int depth);
    Tensor(const Tensor::Size& size) : Tensor(size.height, size.width, size.depth) {}
    Tensor(const Mat2& mat);
    Tensor(const Mat3& mat);

    Tensor turn180();
    void set(const Mat2& mat, int d);
    void set(const Tensor& tensor, int d);
    void copy(const Tensor& toCopy, int from_depth, int to_depth);
    Mat flatten() { return values; }

    double& operator()(int i, int j, int d);
    double operator()(int i, int j, int d) const;
    Tensor operator()(int d);
    Tensor operator+(const Mat& mat);
    Tensor operator+(double num);
    void operator+=(const Tensor& other);
    double& operator[](int index);
    const double& operator[](int index) const;

    int depth() const { return size.depth; }
    int height() const { return size.height; }
    int width() const { return size.width; }
    Tensor::Size getSize() const { return size; }
    int getRawSize() const { return values.size(); }
private:
    Size size;
    Mat values;
    int hw;
};

