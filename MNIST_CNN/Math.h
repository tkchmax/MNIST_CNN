#pragma once
#include <iostream>
#include <cmath>
#include "Tensor.h"

typedef std::vector<double> Mat;
typedef std::vector<std::vector<double>> Mat2;
typedef std::vector<std::vector<std::vector<double>>> Mat3;

enum class EActivation {
    ReLU,
    SIGMOID
};

std::ostream& operator<<(std::ostream& out, const Mat& mat);
std::ostream& operator<<(std::ostream& out, const Mat2& mat);
Mat operator+(const Mat& m1, const Mat& m2);
Mat2 operator+(const Mat2& m2, double n);
void operator+=(Mat2& m3, const Mat2& m2);

Tensor Conv(const Tensor& img, const Tensor& kernel, int stride, int padding);
Mat Flatten(const Tensor& tensor);
double Sigmoid(double x);
double SigmoidDeriv(double x);
double ReLU(double x);
double ReluDeriv(double x);
double Rand(int minus = false);
double NRand(double mean, double stddev);
int ArgMax(const Mat& arr);
