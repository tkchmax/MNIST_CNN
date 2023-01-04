#include "MNIST.h"
#include <fstream>
#include <iostream>
#include <utility>
#include <cassert>

namespace {
    int ReverseInt(int i)
    {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    Mat2 To2d(const Mat& num) {
        Mat2 img = Mat2(MNIST::IMG_HEIGHT, Mat(MNIST::IMG_WIDTH));
        for (int j = 0, w = 0, h = 0; j < MNIST::IMG_HEIGHT * MNIST::IMG_WIDTH; ++j) {
            img[h][w++] = num[j] / 255;
            if (w % MNIST::IMG_WIDTH == 0) {
                w = 0;
                h += 1;
            }
        }
        return img;
    }
}

Mat2 MNIST::GetImg(int label, int n) const
{
    Mat2 res;
    for (const auto& num : images) {
        if (num.first == label) {
            res.push_back(num.second);
            if (!--n) break;
        }
    }

    //Norm
    for (int i = 0; i < res.size(); ++i) {
        for (int j = 0; j < res[0].size(); ++j) {
            res[i][j] /= 255;
        }
    }

    return res;
}

std::vector<std::pair<Mat, Mat2>> MNIST::GetLabeledImages() const
{
    std::vector<std::pair<Mat, Mat2>> imgs;
    for (const auto& flat_img : images) {
        Mat label(10);
        label[flat_img.first] = 1;

        imgs.push_back(std::pair<Mat, Mat2>(label, To2d(flat_img.second)));
    }

    return imgs;
}

std::pair<MNIST::LabeledSamples, MNIST::LabeledSamples> MNIST::GetTrainTestSamples() const
{
    const int SIZE = images.size();
    const int TRAIN_SIZE = 0.8 * SIZE;
    const int TEST_SIZE = SIZE - TRAIN_SIZE;

    auto labeledImages = GetLabeledImages();
    assert(labeledImages.size() == SIZE);

    LabeledSamples train;
    for (int i = 0; i < TRAIN_SIZE; ++i) {
        train.push_back(labeledImages[i]);
    }

    LabeledSamples test;
    for (int i = TRAIN_SIZE; i < SIZE; ++i) {
        test.push_back(labeledImages[i]);
    }

    assert(test.size() == TEST_SIZE);
    assert(train.size() == TRAIN_SIZE);

    return { train, test };
}

const MNIST& MNIST::Get()
{
    static MNIST instance;
    return instance;
}

MNIST::MNIST()
{
    images = ReadMNIST();
}

MNIST::Images MNIST::ReadMNIST()
{
    std::vector<std::pair<int, std::vector<double>>> imgs;

    //Read labels
    std::ifstream labels("mnist\\t10k-labels.idx1-ubyte", std::ios::binary);
    if (labels.is_open()) {
        int magic_number = 0;
        int nLabels = 0;

        labels.read((char*)&magic_number, sizeof(magic_number));
        labels.read((char*)&nLabels, sizeof(nLabels));
        nLabels = ReverseInt(nLabels);
        imgs.resize(nLabels);
        for (int i = 0; i < nLabels; ++i) {
            int temp = 0;
            labels.read((char*)&temp, 1);
            imgs[i].first = temp;
        }
    }

    //Read images
    std::ifstream img("mnist\\t10k-images.idx3-ubyte", std::ios::binary);
    if (img.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        img.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        img.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        img.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        img.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for (int i = 0; i < number_of_images; ++i)
        {
            imgs[i].second.resize(28 * 28);
            for (int r = 0; r < n_rows; ++r)
            {
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    img.read((char*)&temp, sizeof(temp));
                    imgs[i].second[(n_rows * r) + c] = (double)temp;
                }
            }
        }
    }
    return imgs;
}
