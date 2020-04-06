
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "pipeline.hpp"
#include "transforms.hpp"
#include "rand_aug.hpp"


using std::endl;
using std::cout;
using std::string;
using std::array;
using std::vector;
using cv::Mat;

const int height = 224;
const int width = 224;


void LoadTrainImgByPath(string impth, vector<float>* &res, vector<int> &size, bool CHW) {
    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    CHECK(!im.empty()) << "read image error\n";
    im = TransTrain(im, {height, width}, true);
    Mat2Vec(im, res, size, CHW);
}


// TODO: move all the hyperparameters to this function
Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace) {
    array<double, 3> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
    RandAug ra(2, 9);

    Mat mat = RandomResizedCrop(im, size);
    mat = RandomHorizontalFlip(mat, inplace);
    mat = ra(mat);
    mat = Normalize(mat, mean, std);
    return mat;
}


void Mat2Mem (Mat &im, float* res, bool CHW) {
    CHECK(res != nullptr) << "res should not be nullptr\n";
    int row_size = im.cols * 3;
    int chunk_size = row_size * sizeof(float);
    if (CHW) {
        int plane_size = im.rows * im.cols;
        for (int h{0}; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_w = 0;
            for (int w{0}; w < im.cols; ++w) {
                for (int c{0}; c < 3; ++c) {
                    int offset = c * plane_size + h * im.cols + w;
                    res[offset] = ptr[offset_w];
                    ++offset_w;
                }
            }
        }
    } else {
        for (int h{0}; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_res = row_size * h;
            memcpy((void*)(res+offset_res), (void*)ptr, chunk_size);
        }
    }
}

void Mat2Vec (Mat &im, vector<float>* &res, vector<int>& size, bool CHW) {
    CHECK_EQ(im.type(), CV_32FC3) << "im data type must be float32 and has 3 channels\n";
    int row_size = im.cols * 3;
    int chunk_size = row_size * sizeof(float);
    int res_size = im.rows * row_size;
    if (res == nullptr) {
        res = new vector<float>(res_size);
    } else {
        res->resize(res_size);
    }

    size.resize(3);
    if (CHW) {
        size[0] = 3;
        size[1] = im.rows;
        size[2] = im.cols;
        int plane_size = im.rows * im.cols;
        for (int h{0}; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_w = 0;
            for (int w{0}; w < im.cols; ++w) {
                for (int c{0}; c < 3; ++c) {
                    int offset = c * plane_size + h * im.cols + w;
                    (*res)[offset] = ptr[offset_w];
                    offset_w += 1;
                }
            }
        }
    } else {
        size[0] = im.rows;
        size[1] = im.cols;
        size[2] = 3;
        for (int h{0}; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_res = row_size * h;
            memcpy((void*)(&(*res)[offset_res]), (void*)ptr, chunk_size);
        }
    }
}
