
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "pipeline.hpp"
#include "transforms.hpp"
#include "rand_aug.hpp"


using std::endl;
using std::cout;
using std::string;
using std::stringstream;
using std::ifstream;
using std::ios;
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


Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace) {
    array<double, 3> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
    RandAug ra(2, 9);

    Mat mat = RandomResizedCrop(im, size);
    mat = RandomHorizontalFlip(mat, inplace);
    mat = ra(mat);
    mat = Normalize(mat, mean, std);
    return mat;
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



// member function of TransformTrain
// TODO: write it as register ops
DataSet::DataSet(string rootpth, string fname, array<int, 2> size, bool nchw, int ra_n, int ra_m, bool inplace): size(size), inplace(inplace), nchw(nchw) {
    parse_annos(rootpth, fname);
    ra = RandAug(ra_n, ra_m);
}


Mat DataSet::TransTrain(Mat& im) {
    array<double, 3> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
    Mat res;
    if (inplace) res = im;
    res = RandomResizedCrop(im, size);
    res = RandomHorizontalFlip(res, inplace);
    res = ra(res);
    res = Normalize(res, mean, std);
    return res;
}


void DataSet::get_one_by_idx(int idx, float* data) {
    CHECK(data != nullptr) << "memory not allocated, implement error\n";
    string impth = img_paths[idx];
    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    im = TransTrain(im);
    Mat2Mem(im, data);
}


void DataSet::Mat2Mem (Mat &im, float* res) {
    CHECK(res != nullptr) << "res should not be nullptr\n";
    int row_size = im.cols * 3;
    int chunk_size = row_size * sizeof(float);
    if (nchw) {
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

void DataSet::parse_annos(string imroot, string annfile) {
    ifstream fin(annfile, ios::in);
    CHECK(fin) << "file does not exists: " << annfile << endl;
    stringstream ss;
    fin >> ss.rdbuf(); // std::noskipws
    CHECK(!(fin.fail() && fin.eof())) << "error when read ann file\n";
    fin.close();

    n_samples = 0;
    string buf;
    while (std::getline(ss, buf)) {++n_samples;};
    ss.clear();ss.seekg(0);

    img_paths.resize(n_samples);
    labels.resize(n_samples);
    int tmp = 0;
    if (imroot[imroot.size()-1] == '/') {tmp = 1;}
    for (int i{0}; i < n_samples; ++i) {
        ss >> buf >> labels[i];
        int num_split = tmp;
        if (buf[0] == '/') ++num_split;
        if (num_split == 0) {
            img_paths[i] = imroot + "/" + buf;
        } else if (num_split == 1) {
            img_paths[i] = imroot + buf;
        } else {
            img_paths[i] = imroot + buf.substr(1);
        }
    }
}

int DataSet::get_n_samples() {
    return n_samples;
}
