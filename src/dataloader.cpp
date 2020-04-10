
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <array>
#include <fstream>
#include <sstream>
#include <future>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "pipeline.hpp"
#include "random.hpp"
#include "dataloader.hpp"


using std::endl;
using std::cout;
using std::vector;
using std::string;
using std::array;
using std::ifstream;
using std::stringstream;
using std::ios;
using cv::Mat;



DataLoader::DataLoader(string rootpth, string fname, int bs,
        vector<int> sz, bool nchw, int n_workers): batchsize(bs),
        nchw(nchw), num_workers(n_workers) {
    init(rootpth, fname, sz);
}

void DataLoader::init(string rootpth, string fname, vector<int> sz) {
    height = sz[0];
    width = sz[1];
    pos = 0;

    // dataset.init(rootpth, fname, {height, width}, nchw);
    dataset = DataSet(rootpth, fname, {height, width}, nchw);
    n_samples = dataset.get_n_samples();
    indices.resize(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
}

// TODO: consider drop last 
void DataLoader::_get_batch(vector<float>* &data, vector<int>& size) {
    CHECK_LE(pos + batchsize, n_samples) << "want more samples than n_samples, there can be some logical problem\n";
    int single_size = width * height * 3;
    data = new vector<float>(batchsize * single_size);
    for (int b{0}; b < batchsize; ++b) {
        dataset.get_one_by_idx(indices[pos], &((*data)[b * single_size]));
        ++pos;
    }
    if (nchw) {
        size = {batchsize, 3, height, width};
    } else {
        size = {batchsize, height, width, 3};
    }
}

// void DataLoader::_get_batch(vector<float>* &data, vector<int>& size) {
//     CHECK_LE(pos + batchsize, n_samples) << "want more samples than n_samples, there can be some logical problem\n";
//     int single_size = width * height * 3;
//     data = new vector<float>(batchsize * single_size);
//     for (int b{0}; b < batchsize; ++b) {
//         string impth = img_paths[pos];
//         Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
//         im = TransTrain(im, {height, width}, true);
//         Mat2Mem(im, &((*data)[b * single_size]), nchw_layout);
//         ++pos;
//     }
//     size.resize(4);
//     if (nchw_layout) {
//         size[0] = batchsize;size[1] = 3;
//         size[2] = height;size[3] = width;
//     } else {
//         size[0] = batchsize;size[1] = height;
//         size[2] = width;size[3] = 3;
//     }
// }


void DataLoader::_shuffle() {
    std::shuffle(indices.begin(), indices.end(), grandom.engine);
}

void DataLoader::_restart() {
    pos = 0;
}

bool DataLoader::_is_end() {
    if (pos + batchsize > n_samples) return true;
    else return false;
}

int main () {
    string imroot("/data/zzy/imagenet/train/");
    string annfile("../grpc/train.txt");
    DataLoader dl(imroot, annfile, 128, {224, 224}, true, 4);
    for (int i{0}; i < 10; ++i) {
        cout << dl.dataset.img_paths[i] << endl;
    }
    for (int i{0}; i < 10; ++i) {
        cout << dl.indices[i] << endl;
    }

    vector<float> *batch;
    vector<int> size;
    cout << "run get batch: " << endl;
    dl._get_batch(batch, size);
    cout << "batch size: "; 
    for (auto& el : size) cout << el << ", "; cout << endl;
    cout << batch->size() << endl;

    return 0;
}
