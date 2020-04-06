
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <array>
#include <fstream>
#include <sstream>
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
        vector<int> sz, bool nchw) {
    init(rootpth, fname, bs, sz, nchw);
}

void DataLoader::init(string rootpth, string fname, int bs,
        vector<int> sz, bool nchw) {
    n_samples = 0;
    batchsize = bs;
    imroot = rootpth;
    annfile = fname;
    height = sz[0];
    width = sz[1];
    nchw_layout = nchw;
    parse_annos();
    std::shuffle(indices.begin(), indices.end(), grandom.engine);
    pos = 0;
}

void DataLoader::get_batch(vector<float>* &data, vector<int>& size) {
    int single_size = width * height * 3;
    data = new vector<float>(batchsize * single_size);
    for (int b{0}; b < batchsize; ++b) {
        string impth = img_paths[pos];
        Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
        im = TransTrain(im, {height, width}, true);
        Mat2Mem(im, &((*data)[b * single_size]), nchw_layout);
        ++pos;
    }
    size.resize(4);
    if (nchw_layout) {
        size[0] = batchsize;size[1] = 3;
        size[2] = height;size[3] = width;
    } else {
        size[0] = batchsize;size[1] = height;
        size[2] = width;size[3] = 3;
    }
}

void DataLoader::parse_annos() {
    ifstream fin(annfile, ios::in);
    CHECK(fin) << "file does not exists: " << annfile << endl;
    stringstream ss;
    fin >> ss.rdbuf(); // std::noskipws
    CHECK(!(fin.fail() && fin.eof())) << "error when read ann file\n";
    fin.close();

    string buf;
    while (std::getline(ss, buf)) {++n_samples;};
    ss.clear();ss.seekg(0);

    img_paths.resize(n_samples);
    labels.resize(n_samples);
    indices.resize(n_samples);
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
    std::iota(indices.begin(), indices.end(), 0);
}

// int main () {
//     string imroot("/data/zzy/imagenet/train/");
//     string annfile("../grpc/train.txt");
//     DataLoader dl(imroot, annfile, 128);
//     for (int i{0}; i < 10; ++i) {
//         cout << dl.img_paths[i] << endl;
//     }
//     for (int i{0}; i < 10; ++i) {
//         cout << dl.indices[i] << endl;
//     }
//
//     vector<float> batch;
//     dl.get_batch(batch);
//     cout << "batch size: " << 128 * height * width * 3 << endl;
//     cout << batch.size() << endl;
//
//     return 0;
// }
