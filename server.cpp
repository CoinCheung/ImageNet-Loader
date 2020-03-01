
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

#include "comm/interface.grpc.pb.h"
#include "comm/interface.pb.h"
#include "transforms.hpp"
#include "random.hpp"

#include "server.hpp"

using std::string;
using std::vector;
using std::array;
using std::ios;
using std::ifstream;
using std::stringstream;
using std::cout;
using std::endl;

using grpc::Status;
using grpc::ServerBuilder;
using grpc::ServerContext;



ImageServiceImpl::ImageServiceImpl() {
    read_annos();
    curr_pos = 0;
    size = {224, 224};
}


void ImageServiceImpl::read_annos() {
    string trainpth("../train.txt");
    string impth("/data1/zzy/.datasets/imagenet/train/");
    ifstream fin(trainpth, ios::in);
    stringstream ss;
    fin >> ss.rdbuf();
    CHECK(!(fin.fail() && !fin.eof())) << "error read file\n";

    string buf;
    int num_line{0};
    while (std::getline(ss, buf)) {++num_line;}
    ss.clear();ss.seekg(0);

    imgpths.resize(num_line);
    labels.resize(num_line);
    indices.resize(num_line);
    for (int i{0}; i < num_line; ++i) {
        ss >> buf >> labels[i];
        imgpths[i] = string(impth + "/" + buf);
    }
    fin.close();

    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), grandom.engine);

    n_train = num_line;
}


Status ImageServiceImpl::get_img_by_idx(ServerContext *context,
        const comm::IdxRequest *request, 
        comm::ImgReply* reply) {

    int idx = request->num();
    // cout << "request idx is: " << idx << endl;
    string impth = imgpths[idx];

    // cout << impth << endl;
    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    im = TransTrain(im, size);
    if (!im.isContinuous()) im = im.clone();
    // cout << im.size << endl;

    array<int, 3> shape{im.rows, im.cols, im.channels()};
    int bytes_len = im.elemSize1();
    for (auto& el : shape) {
        bytes_len *= el;
        reply->add_shape(el);
    }
    // cout << "bytes_len: " << bytes_len << endl;
    reply->set_data(im.data, bytes_len);
    reply->set_dtype("float32");
    reply->set_label(labels[idx]);

    // cout << im(cv::Rect(0, 0, 4, 4)) << endl;

    return Status::OK;
}


Status ImageServiceImpl::get_batch(ServerContext *context,
        const comm::BatchRequest *request, 
        comm::BatchReply* reply) {

    cout <<"get bach called\n";
    int bs = request->batchsize();

    int64_t chunksize = size[0] * size[1] * 3 * 4;
    int64_t bufsize = chunksize * bs;
    vector<uint8_t> buf(bufsize);

    for (int i{0}; i < bs; ++i) {
        int idx = indices[curr_pos];
        string impth = imgpths[idx];
        reply->add_labels(labels[idx]);

        // TODO: find a method to check this
        Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
        im = TransTrain(im, size);
        if (!im.isContinuous()) im = im.clone();
        std::memcpy(im.data, &buf[0] + chunksize, chunksize);
        ++curr_pos;
        if (curr_pos > n_train) {curr_pos = curr_pos % n_train;}
        cout << idx << ", ";
    }
    cout << endl;

    reply->add_shape(bs);
    reply->add_shape(size[0]);
    reply->add_shape(size[1]);
    reply->add_shape(3);
    reply->set_data(&buf[0], bufsize);
    reply->set_dtype("float32");

    return Status::OK;
}
