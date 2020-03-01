
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

#include "comm/interface.grpc.pb.h"
#include "comm/interface.pb.h"
#include "transforms.hpp"

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
    for (int i{0}; i < num_line; ++i) {
        ss >> buf >> labels[i];
        imgpths[i] = string(impth + "/" + buf);
    }
    fin.close();
}


Status ImageServiceImpl::get_img_by_idx(ServerContext *context,
        const comm::IdxRequest *request, 
        comm::ImgReply* reply) {

    int idx = request->num();
    // cout << "request idx is: " << idx << endl;
    string impth = imgpths[idx];

    // cout << impth << endl;
    array<int, 2> size{224, 224};
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
    reply->set_dtype("uint8");
    reply->set_label(labels[idx]);

    // cout << im(cv::Rect(0, 0, 4, 4)) << endl;

    return Status::OK;
}

