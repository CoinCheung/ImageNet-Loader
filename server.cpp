
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>
#include <mutex>
#include <future>

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
using std::mutex;
using std::cout;
using std::endl;

using grpc::Status;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;

// TODO: 
// done 1. multi-thread 
// 2. prefetch
// 3. do not use batchsize field, make train/val batch size global field
// discard 4. test the transmit speed of serialize-stream-deserialize pipline
// done 4. add run_protoc.sh commands to CMakeLists.txt
//
// 5. bidirect stream and multi thread client
// 6. fix max call problem for get_img_by_idx， use cpp client and see if there is still errors
// 7. multi thread client call get_img_by_idx
// 8. ask on github for the problem of throughput with 500M message
// 
//
//
// NOTES:
// 1. it is very slow to transmit, a 224x224x1024x3x4 large message would require us about 0.5 seconds to transmit.
// 2. it is also quite slow to allocate a 500M memory, which requires about 0.5 seconds


ImageServiceImpl::ImageServiceImpl() {
    read_annos();
    curr_pos = 0;
    size = {224, 224};
    n_workers = 32;
    batch_size = 1024;

    // debug variables
    int64_t len = 1024 * 224 * 224 * 3;
    int64_t buf_len = len * sizeof(float);
    imbuf = new string();
    imbuf->resize(buf_len);
    // TODO: release this

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
    string impth = imgpths[idx];
    cout << "receive call at idx: " << idx << endl;

    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    im = TransTrain(im, size);
    if (!im.isContinuous()) im = im.clone();

    array<int, 3> shape{im.rows, im.cols, im.channels()};
    int bytes_len = im.elemSize1();
    for (auto& el : shape) {
        bytes_len *= el;
        reply->add_shape(el);
    }
    reply->set_data(im.data, bytes_len);
    reply->set_dtype("float32");
    reply->set_label(labels[idx]);

    return Status::OK;
}


// 
Status ImageServiceImpl::get_batch_stream(ServerContext *context,
        const comm::BatchRequest *request,
        grpc::ServerWriter<comm::ImgReply>* writer) {

    auto start = std::chrono::steady_clock::now();
    cout <<"get bach called\n";
    int bs = batch_size;
    cout << "batch size: " << bs << endl;

    int64_t chunksize = size[0] * size[1] * 3 * sizeof(float);
    int64_t bufsize = chunksize * bs;
    // vector<int64_t> lbbuf(bs);

    // string *imbuf_ptr = reply->mutable_data();
    // imbuf_ptr->resize(bufsize);
    // string *imbuf = new string();
    // imbuf->resize(bufsize);

    mutex m;

    auto thread_func = [&](int thread_idx) {
        int bs_thread = (bs / n_workers) + 1;
        for (int i{0}; i < bs_thread; ++i) {
            int bs_idx = thread_idx + i * n_workers;
            int im_idx = curr_pos + bs_idx;
            if (im_idx >= n_train || bs_idx >= bs) continue;

            string impth = imgpths[im_idx];
            // lbbuf[bs_idx] = labels[im_idx];

            Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
            im = TransTrain(im, size);
            if (im.isContinuous()) im = im.clone();

            comm::ImgReply reply;
            reply.set_data(im.data, chunksize);
            reply.set_dtype("float32");
            reply.set_label(labels[im_idx]);
            reply.add_shape(1);
            reply.add_shape(224);
            reply.add_shape(224);
            reply.add_shape(3);
            {
                std::lock_guard<mutex> lock(m);
                writer->Write(reply);
                // if (!write->Write(reply)) {cout << "write failed\n";}
            }

            // std::memcpy(&imbuf->at(bs_idx * chunksize), im.data, chunksize);
            // std::memcpy(&imbuf[bs_idx * chunksize], im.data, chunksize);
            // std::memcpy(&imbuf_ptr->at(bs_idx * chunksize), im.data, chunksize);
        }
    };

    vector<std::future<void>> tpool(n_workers);
    for (int i{0}; i < n_workers; ++i) {
        tpool[i] = std::async(std::launch::async, thread_func, i);
    }
    for (int i{0}; i < n_workers; ++i) {
        tpool[i].get();
    }
    cout << "done batch process\n";
    curr_pos += bs;
    if (curr_pos > n_train) {
        cout << "one epoch done, start from beginning without shuffling\n";
        curr_pos = curr_pos % n_train;
    }

    // for (int64_t &lb : lbbuf) reply->add_labels(lb);
    // reply->add_shape(bs);
    // reply->add_shape(size[0]);
    // reply->add_shape(size[1]);
    // reply->add_shape(3);
    // reply->set_data(imbuf.data(), bufsize);
    // reply->set_dtype("float32");
    // reply->set_allocated_data(imbuf);
    // reply->release_data();

    // float *ptr = reinterpret_cast<float*>(&imbuf->at(0));
    // for (int i{0}; i < 12; ++i) {
    //     cout << ptr[i] << ", ";
    // }
    // cout << endl;

    auto end = std::chrono::steady_clock::now();
    cout << "time is: " << std::chrono::duration<double, std::milli>(end - start).count() << endl;
    return Status::OK;
}

// single thread version
// Status ImageServiceImpl::get_batch(ServerContext *context,
//         const comm::BatchRequest *request,
//         comm::BatchReply* reply) {
//
//     cout <<"get bach called\n";
//     int bs = batch_size;
//     cout << "batch size: " << bs << endl;
//
//     int64_t chunksize = size[0] * size[1] * 3 * sizeof(float);
//     int64_t bufsize = chunksize * bs;
//     vector<uint8_t> buf(bufsize, 1);
//
//     for (int i{0}; i < bs; ++i) {
//         int idx = indices[curr_pos];
//         string impth = imgpths[idx];
//         reply->add_labels(labels[idx]);
//
//         Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
//         im = TransTrain(im, size);
//
//         if (im.isContinuous()) im = im.clone();
//         std::memcpy(&buf[i * chunksize], im.data, chunksize);
//         ++curr_pos;
//         if (curr_pos > n_train) {curr_pos = curr_pos % n_train;}
//     }
//
//     reply->add_shape(bs);
//     reply->add_shape(size[0]);
//     reply->add_shape(size[1]);
//     reply->add_shape(3);
//     reply->set_data(buf.data(), bufsize);
//     reply->set_dtype("float32");
//
//     return Status::OK;
// }



void ImageServiceImpl::fill_batch(vector<float>& imbuf, vector<int64_t>& lbbuf) {
    // cout <<"get bach called\n";
    // int bs = batch_size;
    // cout << "batch size: " << bs << endl;
    //
    // int64_t chunksize = size[0] * size[1] * 3 * sizeof(float);
    // int64_t bufsize = chunksize * bs;
    // vector<uint8_t> imbuf(bufsize, 1);
    // vector<int64_t> lbbuf(bs);
    //
    // auto thread_func = [&](int thread_idx) {
    //     int bs_thread = (bs / n_workers) + 1;
    //     for (int i{0}; i < bs_thread; ++i) {
    //         int bs_idx = thread_idx + i * n_workers;
    //         int im_idx = curr_pos + bs_idx;
    //         if (im_idx >= n_train || bs_idx >= bs) continue;
    //
    //         string impth = imgpths[im_idx];
    //         lbbuf[bs_idx] = labels[im_idx];
    //
    //         Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    //         im = TransTrain(im, size);
    //         if (im.isContinuous()) im = im.clone();
    //         std::memcpy(&imbuf[bs_idx * chunksize], im.data, chunksize);
    //     }
    // };
    //
    // vector<std::future<void>> tpool(n_workers);
    // for (int i{0}; i < n_workers; ++i) {
    //     tpool[i] = std::async(std::launch::async, thread_func, i);
    // }
    // for (int i{0}; i < n_workers; ++i) {
    //     tpool[i].get();
    // }
    // cout << "done batch process\n";
    // curr_pos += bs;
    // if (curr_pos > n_train) {
    //     cout << "one epoch done, start from beginning without shuffling\n";
    //     // curr_pos = curr_pos % n_train;
    // }
}

/* 
 * // multi - thread
 * Status ImageServiceImpl::get_batch(ServerContext *context,
 *         const comm::BatchRequest *request,
 *         comm::BatchReply* reply) {
 *
 *     auto start = std::chrono::steady_clock::now();
 *     cout <<"get bach called\n";
 *     int bs = batch_size;
 *     cout << "batch size: " << bs << endl;
 *
 *     int64_t chunksize = size[0] * size[1] * 3 * sizeof(float);
 *     int64_t bufsize = chunksize * bs;
 *     // string *imbuf_ptr = reply->mutable_data();
 *     // imbuf_ptr->resize(bufsize);
 *     vector<uint8_t> imbuf(bufsize, 1);
 *     vector<int64_t> lbbuf(bs);
 *
 *     auto thread_func = [&](int thread_idx) {
 *         int bs_thread = (bs / n_workers) + 1;
 *         for (int i{0}; i < bs_thread; ++i) {
 *             int bs_idx = thread_idx + i * n_workers;
 *             int im_idx = curr_pos + bs_idx;
 *             if (im_idx >= n_train || bs_idx >= bs) continue;
 *
 *             string impth = imgpths[im_idx];
 *             lbbuf[bs_idx] = labels[im_idx];
 *
 *             Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
 *             im = TransTrain(im, size);
 *             if (im.isContinuous()) im = im.clone();
 *             std::memcpy(&imbuf[bs_idx * chunksize], im.data, chunksize);
 *             // std::memcpy(&imbuf_ptr->at(bs_idx * chunksize), im.data, chunksize);
 *         }
 *     };
 *
 *     vector<std::future<void>> tpool(n_workers);
 *     for (int i{0}; i < n_workers; ++i) {
 *         tpool[i] = std::async(std::launch::async, thread_func, i);
 *     }
 *     for (int i{0}; i < n_workers; ++i) {
 *         tpool[i].get();
 *     }
 *     cout << "done batch process\n";
 *     curr_pos += bs;
 *     if (curr_pos > n_train) {
 *         cout << "one epoch done, start from beginning without shuffling\n";
 *         curr_pos = curr_pos % n_train;
 *     }
 *
 *     for (int64_t &lb : lbbuf) reply->add_labels(lb);
 *     reply->add_shape(bs);
 *     reply->add_shape(size[0]);
 *     reply->add_shape(size[1]);
 *     reply->add_shape(3);
 *     reply->set_data(imbuf.data(), bufsize);
 *     reply->set_dtype("float32");
 *
 *     auto end = std::chrono::steady_clock::now();
 *     cout << "time is: " << std::chrono::duration<double, std::milli>(end - start).count() << endl;
 *     return Status::OK;
 * }
 *  */

// multi - thread string
Status ImageServiceImpl::get_batch(ServerContext *context,
        const comm::BatchRequest *request,
        comm::BatchReply* reply) {

    auto start = std::chrono::steady_clock::now();
    cout <<"get bach called\n";
    int bs = batch_size;
    cout << "batch size: " << bs << endl;

    int64_t chunksize = size[0] * size[1] * 3 * sizeof(float);
    int64_t bufsize = chunksize * bs;
    vector<int64_t> lbbuf(bs);

    string *imbuf_ptr = reply->mutable_data();
    imbuf_ptr->resize(bufsize);
    // string *imbuf = new string();
    // imbuf->resize(bufsize);

    auto thread_func = [&](int thread_idx) {
        int bs_thread = (bs / n_workers) + 1;
        for (int i{0}; i < bs_thread; ++i) {
            int bs_idx = thread_idx + i * n_workers;
            int im_idx = curr_pos + bs_idx;
            if (im_idx >= n_train || bs_idx >= bs) continue;

            string impth = imgpths[im_idx];
            lbbuf[bs_idx] = labels[im_idx];

            Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
            im = TransTrain(im, size);
            if (im.isContinuous()) im = im.clone();
            // std::memcpy(&imbuf->at(bs_idx * chunksize), im.data, chunksize);
            // std::memcpy(&imbuf[bs_idx * chunksize], im.data, chunksize);
            std::memcpy(&imbuf_ptr->at(bs_idx * chunksize), im.data, chunksize);
        }
    };

    vector<std::future<void>> tpool(n_workers);
    for (int i{0}; i < n_workers; ++i) {
        tpool[i] = std::async(std::launch::async, thread_func, i);
    }
    for (int i{0}; i < n_workers; ++i) {
        tpool[i].get();
    }
    cout << "done batch process\n";
    curr_pos += bs;
    if (curr_pos > n_train) {
        cout << "one epoch done, start from beginning without shuffling\n";
        curr_pos = curr_pos % n_train;
    }

    for (int64_t &lb : lbbuf) reply->add_labels(lb);
    reply->add_shape(bs);
    reply->add_shape(size[0]);
    reply->add_shape(size[1]);
    reply->add_shape(3);
    // reply->set_data(imbuf.data(), bufsize);
    reply->set_dtype("float32");
    // reply->set_allocated_data(imbuf);
    // reply->release_data();

    // float *ptr = reinterpret_cast<float*>(&imbuf->at(0));
    // for (int i{0}; i < 12; ++i) {
    //     cout << ptr[i] << ", ";
    // }
    // cout << endl;

    auto end = std::chrono::steady_clock::now();
    cout << "time is: " << std::chrono::duration<double, std::milli>(end - start).count() << endl;
    return Status::OK;
}


// Status ImageServiceImpl::get_batch(ServerContext *context,
//         const comm::BatchRequest *request,
//         comm::BatchReply* reply) {
//
//     // int64_t len = 1024 * 224 * 224;
//     // buf_len = len * sizeof(float);
//     // imbuf.resize(buf_len);
//     int64_t buf_len = 1024 * 224 * 224 * 3 * sizeof(float);
//     // vector<float> imbuf(buf_len);
//     auto start = std::chrono::steady_clock::now();
//
//     reply->add_labels(1);
//     reply->add_shape(1);
//     reply->add_shape(1);
//     reply->add_shape(1);
//     reply->add_shape(1);
//     reply->set_dtype("float32");
//     // reply->set_data(imbuf.data(), buf_len);
//     string *sptr = reply->mutable_data();
//     sptr->resize(buf_len);
//     // memset(&(sptr->at(0)), 1, buf_len / 4);
//
//     auto end = std::chrono::steady_clock::now();
//     cout << "time is: " << std::chrono::duration<double, std::milli>(end - start).count() << endl;
//     return Status::OK;
// }
