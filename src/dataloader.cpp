
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
#include <future>
#include <thread>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "pipeline.hpp"
#include "random.hpp"
#include "dataloader.hpp"
#include "blocking_queue.hpp"


using std::endl;
using std::cout;
using std::vector;
using std::string;
using std::array;
using std::ifstream;
using std::stringstream;
using std::ios;
using cv::Mat;


// functions of Batch
Batch::Batch(vector<float> *dt, vector<int> dsz, vector<int64_t> *lbs, vector<int> lsz): data(dt), dsize(dsz), labels(lbs), lsize(lsz) {}

/* 
 * functions of DataLoader */
DataLoader::DataLoader(string rootpth, string fname, int bs,
        vector<int> sz, bool nchw, bool train, bool shuffle, int n_workers, 
        bool drop_last):batchsize(bs), nchw(nchw), shuffle(shuffle),
        num_workers(n_workers), drop_last(drop_last) {
    init(rootpth, fname, sz, train);
}

void DataLoader::init(string rootpth, string fname, vector<int> sz, bool train) {
    height = sz[0];
    width = sz[1];
    pos = 0;
    epoch = 0;

    // dataset.init(rootpth, fname, {height, width}, nchw);
    dataset = DataSet(rootpth, fname, {height, width}, train, nchw);
    n_all_samples = dataset.get_n_samples();
    n_samples = n_all_samples;
    all_indices.resize(n_samples);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    indices = all_indices;
    // start prefetcher as a sub-thread
    start_prefetcher();
}

DataLoader::~DataLoader() {
    stop_prefetcher();
}

Batch DataLoader::_get_batch() {
    auto t1 = std::chrono::steady_clock::now();
    CHECK (!pos_end()) << "want more samples than n_samples, there can be some logical problem\n";

    Batch spl;

    int n_batch = batchsize;
    if (pos + batchsize > n_samples) n_batch = n_samples - pos;
    int single_size = width * height * 3;
    vector<float> *data = new vector<float>(n_batch * single_size);
    vector<int64_t> *labels = new vector<int64_t>(n_batch);
    int bs_thread = n_batch / num_workers + 1;
    auto thread_func = [&](int thread_idx) {
        for (int i{0}; i < bs_thread; ++i) {
            int pos_thread = i * num_workers + thread_idx;
            if (pos_thread >= n_batch) break;
            dataset.get_one_by_idx(
                    indices[pos + pos_thread],
                    &((*data)[pos_thread * single_size]),
                    (*labels)[pos_thread]
                    );
        }
    };

    auto t2 = std::chrono::steady_clock::now();
    vector<std::future<void>> tpool(num_workers);
    for (int i{0}; i < num_workers; ++i) {
        tpool[i] = std::async(std::launch::async, thread_func, i);
    }
    for (int i{0}; i < num_workers; ++i) {
        tpool[i].get();
    }
    pos += n_batch;
    auto t3 = std::chrono::steady_clock::now();

    vector<int> dsize;
    if (nchw) {
        dsize = {n_batch, 3, height, width};
    } 
    if (!nchw){
        dsize = {n_batch, height, width, 3};
    }
    vector<int> lsize{n_batch};
    auto t4 = std::chrono::steady_clock::now();
    //
    // cout << "prepare thread_func and memory: "
    //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
    // cout << "processing: "
    //     << std::chrono::duration<double, std::milli>(t3 - t2).count() << endl;
    spl.data = data;
    spl.labels = labels;
    spl.dsize.swap(dsize);
    spl.lsize.swap(lsize);
    return spl;
}


void DataLoader::_shuffle() {
    std::shuffle(indices.begin(), indices.end(), grandom.engine);
}

void DataLoader::_start() {
    pos = 0;
    if (is_dist) {
        _split_by_rank();
    } else {
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), grandom.engine);
        }
    }
    ++epoch;
    prefetch_cond_var.notify_all();
}

void DataLoader::_restart() {
    pos = 0;
    epoch = 0;
}

bool DataLoader::_is_end() {
    bool end{false};
    if (pos_end() && data_pool.empty()) end = true;
    return end;
}

void DataLoader::_set_epoch(int ep) {
    epoch = ep;
}

void DataLoader::_init_dist(int rank, int num_ranks) {
    is_dist = true;
    this->rank = rank;
    this->num_ranks = num_ranks;
}

void DataLoader::_split_by_rank() {
    if (shuffle) {
        randeng.seed(epoch);
        std::shuffle(all_indices.begin(), all_indices.end(), randeng);
    }
    n_samples = static_cast<int>(ceil((float)n_all_samples / num_ranks));
    indices.resize(n_samples);
    int rank_pos = rank;
    for (int i{0}; i < n_samples; ++i) {
        indices[i] = all_indices[rank_pos];
        rank_pos += num_ranks;
        if (rank_pos >= n_all_samples) rank_pos = rank_pos % n_all_samples;
    }
}

int64_t DataLoader::_get_ds_length() {
    return static_cast<int64_t>(indices.size());
}

int64_t DataLoader::_get_n_batches() {
    int64_t len = n_samples / batchsize;
    if ((n_samples % batchsize != 0) && !drop_last) ++len;
    return len;
}

void DataLoader::start_prefetcher() {
    auto prefetch = [&]() {
        // cout << "prefetch enter while loop \n";
        while (true) {
            if (pos_end() && !quit_prefetch) {
                std::unique_lock<mutex> lock(prefetch_mtx);
                prefetch_cond_var.wait(lock, [&] {return quit_prefetch || !pos_end();});
            }
            if (quit_prefetch) break;

            data_pool.push(_get_batch());
        }
    };
    th_prefetch = std::async(std::launch::async, prefetch);
}


void DataLoader::stop_prefetcher() {
    quit_prefetch = true;
    prefetch_cond_var.notify_all();
    data_pool.abort();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    th_prefetch.get();
}


Batch DataLoader::_next_batch() {
    return data_pool.get();
}

bool DataLoader::pos_end() {
    bool end{false};
    if ((pos >= n_samples) || ((pos + batchsize > n_samples) && drop_last)) {
        end = true;
        // if (pos >= n_samples)
        //     cout << "pos >= n_samples" << endl;
        // if ((pos + batchsize > n_samples) && drop_last)
        //     cout << "(pos + batchsize > n_samples) && drop_last" << endl;
        // cout << "pos: " << pos << endl;
        // cout << "batchsize: " << batchsize << endl;
        // cout << "n_samples: " << n_samples << endl;
    }
    return end;
}

//
// int main () {
//     string imroot("/data/zzy/imagenet/train/");
//     string annfile("../grpc/train.txt");
//     DataLoader dl(imroot, annfile, 128, {224, 224}, true, 4);
//     for (int i{0}; i < 10; ++i) {
//         cout << dl.dataset.img_paths[i] << endl;
//     }
//     for (int i{0}; i < 10; ++i) {
//         cout << dl.indices[i] << endl;
//     }
//
//     vector<float> *batch;
//     vector<int> size;
//     cout << "run get batch: " << endl;
//     dl._get_batch(batch, size);
//     cout << "batch size: ";
//     for (auto& el : size) cout << el << ", "; cout << endl;
//     cout << batch->size() << endl;
//
//     return 0;
// }
