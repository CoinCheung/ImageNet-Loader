#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

#include <random>
#include <vector>
#include <string>
#include <array>
#include <thread>
#include <future>
#include <atomic>
#include <condition_variable>
#include <mutex>

#include "pipeline.hpp"
#include "blocking_queue.hpp"
#include "thread_pool.hpp"


using std::vector;
using std::mutex;
using std::string;
using std::condition_variable;
using std::array;


class Batch {
    public: 
        vector<float> *data{nullptr};
        vector<int> dsize;
        vector<int64_t> *labels{nullptr};
        vector<int> lsize;

        Batch()=default;
        Batch(vector<float> *dt, vector<int> dsz, vector<int64_t> *lbs, vector<int> lsz);
        Batch(const Batch& spl)=default;
        Batch(Batch&& spl)=default;
        Batch& operator=(const Batch& spl)=default;
        Batch& operator=(Batch&& spl)=default;
};


template<typename T>
class BaseDataLoader {
    public:
        int batchsize;
        string imroot;
        int height;
        int width;
        bool nchw;
        bool shuffle;
        int n_samples;
        int n_all_samples;
        vector<int> indices;
        vector<int> all_indices;
        int pos;
        int num_workers;
        bool drop_last{true};
        bool is_dist{false};
        int rank;
        int num_ranks;
        int epoch{0};
        std::mt19937 randeng; // only for dist sampler
        std::future<void> th_prefetch;
        BlockingQueue<T> data_pool;
        int prefetch_limit{4};
        std::atomic<bool> quit_prefetch{false};
        condition_variable prefetch_cond_var;
        mutex prefetch_mtx;

        ThreadPool thread_pool;
        DataSet dataset;

        BaseDataLoader() {}
        BaseDataLoader(string rootpth, string fname, int bs,
            vector<int> sz, bool nchw, bool shuffle, int n_workers,
            bool drop_last);
        virtual ~BaseDataLoader();

        void init(string rootpth, string fname, vector<int> sz);
        void start_prefetcher();
        void stop_prefetcher();
        bool pos_end();

        virtual T _get_batch()=0; // this should be overloaded

        void _shuffle();
        void _start();
        void _restart();
        bool _is_end();
        void _set_epoch(int ep);
        void _init_dist(int rank, int num_ranks);
        void _split_by_rank();
        int64_t _get_ds_length();
        int64_t _get_num_batches();
        T _next_batch();
};

template<typename T>
BaseDataLoader<T>::BaseDataLoader(string rootpth, string fname, int bs,
        vector<int> sz, bool nchw, bool shuffle, int n_workers, 
        bool drop_last): batchsize(bs), nchw(nchw), shuffle(shuffle),
        num_workers(n_workers), drop_last(drop_last) {
    init(rootpth, fname, sz);
}

template<typename T>
void BaseDataLoader<T>::init(string rootpth, string fname, vector<int> sz) {
    height = sz[0];
    width = sz[1];
    pos = 0;
    epoch = 0;

    // dataset.init(rootpth, fname, {height, width}, nchw);
    dataset = DataSet(rootpth, fname, {height, width}, nchw);
    n_all_samples = dataset.get_n_samples();
    n_samples = n_all_samples;
    all_indices.resize(n_samples);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    indices = all_indices;

    thread_pool.init(1024, num_workers);
}

template<typename T>
BaseDataLoader<T>::~BaseDataLoader() {
    stop_prefetcher();
}


template<typename T>
void BaseDataLoader<T>::_shuffle() {
    std::shuffle(indices.begin(), indices.end(), grandom.engine);
}

template<typename T>
void BaseDataLoader<T>::_start() {
    pos = 0;
    if (is_dist) {
        _split_by_rank();
    } else {
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), grandom.engine);
        }
    }
    start_prefetcher();
    ++epoch;
}


template<typename T>
void BaseDataLoader<T>::_restart() {
    pos = 0;
    if (is_dist) {
        _split_by_rank();
    } else {
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), grandom.engine);
        }
    }
    prefetch_cond_var.notify_all();
}

template<typename T>
bool BaseDataLoader<T>::_is_end() {
    bool end{false};
    if (pos_end() && data_pool.empty()) end = true;
    return end;
}

template<typename T>
void BaseDataLoader<T>::_set_epoch(int ep) {
    epoch = ep;
}

template<typename T>
void BaseDataLoader<T>::_init_dist(int rank, int num_ranks) {
    is_dist = true;
    this->rank = rank;
    this->num_ranks = num_ranks;
    this->n_samples = static_cast<int>(ceil((float)n_all_samples / num_ranks));
    indices.resize(this->n_samples);
}


template<typename T>
void BaseDataLoader<T>::_split_by_rank() {
    if (shuffle) {
        randeng.seed(epoch);
        std::shuffle(all_indices.begin(), all_indices.end(), randeng);
    }
    int rank_pos = rank;
    for (int i{0}; i < n_samples; ++i) {
        indices[i] = all_indices[rank_pos];
        rank_pos += num_ranks;
        if (rank_pos >= n_all_samples) rank_pos = rank_pos % n_all_samples;
    }
}

template<typename T>
int64_t BaseDataLoader<T>::_get_ds_length() {
    return static_cast<int64_t>(indices.size());
}

template<typename T>
int64_t BaseDataLoader<T>::_get_num_batches() {
    int64_t len = n_samples / batchsize;
    if ((n_samples % batchsize != 0) && !drop_last) ++len;
    return len;
}

template<typename T>
void BaseDataLoader<T>::start_prefetcher() {
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

template<typename T>
void BaseDataLoader<T>::stop_prefetcher() {
    quit_prefetch = true;
    prefetch_cond_var.notify_all();
    data_pool.abort();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    th_prefetch.get();
}

template<typename T>
bool BaseDataLoader<T>::pos_end() {
    bool end{false};
    if ((pos >= n_samples) || ((pos + batchsize > n_samples) && drop_last)) {
        end = true;
    }
    return end;
}


template<typename T>
T BaseDataLoader<T>::_next_batch() {
    return data_pool.get();
}

/* 
 * definition of DataLoaderNp
 *  */
class DataLoaderNp: public BaseDataLoader<Batch> {
public:
    DataLoaderNp(string rootpth, string fname, int bs,
        vector<int> sz, bool nchw, bool shuffle, int n_workers, 
        bool drop_last): BaseDataLoader<Batch>(rootpth, fname, bs,
        sz, nchw, shuffle, n_workers, drop_last) {}
    Batch _get_batch();
};


#endif
