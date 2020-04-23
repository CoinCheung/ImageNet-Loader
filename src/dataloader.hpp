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


using std::vector;
using std::mutex;
using std::string;
using std::condition_variable;
using std::array;


// TODO: rename as batch
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


class DataLoader {
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
        BlockingQueue<Batch> data_pool;
        int prefetch_limit{4};
        std::atomic<bool> quit_prefetch{false};
        condition_variable prefetch_cond_var;
        mutex prefetch_mtx;


        DataSet dataset;

        DataLoader() {}
        DataLoader(string rootpth, string fname, int bs,
            vector<int> sz, bool nchw, bool train, bool shuffle, int n_workers,
            bool drop_last);
        virtual ~DataLoader();

        void init(string rootpth, string fname, vector<int> sz, bool is_train);
        void _get_batch(vector<float>* &data, vector<int>& size, vector<int64_t>* &labels);
        Batch _get_batch();
        void _shuffle();
        void _start();
        void _restart();
        bool _is_end();
        void _set_epoch(int ep);
        void _init_dist(int rank, int num_ranks);
        void _split_by_rank();
        int64_t _get_ds_length();
        int64_t _get_n_batches();
        void start_prefetcher();
        void stop_prefetcher();
        Batch _next_batch();
        bool pos_end();
};


#endif
