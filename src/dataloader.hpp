#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

#include <random>
#include <vector>
#include <string>
#include <array>

#include "pipeline.hpp"


using std::vector;
using std::string;
using std::array;


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
        std::mt19937 randeng;


        DataSet dataset;

        DataLoader() {}
        DataLoader(string rootpth, string fname, int bs,
            vector<int> sz, bool nchw, bool train, bool shuffle, int n_workers,
            bool drop_last);
        virtual ~DataLoader() {}

        void init(string rootpth, string fname, vector<int> sz, bool is_train);
        void _get_batch(vector<float>* &data, vector<int>& size, vector<int64_t>* &labels);
        void _shuffle();
        void _start();
        void _restart();
        bool _is_end();
        void _set_epoch(int ep);
        void _init_dist(int rank, int num_ranks);
        void _split_by_rank();
        int _get_length();
};

#endif
