#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

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
        int n_samples;
        vector<int> indices;
        int pos;
        int num_workers;

        DataSet dataset;

        DataLoader() {}
        DataLoader(string rootpth, string fname, int bs,
            vector<int> sz, bool nchw, int n_workers);
        virtual ~DataLoader() {}

        void init(string rootpth, string fname, vector<int> sz);
        void _get_batch(vector<float>* &data, vector<int>& size);
        void _shuffle();
        void _restart();
        bool _is_end();
};

#endif
