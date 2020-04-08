#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

#include <vector>
#include <string>
#include <array>


using std::vector;
using std::string;
using std::array;


class DataLoader {
    public:
        int batchsize;
        string imroot;
        string annfile;
        int height;
        int width;
        bool nchw_layout;
        int n_samples;
        vector<string> img_paths;
        vector<int> labels;
        vector<int> indices;
        int pos;
        int num_workers;

        DataLoader() {}
        DataLoader(string rootpth, string fname, int bs,
                vector<int> sz, bool nchw, int n_workers);
        virtual ~DataLoader() {}

        void init(string rootpth, string fname, int bs, 
                vector<int> sz, bool nchw, int n_workers);
        void _get_batch(vector<float>* &data, vector<int>& size);
        void parse_annos();
        void _shuffle();
        void _restart();
        bool _is_end();
};

#endif
