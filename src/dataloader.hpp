#ifndef _DATALOADER_HPP_
#define _DATALOADER_HPP_

#include <vector>
#include <string>
#include <array>


using std::vector;
using std::string;
using std::array;


class DataLoader{
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

        DataLoader() {}
        DataLoader(string rootpth, string fname, int bs,
                vector<int> sz, bool nchw);
        virtual ~DataLoader() {}

        void init(string rootpth, string fname, int bs, 
                vector<int> sz, bool nchw);
        void get_batch(vector<float>* &data, vector<int>& size);
        void parse_annos();
};

#endif
