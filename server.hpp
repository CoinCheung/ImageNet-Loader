#ifndef _SERVER_HPP_
#define _SERVER_HPP_


#include <vector>
#include <string>

#include <grpcpp/grpcpp.h>

#include "comm/interface.grpc.pb.h"
#include "comm/interface.pb.h"


using std::string;
using std::vector;

using grpc::Status;
using grpc::ServerBuilder;
using grpc::ServerContext;


class ImageServiceImpl final: public comm::ImageService::Service {
    public:
        vector<string> imgpths;
        vector<int64_t> labels;
        vector<int> indices;
        array<int, 2> size;
        int curr_pos;
        int n_train;
        int n_workers;
        int batch_size;

        ImageServiceImpl();
        Status get_img_by_idx(ServerContext *context,
                const comm::IdxRequest *request, 
                comm::ImgReply* reply);
        Status get_batch(ServerContext *context,
                const comm::BatchRequest *request, 
                comm::BatchReply* reply);

        void read_annos();
};

#endif
