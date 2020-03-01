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
        vector<int> labels;

        ImageServiceImpl();
        Status get_img_by_idx(ServerContext *context,
                const comm::IdxRequest *request, 
                comm::ImgReply* reply);

        void read_annos();
};

#endif
