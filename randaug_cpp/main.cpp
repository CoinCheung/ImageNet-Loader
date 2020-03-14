
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <iterator>
#include <numeric>
#include <memory>
#include <algorithm>
#include <functional>
#include <utility>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "transforms.hpp"
#include "rand_aug.hpp"
// #include "randaugment.hpp"

using cv::Mat;
using cv::MatND;
using std::string;
using std::stringstream;
using std::ios;
using std::vector;
using std::array;
using std::endl;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::unique_ptr;



void dump_bytes(Mat im) {

    // int nbytes = im.rows * im.cols * im.channels();
    // int nbytes = std::accumulate(std::begin(im.size), std::end(im.size), im.elemSize1(), std::multiply<int>());
    int nbytes = im.rows * im.cols * im.channels() * im.elemSize1();
    cout << im.size << endl;
    cout << "nbytes: " << nbytes << endl;
    ofstream fout("./res_cpp.bin", std::ios::out | std::ios::binary);
    fout.write((const char*)im.data, nbytes);
    fout.close();
}



int main() {

    string impth("../../example.png");
    // string impth("/root/139.jpg");

    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    CHECK(!im.empty()) << "read image error\n";

    Mat hist;
    for (int i{0}; i < 1; ++i) {
        // cout << i << endl;
        // hist = EqualizeFunc(im);
        // hist = AutocontrastFunc(im, 0);
        // hist = SharpnessFunc(im, 0.3, true);

        hist = ShearXFunc(im, 0.2, {128, 128, 128}, true);
    }

    dump_bytes(hist);

    // RA method 1: function object
    // array<uint8_t, 3> replace{128, 128, 128};
    // Cutout cutout(0.4, 40, {128, 128, 128}, true);
    // // hist = cutout.Apply(im);
    // vector<unique_ptr<RandApply>> ra;
    // ra.emplace_back(new Cutout(0.4, 40, {128, 128, 128}, true));
    // ra.emplace_back(new Equalize(0.4));
    // ra.emplace_back(new Autocontrast(0.4, 10));
    //
    // hist = ra[0]->Apply(im);
    // hist = ra[1]->Apply(im);
    // hist = ra[1]->Apply(im);

    // RandAug ra(2, 4);
    // hist = ra(im);
    // cv::imwrite("ra_res.jpg", hist);


    // // RA method 2: c++ std::functional
    // std::function<Mat(Mat&)> cutout_func = std::bind(CutoutFunc, std::placeholders::_1, 40, replace, true);
    // // hist = cutout_func(im);
    //
    // vector<std::function<Mat(Mat&)>> ra2;
    // ra2.push_back(cutout_func);
    // // hist = ra2[0](im);
    //
    // // hist = CutoutFunc(im, 40, replace, true);
    // cv::imwrite("ra_res.jpg", hist);

    // vector<vector<int>> sq(3, vector<int>(256));
    // std::fill(sq.begin(), sq.end(), );
    //
    // vector<int> sq(256);
    // std::iota(sq.begin(), sq.end(), 0);
    // for (auto &el : sq)
    //     cout << el << ", ";
    // cout << endl;
    //
    // for (int i{0}; i < 1; ++i) {
    //     hist = RandomResizedCrop(im, {224, 224});
    // }
    // cv::imwrite("resize_res.jpg", hist);
    //
    // cv::imwrite("orgim.jpg", im);
    // hist = RandomHorizontalFlip(im, 1);
    // cv::imwrite("hflip_res.jpg", im);
    // // cv::imwrite("hflip_res.jpg", hist);

    // hist = hwc2chw(im);
    // cout << im.size << endl;
    // cout << hist.size << endl;
    // cout << hist.type() << endl;
    // cout << hist.rows << endl;
    // cout << hist.cols << endl;
    // cout << hist.channels() << endl;
    // // int size[] = {3,4,5,6};
    // vector<int> sz{3,4,5,6};
    // Mat res(sz, CV_16U);
    // // Mat res(sz.size(), &sz[0], CV_8U);
    // cout << res.size << endl;
    // cout << im.elemSize() << endl;
    // cout << im.depth() << endl;
    // cout << res.type() << endl;
    // cout << res.depth() << endl;
    // cout << CV_8U << endl;
    // cout << CV_16U << endl;
    //
    // hist = HWC2CHW(im);
    // dump_bytes(hist);
    // cout << im.size << endl;
    // cout << im.total() << endl;
    // cout << im.channels() << endl;
    // cout << hist.size << endl;
    // cout << hist.total() << endl;
    // cout << hist.channels() << endl;
    // cout << hist.depth() << endl;
    // // vector<int> sss(hist.size.begin<int>(), hist.size.end<int>());
    // cout << hist.size << endl;

    // Mat kernel = Mat::ones(3, 3, CV_64FC1);
    // kernel.at<double>(1, 1) = 5;
    // kernel = kernel * (1. / 13);
    // cout << kernel << endl;

    // Mat kernel = Mat::ones(3, 3, CV_32FC1);
    // kernel.at<double>(1, 1) = 5;
    // kernel = kernel * (1. / 13);
    // Mat degen;
    // cv::filter2D(im, degen, -1, kernel);
    //
    // ifstream fin("./degen.bin", ios::in | ios::binary);
    // stringstream ss1;
    // fin >> std::noskipws >> ss1.rdbuf();
    // string buf1 = ss1.str();
    // degen = Mat(cv::Size(2048, 1024), CV_8UC3, &buf1.at(0));
    // fin.close();
    // // stringstream ss2;
    // // fin.open("./im.bin", ios::in | ios::binary);
    // // fin >> std::noskipws >> ss2.rdbuf();
    // // string buf2 = ss2.str();
    // // cout << buf1[8] - buf2[8] << endl;
    // // // im = Mat(cv::Size(2046, 1022), CV_8UC3, &buf2.at(0));
    // // fin.close();
    // float factor = 0.3;
    // auto roi = cv::Rect(1, 1, im.cols - 2, im.rows - 2);
    // Mat res = im.clone();
    // cv::addWeighted(degen(roi), 1. - factor, im(roi), factor, 0, res(roi));
    //
    // // cout << degen.size << endl;
    // // cout << im.size << endl;
    // // float factor = 0.3;
    // // Mat res = im.clone();
    // // auto roi = cv::Rect(1, 1, im.cols - 2, im.rows - 2);
    // // cout << im(roi).size << endl;
    // // cout << "add weighted\n";
    // // cout << degen(roi).elemSize1() << endl;
    // // cout << im(roi).elemSize1() << endl;
    // // cout << res(roi).elemSize1() << endl;
    // // cv::addWeighted(degen(roi), 1. - factor, im(roi), factor, 0, res(roi));
    // // cout << res(roi).elemSize1() << endl;
    //     // res.convertTo(res, CV_8UC3);
    //
    // dump_bytes(res);
    return 0;
}
