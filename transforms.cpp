
#include <array>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "random.hpp"
#include "transforms.hpp"


using std::cout;
using std::endl;
using std::array;
using cv::Mat;


// NOTE: it is very import to use same data types as in python. integers should all
// be int64_t and float should be double by default.
//

// 1. calhist 同时计算三个channel, 或者forEach计算hist
// 2. autocontrast 三个channel放一起是否更快
//
// 结论:
// 1. 合到一起使用forEach代替LUT会比split/merge快不少:
// 2. 使用forEach计算hist要比for loop还慢, 而且forEach是线程不安全的，可以批量修改Mat，但是不同使用Mat修改外部变量，不然要坏事。
// 3. calcHist的确能快点，但是因为要split，所以需要反而更慢了。
//

Mat RandomHorizontalFlip(Mat &im, double p, bool inplace) {
    Mat res;
    if (grandom.rand() < p) {
        if (inplace) {
            cv::flip(im, im, 1); // 0 vflip, 1 hflip, -1 hflip+vflip 
            res = im;
        } else {
            cv::flip(im, res, 1); // 0 vflip, 1 hflip, -1 hflip+vflip 
        }
    } else {
        res = im;
    }
    return res;
}


Mat RandomResizedCrop(Mat &im, array<int, 2> size, array<double, 2> scale, array<double, 2> ratio) {
    CHECK_LE(scale[0], scale[1]) << "scale should be [min, max]\n";
    CHECK_LE(ratio[0], ratio[1]) << "w / h ratio should be [min, max]\n";

    int H{im.rows}, W{im.cols};
    double area = static_cast<double>(H * W);
    int i, j, h, w;
    bool find_roi = false;

    double target_area, aspect_ratio;
    for (int iter{0}; iter < 10; ++iter) {
        target_area = grandom.rand(scale[0], scale[1]) * area;
        array<double, 2> log_ratio{std::log(ratio[0]), std::log(ratio[1])}; 
        aspect_ratio = grandom.rand(log_ratio[0], log_ratio[1]);
        aspect_ratio = std::exp(aspect_ratio);
        w = static_cast<int>(std::sqrt(target_area * aspect_ratio));
        h = static_cast<int>(std::sqrt(target_area / aspect_ratio));

        if (w < W && h < H && w > 0 && h > 0) {
            i = grandom.randint(0, W - w);
            j = grandom.randint(0, H - h);
            find_roi = true;
            break;
        }
    }
    if (!find_roi) {
        double in_ratio = (double)W / double(H);
        if (in_ratio < ratio[0]) {
            w = W;
            h = static_cast<int>(w / ratio[0]);
        } else if (in_ratio > ratio[1]) {
            h = H;
            w = static_cast<int>(h * ratio[1]);
        } else {
            w = W;
            h = H;
        }
        i = (W - w) / 2;
        j = (H - h) / 2;
    }
    // if (!find_roi)  {
    //     cout << "not find roi\n";
    //     // cout << "W, H: " << W << ", " << H << endl;
    //     // cout << "i,j,w,h: " <<i << ", " << j << ", " << w << ", " << h << endl;
    // } else cout << "find roi\n";
    // // cout << "W, H: " << i << ", "<< j << ", "<< w << ", "<< h << ", " << W << ", " << H << endl;
    // cout << "target_area: " << target_area << endl;
    // cout << "aspect ratio: " << aspect_ratio << endl;
    // cout << "i,j,w,h: " <<i << ", " << j << ", " << w << ", " << h << endl;

    Mat res = im(cv::Rect(i, j, w, h));
    cv::resize(res, res, {size[0], size[1]}, cv::INTER_CUBIC);

    return res;
}


Mat Normalize(Mat &im, array<double, 3> mean, array<double, 3> std) {
    Mat res;
    im.convertTo(res, CV_32FC3, 1. / 255);
    cv::subtract(res, cv::Scalar(mean[0], mean[1], mean[2]), res);
    cv::divide(res, cv::Scalar(std[0], std[1], std[2]), res);

    return res;
}


Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace) {
    Mat res = RandomResizedCrop(im, size);
    res = RandomHorizontalFlip(res, inplace);
    array<double, 3> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225}; 
    res = Normalize(res, mean, std);
    return res;
}


Mat HWC2CHW (Mat &im) {
    int rows = im.rows;
    int cols = im.cols;
    int chans = im.channels();
    int size[] = {chans, rows, cols};
    Mat res(3, size, im.type());
    vector<Mat> planes(chans);
    for (int c{0}; c < chans; ++c) {
        planes[c] = Mat(rows, cols, im.depth(), res.ptr<uint8_t>(c));
    }
    cv::split(im, planes);
    return res;
}


//// random aug funcs
array<int64_t, 256> calcHist1C(Mat &im);
array<array<int64_t, 256>, 3> calcHist(Mat &im);


array<array<int64_t, 256>, 3> calcHist(Mat &im) {
    CHECK_EQ(im.depth(), 0) << "pixel data type should 8 CV_8U\n"; // 0 is 8U
    CHECK_EQ(im.channels(),  3) << "image must be 3 channels\n";
    array<array<int64_t, 256>, 3> res{{}};

    // im.forEach<cv::Vec3b>([&](cv::Vec3b &pix, const int* pos) {
    //         ++res[0][pix[0]];
    //         ++res[1][pix[1]];
    //         ++res[2][pix[2]];
    // });

    // cv::parallel_for_(cv::Range(0, im.rows), [&](const cv::Range &range) {
    //     for (int r{range.start}; r < range.end; ++r) {
    //         uint8_t* ptr = im.ptr<uint8_t>(r);
    //         for (int j{0}; j < im.rows; ++j) {
    //             ++res[0][ptr[0]];
    //             ++res[1][ptr[1]];
    //             ++res[2][ptr[2]];
    //             // for (int c{0}; c < 3; ++c) {
    //             //     ++res[c][ptr[c]];
    //             // }
    //             ptr += 3;
    //         }
    //     }
    // });

    int nrows = im.rows;
    int ncols = im.cols;
    for (int i{0}; i < nrows; ++i) {
        uint8_t* ptr = im.ptr<uint8_t>(i);
        for (int j{0}; j < ncols; ++j) {
            for (int c{0}; c < 3; ++c) {
                ++res[c][ptr[c]];
            }
            ptr += 3;
        }
    }

    
    // array<Mat, 3> channels;
    // cv::split(im, channels);
    // Mat hist;
    // int size = 256;
    // float bRange[] = {0, 256};
    // const float* range[] = {bRange};
    // cv::calcHist(&channels[0], 1, 0, Mat(), hist, 1, &size, range);
    // cv::calcHist(&channels[1], 1, 0, Mat(), hist, 1, &size, range);
    // cv::calcHist(&channels[2], 1, 0, Mat(), hist, 1, &size, range);

    return res;
}


array<int64_t, 256> calcHist1C(Mat &im) {
    CHECK_EQ(im.depth(), 0) << "pixel data type should 8 CV_8U\n"; // 0 is 8U
    CHECK_EQ(im.channels(), 1) << "image must be 1 channels\n";
    array<int64_t, 256> res{};
    int nrows = im.rows;
    int ncols = im.cols;
    for (int i{0}; i < nrows; ++i) {
        uint8_t* ptr = im.ptr<uint8_t>(i);  
        for (int j{0}; j < ncols; ++j) {
            ++res[ptr[j]];
        }
    }
    return res;
}



Mat EqualizeFunc(Mat &im) {
    CHECK_EQ(im.depth(), 0) << "pixel data type should 8 CV_8U\n"; // 0 is 8U
    // array<Mat, 3> channels;
    // cv::split(im, channels);
    array<Mat, 3> outs;

    array<array<uint8_t, 256>, 3> tables;
    for (int c{0}; c < 3; ++c) std::iota(tables[c].begin(), tables[c].end(), 0);
    array<array<int64_t, 256>, 3> hists = calcHist(im);
    for (int i{0}; i < 3; ++i) {
        array<int64_t, 256> hist = hists[i];
        int step{0}, last_non_zero{0};
        for (int64_t& el : hist) {
            if (el != 0) {
                step += el;
                last_non_zero = el;
            }
        }
        step = (step - last_non_zero) / 255;
        if (step != 0) {
            array<int64_t, 256> n{};
            n[0] = step / 2;
            for (int j{1}; j < 256; ++j) {
                n[j] = n[j-1] + hist[j-1];
            }
            std::transform(n.begin(), n.end(), tables[i].begin(), [&](int64_t el){
                el /= step;
                if (el < 0) el = 0;
                if (el > 255) el = 255;
                return static_cast<uint8_t>(el);
            });
        }
        // cv::LUT(channels[i], tables[i], outs[i]);
    }
    // Mat res;
    // cv::merge(outs, res);
    
    Mat res = im.clone();
    res.forEach<cv::Vec3b>([&tables](cv::Vec3b &pix, const int* pos) {
            pix[0] = tables[0][pix[0]];
            pix[1] = tables[1][pix[1]];
            pix[2] = tables[2][pix[2]];
            });
    return res;
}


// Mat AutocontrastFunc1C(Mat &im, int cutoff) {
//     CHECK_EQ(im.depth(), 0) << "pixel data type should 8 CV_8U\n"; // 0 is 8U
//     int64_t n = im.rows * im.cols * im.elemSize();
//     int64_t cut = cutoff * n / 100;
//     int64_t high{255}, low{0};
//     array<int64_t, 256> hist = calcHist1C(im);
//     if (cut == 0) {
//         uint8_t max{im.data[0]}, min{im.data[0]};
//         // min = std::distance(hist.begin(), std::find_if(hist.begin(), hist.end(), [&](int64_t el) {return el != 0;}));
//         // min = std::distance(hist.rbegin(), std::find_if(hist.rbegin(), hist.rend(), [&](int64_t el) {return el != 0;}));
//         std::for_each(im.data, im.data + n, [&](uint8_t &el) {
//             if (el > max) max = el;
//             if (el < min) min = el;
//         });
//         high = max;
//         low = min;
//     } else {
//         array<int64_t, 256> hist = calcHist1C(im);
//         int64_t acc{0};
//         for (int i{0}; i < 256; ++i) {
//             acc += hist[i];
//             if (acc > cut) {
//                 low = i;
//                 break;
//             }
//         }
//         acc = 0;
//         for (int i{255}; i >=0; --i) {
//             acc += hist[i];
//             if (acc > cut) {
//                 high = i;
//                 break;
//             }
//         }
//     }
//     array<uint8_t, 256> table;
//     double scale{1}, offset{0};
//     if (high > low) {
//         scale = 255. / (high - low);
//         // tricky here considering types conversion
//         offset = -static_cast<uint8_t>(low) * scale;
//     }
//     for (int64_t i{0}; i < 256; ++i) {
//         double el = i * scale + offset;
//         if (el < 0) {el = 0;}
//         if (el > 255) {el = 255;}
//         table[i] = static_cast<uint8_t>(el);
//     }
//     Mat out;
//     cv::LUT(im, table, out);
//     return out;
// }
//
//
//
// Mat AutocontrastFunc(Mat &im, int cutoff) {
//     CHECK_EQ(im.channels(),  3) << "image must be 3 channels\n";
//     array<Mat, 3> channels;
//     cv::split(im, channels);
//     array<Mat, 3> outs;
//     for (int i{0}; i < 3; ++i) {
//         outs[i] = AutocontrastFunc1C(channels[i], cutoff);
//     }
//     Mat res;
//     cv::merge(outs, res);
//     return res;
// }


Mat AutocontrastFunc(Mat &im, int cutoff) {
    CHECK_EQ(im.channels(),  3) << "image must be 3 channels\n";

    array<array<int64_t, 256>, 3> hists = calcHist(im);
    array<array<uint8_t, 256>, 3> tables;

    for (int c{0}; c < 3; ++c) {
        int64_t n = im.rows * im.cols;
        int64_t cut = cutoff * n / 100;
        int64_t high{255}, low{0};
        if (cut == 0) {
            low = std::distance(hists[c].begin(), 
                    std::find_if(hists[c].begin(), hists[c].end(),
                        [&](int64_t &el) {return el != 0;}));
            high = 255 - std::distance(hists[c].rbegin(),
                    std::find_if(hists[c].rbegin(), hists[c].rend(), 
                        [&](int64_t &el) {return el != 0;}));
        } else {
            int64_t acc{0};
            for (int i{0}; i < 256; ++i) {
                acc += hists[c][i];
                if (acc > cut) {
                    low = i;break;
                }
            }
            acc = 0;
            for (int i{255}; i >=0; --i) {
                acc += hists[c][i];
                if (acc > cut) {
                    high = i;break;
                }
            }
        }

        double scale{1}, offset{0};
        if (high > low) {
            scale = 255. / (high - low);
            // tricky here considering types conversion
            offset = -static_cast<uint8_t>(low) * scale;
            for (int64_t i{0}; i < 256; ++i) {
                double el = i * scale + offset;
                if (el < 0) {el = 0;}
                if (el > 255) {el = 255;}
                tables[c][i] = static_cast<uint8_t>(el);
            }
        } else {
            std::iota(tables[c].begin(), tables[c].end(), 0);
        }
    }

    Mat res = im.clone();
    res.forEach<cv::Vec3b>([&tables] (cv::Vec3b &pix, const int* pos) {
            pix[0] = tables[0][pix[0]];
            pix[1] = tables[1][pix[1]];
            pix[2] = tables[2][pix[2]];
    });


    return res;
}



Mat CutoutFunc(Mat &im, int pad_size, array<uint8_t, 3> replace, bool inplace) {
    int H{im.rows}, W{im.cols};

    int h = static_cast<int>(grandom.rand() * H);
    int w = static_cast<int>(grandom.rand() * W);
    int x1 = std::max(0, w - pad_size);
    int x2 = std::min(W, w + pad_size);
    int y1 = std::max(0, h - pad_size);
    int y2 = std::min(H, h + pad_size);

    Mat convas;
    if (inplace) {convas = im;} else {convas = im.clone();}
    auto roi = convas(cv::Rect(x1, y1, x2 - x1, y2 - y1)); // x, y means w, h(c, r)
    roi.setTo(cv::Scalar(replace[0] , replace[1] , replace[2]));

    return convas;

}



//// class 

Mat RandApply::Apply(Mat &im) {
    Mat res = im;
    if (p <= 1 && p > 0 && grandom.rand() < p) {
        res = Func(im);
        cout << "applying: " << name << endl;
    } else {cout << "not applied by prob\n";}
    return res;
}


Mat Equalize::Func(Mat &im) {
    Mat res = EqualizeFunc(im);
    return  res;
}


Mat Autocontrast::Func(Mat &im) {
    Mat res = AutocontrastFunc(im);
    return  res;
}


Mat Cutout::Func(Mat &im) {
    Mat res = CutoutFunc(im, pad_size, replace, inplace);
    return  res;
}
