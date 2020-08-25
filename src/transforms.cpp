
#include <array>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstring>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "random.hpp"
#include "transforms.hpp"


using std::cout;
using std::endl;
using std::stringstream;
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
    } else {res = im;}
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

    Mat res = im(cv::Rect(i, j, w, h));
    cv::resize(res, res, {size[0], size[1]}, cv::INTER_CUBIC);

    return res;
}


Mat ResizeCenterCrop(Mat &im, array<int, 2> size) {
    int H{im.rows}, W{im.cols};
    int th{size[0]}, tw{size[1]};
    int w, h;
    if (W < H) {
        // w = tw + 32;
        w = static_cast<int>(tw * (256. / 224.));
        h = static_cast<int>(w * H / W);
    } else {
        // h = th + 32;
        h = static_cast<int>(th * (256. / 224.));
        w = static_cast<int>(h * W / H);
    }
    Mat res;
    cv::resize(im, res, {w, h}, cv::INTER_CUBIC);
    int i{(w - tw) >> 1}, j{(h - th) >> 1};
    res = res(cv::Rect(i, j, tw, th));
    return res;
}


Mat Normalize(Mat &im, array<double, 3> mean, array<double, 3> std) {
    Mat res = im;
    // im.convertTo(res, CV_32FC3, 1. / 255);
    // cv::subtract(res, cv::Scalar(mean[0], mean[1], mean[2]), res);

    // cv::subtract(im, cv::Scalar(mean[0], mean[1], mean[2]), res);
    // cv::divide(res, cv::Scalar(std[0], std[1], std[2]), res);

    array<float, 3> m, s;
    m[0] = static_cast<float>(mean[0]);
    m[1] = static_cast<float>(mean[1]);
    m[2] = static_cast<float>(mean[2]);

    s[0] = static_cast<float>(1. / std[0]);
    s[1] = static_cast<float>(1. / std[1]);
    s[2] = static_cast<float>(1. / std[2]);

    res.forEach<cv::Vec3f>([&] (cv::Vec3f &pix, const int* pos) {
        pix[0] = (pix[0] - m[0]) * s[0];
        pix[1] = (pix[1] - m[1]) * s[1];
        pix[2] = (pix[2] - m[2]) * s[2];
    });

    return res;
}


void Normalize(Mat &im, array<float, 3> mean, array<float, 3> std, float* p_res, bool pca_noise, double pca_std, bool nchw) {
    /* merge 1/255, pca-noise, mean/var and layout change operations together, so see if this can be faster */

    // for pca noise
    vector<float> rgb(3, 0);
    if (pca_noise) {
        vector<float> alpha(3);
        std::generate(alpha.begin(), alpha.end(), [&](){
                return static_cast<float>(grandom.normal(0., pca_std));});
        vector<vector<float>> eig_vec{
            {-0.5675f, 0.7192f, 0.4009f},
            {-0.5808f, -0.0045f, -0.8140f}, 
            {-0.5836f, -0.6948f, 0.4203f}};
        vector<float> eig_val{0.2175f, 0.0188f, 0.0045f};
        for (int i{0}; i < 3; ++i) {
            rgb[0] += eig_vec[0][i] * alpha[i] * eig_val[i];
            rgb[1] += eig_vec[1][i] * alpha[i] * eig_val[i];
            rgb[2] += eig_vec[2][i] * alpha[i] * eig_val[i];
        }
    }

    // for mean/var
    for (int i{0}; i < 3; ++ i) {
        std[i] = 1.f / std[i];
    }

    float scale = static_cast<float>(1. / 255.);
    im.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
        for (int i{0}; i < 3; ++i) {
            float tmp = static_cast<float>(pix[i]) * scale; 
            tmp += rgb[2 - i];
            tmp = (tmp - mean[i]) * std[i];
            int offset;
            if (nchw) {
                offset = i * im.rows * im.cols + pos[0] * im.cols + pos[1];
            } else {
                offset = pos[0] * im.cols * 3 + pos[1] * 3 + i;
            }
            p_res[offset] = tmp;
        }
    });
}

Mat AddPCANoise(Mat &im, double std, bool inplace) {
    vector<float> alpha(3);
    std::generate(alpha.begin(), alpha.end(), [&](){
            return static_cast<float>(grandom.normal(0., std));});
    vector<vector<float>> eig_vec{
        {-0.5675f, 0.7192f, 0.4009f},
        {-0.5808f, -0.0045f, -0.8140f}, 
        {-0.5836f, -0.6948f, 0.4203f}};
    vector<float> eig_val{0.2175f, 0.0188f, 0.0045f};
    vector<float> rgb(3, 0);
    for (int i{0}; i < 3; ++i) {
        rgb[0] += eig_vec[0][i] * alpha[i] * eig_val[i];
        rgb[1] += eig_vec[1][i] * alpha[i] * eig_val[i];
        rgb[2] += eig_vec[2][i] * alpha[i] * eig_val[i];
    }

    Mat res;
    if (inplace) res = im; else res = im.clone();
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
        pix[0] += rgb[2];
        pix[1] += rgb[1];
        pix[2] += rgb[0];
    });
    return im;
}


Mat ColorJitter (
        Mat &im, double brightness, double contrast, 
        double saturation, bool inplace) {
    brightness = grandom.rand(std::max(0., 1. - brightness), 1. + brightness);
    contrast = grandom.rand(std::max(0., 1. - contrast), 1. + contrast);
    saturation = grandom.rand(std::max(0., 1. - saturation), 1. + saturation);

    Mat res;
    vector<int> order{0, 1, 2};
    std::shuffle(order.begin(), order.end(), grandom.engine);
    for (int i{0}; i < 3; ++i) {
        if (order[i] == 0) {
            res = BrightnessFunc(im, brightness, inplace);
        } else if (order[i] == 1) {
            res = ContrastFunc(res, contrast, inplace);
        } else if (order[i] == 2) {
            res = ColorFunc(res, saturation, inplace);
        }
    }
    return res;
}


// vector<float> HWC2CHW (Mat &im) {
//     CHECK (im.depth() == CV_32F || im.depth() == CV_64F)
//         << "image type must be float32 or float64\n";
//     Mat res;
//     if (im.depth() == CV_64F) {
//         im.convertTo(res, CV_32FC3, 1.);
//     } else {res = im;}
//     int size = res.size[0] * res.size[1] * res.elemSize();
//
//     vector<float> out(size);
// }


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
    }
    
    Mat res = im.clone();
    res.forEach<cv::Vec3b>([&tables](cv::Vec3b &pix, const int* pos) {
            pix[0] = tables[0][pix[0]];
            pix[1] = tables[1][pix[1]];
            pix[2] = tables[2][pix[2]];
            });
    return res;
}


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


inline Mat WarpAffine(Mat &im, Mat &M, array<uint8_t, 3> replace, bool inplace) {
    Mat res;
    if (inplace) res = im;
    cv::warpAffine(im, res, M, im.size(), 
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, 
            cv::Scalar(replace[0], replace[1], replace[2]));
    return res;
}


Mat RotateFunc(Mat &im, float degree, array<uint8_t, 3> replace, bool inplace) {
    cv::Point2f center(im.cols / 2., im.rows / 2.);
    Mat M = cv::getRotationMatrix2D(center, degree, 1.);
    return WarpAffine(im, M, replace, inplace);
    // Mat res;
    // if (inplace) res = im;
    // cv::warpAffine(im, res, M, im.size(),
    //         cv::INTER_LINEAR,
    //         cv::BORDER_CONSTANT,
    //         cv::Scalar(replace[0], replace[1], replace[2]));
    // return res;
}


Mat ShearXFunc(Mat &im, float factor, array<uint8_t, 3> replace, bool inplace) {
    Mat M = Mat::eye(2, 3, CV_64FC1);
    M.at<double>(0, 1) = factor;
    return WarpAffine(im, M, replace, inplace);
}


Mat ShearYFunc(Mat &im, float factor, array<uint8_t, 3> replace, bool inplace) {
    Mat M = Mat::eye(2, 3, CV_64FC1);
    M.at<double>(1, 0) = factor;
    return WarpAffine(im, M, replace, inplace);
}


Mat TranslateXFunc(Mat &im, float offset, array<uint8_t, 3> replace, bool inplace) {
    Mat M = Mat::eye(2, 3, CV_64FC1);
    M.at<double>(0, 2) = -offset;
    return WarpAffine(im, M, replace, inplace);
}


Mat TranslateYFunc(Mat &im, float offset, array<uint8_t, 3> replace, bool inplace) {
    Mat M = Mat::eye(2, 3, CV_64FC1);
    M.at<double>(1, 2) = -offset;
    return WarpAffine(im, M, replace, inplace);
}


Mat SharpnessFunc(Mat &im, float factor, bool inplace) {
    Mat kernel = Mat::ones(3, 3, CV_64FC1);
    kernel.at<double>(1, 1) = 5;
    kernel = kernel * (1. / 13);
    Mat res;
    if (factor == 0) {
        cv::filter2D(im, res, -1, kernel);
    } else if (factor == 1) {
        res = im;
    } else {
        Mat degen;
        cv::filter2D(im, degen, -1, kernel);
        if (inplace) res = im; else res = im.clone();
        auto roi = cv::Rect(1, 1, im.cols - 2, im.rows - 2);
        cv::addWeighted(degen(roi), 1. - factor, im(roi), factor, 0, res(roi));
    }
    return res;
}


Mat PosterizeFunc(Mat &im, int bits, bool inplace) {
    CHECK_EQ(im.elemSize1(), 1) << "must be uint8_t Mat\n";
    Mat res;
    if (inplace) res = im; else res = im.clone();
    uint8_t mask = (0xff) << (8- bits);
    res.forEach<cv::Vec3b>([&mask] (cv::Vec3b &pix, const int* pos) {
            pix[0] &= mask;
            pix[1] &= mask;
            pix[2] &= mask;
    });
    return res;
}


Mat ColorFunc(Mat &im, float factor, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();
    array<float, 3> m1{
        0.886f * factor + 0.114f,
        -0.587f * factor + 0.587f,
        -0.299f * factor + 0.299f};
    array<float, 3> m2{
        -0.114f * factor + 0.114f,
        0.413f * factor + 0.587f,
        -0.299f * factor + 0.299f};
    array<float, 3> m3{
        -0.114f * factor + 0.114f,
        -0.587f * factor + 0.587f,
        0.701f * factor + 0.299f};

    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            float r1 = m1[0] * pix[0] + m1[1] * pix[1] + m1[2] * pix[2];
            float r2 = m2[0] * pix[0] + m2[1] * pix[1] + m2[2] * pix[2];
            float r3 = m3[0] * pix[0] + m3[1] * pix[1] + m3[2] * pix[2];
            pix[0] = (uint8_t)std::max(std::min(r1, 255.f), 0.f);
            pix[1] = (uint8_t)std::max(std::min(r2, 255.f), 0.f);
            pix[2] = (uint8_t)std::max(std::min(r3, 255.f), 0.f);
    });

    // float data[3][3] = {
    //     {0.886f * factor + 0.114f, -0.114f * factor + 0.114f, -0.114f * factor + 0.114f},
    //     {-0.587f * factor + 0.587f, 0.413f * factor + 0.587f, -0.587f * factor + 0.587f},
    //     {-0.299f * factor + 0.299f, -0.299f * factor + 0.299f, 0.701f * factor + 0.299f}};
    // im.convertTo(res, CV_32FC3, 1.);
    // Mat M = Mat(cv::Size(3, 3), CV_32FC1, &data[0][0]);
    // res = res * M;
    return res;
}


Mat InvertFunc(Mat &im, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();
    uint8_t byt = 255;
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            for (int i{0}; i < 3; ++i) {
                pix[i] = byt - pix[i];
            }
    });
    return res;
}


// 1. compute gray mode
// 2. compute mean of the whole gray mode image(one int value rounded)
// 3. blend the org image with the mean value according to factor
Mat ContrastFunc(Mat &im, float factor, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();

    cv::Scalar channel_mean = cv::mean(res);
    float mean = channel_mean[0] * 0.114 + channel_mean[1] * 0.587 + 0.299 * channel_mean[2];

    // vector<float> means(3, 0);
    // cv::parallel_for_(cv::Range(0, im.channels()), [&](const cv::Range &range) {
    //     for (int r{range.start}; r < range.end; ++r) {
    //         double value{0};
    //         for (int i{0}; i < im.rows; ++i) {
    //             uint8_t* ptr = im.ptr<uint8_t>(i);
    //             for (int j{0}; j < im.cols; ++j) {
    //                 means[r] += ptr[r];
    //                 ptr += im.channels();
    //             }
    //         }
    //         means[r] /= im.rows * im.cols;
    //     }
    // });
    // float mean = means[0] * 0.114 + means[1] * 0.587 + means[2] * 0.299;

    array<uint8_t, 256> table;
    float lo{0}, hi{255};
    for (int i{0}; i < 256; ++i) {
        float value = (static_cast<float>(i) - mean) * factor + mean;
        value = std::max(lo, std::min(value, hi));
        table[i] = static_cast<uint8_t>(value);
    }
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            for (int i{0}; i < 3; ++i) {
                pix[i] = table[pix[i]];
            }
    });
    return res;
}


Mat BrightnessFunc(Mat &im, float factor, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();

    array<uint8_t, 256> table;
    float lo{0}, hi{255};
    for (int i{0}; i < 256; ++i) {
        table[i] = static_cast<uint8_t>(std::max(lo, std::min(factor * i, hi)));
    }
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            for (int i{0}; i < 3; ++i) {
                pix[i] = table[pix[i]];
            }
    });
    return res;
}


Mat SolarizeFunc(Mat &im, int thresh, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();

    array<uint8_t, 256> table;
    for (int i{0}; i < 256; ++i) {
        if (i < thresh) {
            table[i] = i;
        } else {
            table[i] = 255 - i;
        }
    }
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            for (int i{0}; i < 3; ++i) {
                pix[i] = table[pix[i]];
            }
    });
    return res;
}


Mat SolarizeAddFunc(Mat &im, int addition, int thresh, bool inplace) {
    Mat res;
    if (inplace) res = im; else res = im.clone();

    array<uint8_t, 256> table;
    for (int i{0}; i < 256; ++i) {
        if (i < thresh) {
            table[i] = std::max(0, std::min(i + addition, 255));
        } else {
            table[i] = i;
        }
    }
    res.forEach<cv::Vec3b>([&] (cv::Vec3b &pix, const int* pos) {
            for (int i{0}; i < 3; ++i) {
                pix[i] = table[pix[i]];
            }
    });
    return res;
}



//// class 

// RandApply
Mat RandApply::FuncWithProb(Mat &im) {
    Mat res = im;
    if (p <= 1 && p > 0 && grandom.rand() < p) {
        SetMagnitude(M);
        res = Func(im);
        // cout << "applying: " << name << endl;
    } else {cout << "not applied by prob\n";}
    return res;
}


// Equalize
Mat Equalize::Func(Mat &im) {
    Mat res = EqualizeFunc(im);
    return  res;
}

void Equalize::SetMagnitude(int mag) {}


// Autocontrast
Mat Autocontrast::Func(Mat &im) {
    Mat res = AutocontrastFunc(im, cutoff);
    return  res;
}

void Autocontrast::SetMagnitude(int mag) {}


// Cutout
Mat Cutout::Func(Mat &im) {
    Mat res = CutoutFunc(im, pad_size, replace, inplace);
    return  res;
}

void Cutout::SetMagnitude(int mag) {
    pad_size = (int)(((float)mag / MAX_LEVEL) * cutout_const);
}


// Rotate
Mat Rotate::Func(Mat &im) {
    Mat res = RotateFunc(im, degree, replace, inplace);
    return  res;
}

void Rotate::SetMagnitude(int mag) {
    degree = ((float)mag / MAX_LEVEL) * 30;
    if (grandom.rand() < 0.5) degree = -degree;
}


// ShearX
Mat ShearX::Func(Mat &im) {
    Mat res = ShearXFunc(im, factor, replace, inplace);
    return  res;
}

void ShearX::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 0.3;
    if (grandom.rand() < 0.5) factor = -factor;
}


// ShearY
Mat ShearY::Func(Mat &im) {
    Mat res = ShearYFunc(im, factor, replace, inplace);
    return  res;
}

void ShearY::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 0.3;
    if (grandom.rand() < 0.5) factor = -factor;
}


// TranslateX
Mat TranslateX::Func(Mat &im) {
    Mat res = TranslateXFunc(im, offset, replace, inplace);
    return  res;
}

void TranslateX::SetMagnitude(int mag) {
    offset = ((float)mag / MAX_LEVEL) * 100;
    if (grandom.rand() < 0.5) offset = -offset;
}


// TranslateY
Mat TranslateY::Func(Mat &im) {
    Mat res = TranslateYFunc(im, offset, replace, inplace);
    return  res;
}

void TranslateY::SetMagnitude(int mag) {
    offset = ((float)mag / MAX_LEVEL) * 100;
    if (grandom.rand() < 0.5) offset = -offset;
}


// Sharpness
Mat Sharpness::Func(Mat &im) {
    Mat res = SharpnessFunc(im, factor, inplace);
    return  res;
}

void Sharpness::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 1.8 + 0.1;
}


// Posterize
Mat Posterize::Func(Mat &im) {
    Mat res = PosterizeFunc(im, bits, inplace);
    return  res;
}

void Posterize::SetMagnitude(int mag) {
    bits = (int)(((float)mag / MAX_LEVEL) * 4);
}


// Color
Mat Color::Func(Mat &im) {
    Mat res = ColorFunc(im, factor, inplace);
    return  res;
}

void Color::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 1.8 + 0.1;
}


// Invert 
Mat Invert::Func(Mat &im) {
    Mat res = InvertFunc(im, inplace);
    return  res;
}

void Invert::SetMagnitude(int mag) {
}


// Contrast
Mat Contrast::Func(Mat &im) {
    Mat res = ContrastFunc(im, factor, inplace);
    return  res;
}

void Contrast::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 1.8 + 0.1;
}


// Brightness
Mat Brightness::Func(Mat &im) {
    Mat res = BrightnessFunc(im, factor, inplace);
    return  res;
}

void Brightness::SetMagnitude(int mag) {
    factor = ((float)mag / MAX_LEVEL) * 1.8 + 0.1;
}


// Solarize
Mat Solarize::Func(Mat &im) {
    Mat res = SolarizeFunc(im, thresh, inplace);
    return  res;
}

void Solarize::SetMagnitude(int mag) {
    thresh = static_cast<int>(((float)mag / MAX_LEVEL) * 256);
}


// SolarizeAdd
Mat SolarizeAdd::Func(Mat &im) {
    Mat res = SolarizeAddFunc(im, addition, thresh, inplace);
    return  res;
}

void SolarizeAdd::SetMagnitude(int mag) {
    addition = static_cast<int>(((float)mag / MAX_LEVEL) * 110);
}
