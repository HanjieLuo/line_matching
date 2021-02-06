#include "klt/klt.h"
using namespace cv;

void getImageNormParams(const cv::Mat& src, const cv::Mat& dst, float& alpha, float& beta) {
    Mat src_m, src_sd, dst_m, dst_sd;
    meanStdDev(src, src_m, src_sd);
    meanStdDev(dst, dst_m, dst_sd);
    alpha = float(src_sd.at<double>(0, 0) / dst_sd.at<double>(0, 0));
    beta = float(src_m.at<double>(0, 0) - alpha * dst_m.at<double>(0, 0));
}

void getImageNormParams(const float& src_mean, const float& src_sd, const cv::Mat& dst, float& alpha, float& beta) {
    // Mat src_m, src_sd, dst_m, dst_sd;
    Mat dst_m, dst_sd;
    // meanStdDev(src, src_m, src_sd);
    meanStdDev(dst, dst_m, dst_sd);
    alpha = src_sd / float(dst_sd.at<double>(0, 0));
    beta = src_mean - alpha * float(dst_m.at<double>(0, 0));
}

KLT::KLT(cv::Size winSize, int maxLevel, cv::TermCriteria criteria, double minEigThreshold, int flags) : winSize_(winSize), maxLevel_(maxLevel), criteria_(criteria), minEigThreshold_(minEigThreshold) {
    if ((criteria_.type & TermCriteria::COUNT) == 0) {
        criteria_.maxCount = 30;
    } else {
        criteria_.maxCount = std::min(std::max(criteria_.maxCount, 0), 100);
    }

    if ((criteria_.type & TermCriteria::EPS) == 0) {
        criteria_.epsilon = 0.01;
    } else {
        criteria_.epsilon = std::min(std::max(criteria_.epsilon, 0.), 10.);
    }
    criteria_.epsilon *= criteria_.epsilon;

    flags_ = flags;
}

KLT::~KLT() {
}


void KLT::calcSharrDeriv(const cv::Mat& src, cv::Mat& dst) {
    using namespace cv;
    using cv::detail::deriv_type;

    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols * cn, depth = src.depth();

    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn * 2));

    int x, y, delta = (int)alignSize((cols + 2) * cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta * 2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

    v_int16x8 c3 = v_setall_s16(3), c10 = v_setall_s16(10);
    bool haveSIMD = checkHardwareSupport(CV_CPU_SSE2);

    for (y = 0; y < rows; y++) {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y - 1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows - 1 ? y + 1 : rows > 1 ? rows - 2 : 0);
        deriv_type* drow = dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;

        if (haveSIMD) {
            for (; x <= colsn - 8; x += 8) {
                v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(srow0 + x));
                v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(srow1 + x));
                v_int16x8 s2 = v_reinterpret_as_s16(v_load_expand(srow2 + x));

                v_int16x8 t1 = s2 - s0;
                v_int16x8 t0 = (s0 + s2) * c3 + s1 * c10;

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }

        for (; x < colsn; x++) {
            int t0 = (srow0[x] + srow2[x]) * 3 + srow1[x] * 10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0) * cn, x1 = (cols > 1 ? cols - 2 : 0) * cn;
        for (int k = 0; k < cn; k++) {
            trow0[-cn + k] = trow0[x0 + k];
            trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k];
            trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;

        if (haveSIMD) {
            for (; x <= colsn - 8; x += 8) {
                v_int16x8 s0 = v_load(trow0 + x - cn);
                v_int16x8 s1 = v_load(trow0 + x + cn);
                v_int16x8 s2 = v_load(trow1 + x - cn);
                v_int16x8 s3 = v_load(trow1 + x);
                v_int16x8 s4 = v_load(trow1 + x + cn);

                v_int16x8 t0 = s1 - s0;
                v_int16x8 t1 = ((s2 + s4) * c3) + (s3 * c10);

                v_store_interleave((drow + x * 2), t0, t1);
            }
        }

        for (; x < colsn; x++) {
            deriv_type t0 = (deriv_type)(trow0[x + cn] - trow0[x - cn]);
            deriv_type t1 = (deriv_type)((trow1[x + cn] + trow1[x - cn]) * 3 + trow1[x] * 10);
            drow[x * 2] = t0;
            drow[x * 2 + 1] = t1;
        }
    }
}

void KLT::getAffineModel(const cv::Point2f& kp_ref,
                         const cv::Mat& K_ref,
                         const cv::Mat& K_cur,
                         const cv::Mat& T_cur_ref,
                         const float& kp_ref_z,  // z axix
                         cv::Mat& affine) {
    cv::Mat R_cur_ref = T_cur_ref(cv::Rect(0, 0, 3, 3));
    cv::Mat t_cur_ref = T_cur_ref(cv::Rect(3, 0, 1, 3));
    cv::Mat K_ref_inv = K_ref.inv();

    cv::Mat pt = (cv::Mat_<float>(3, 1) << kp_ref.x, kp_ref.y, 1.);  //ref, x axis (1, 0)
    pt = K_ref_inv * pt;                                             //world
    pt = pt * (kp_ref_z / pt.at<float>(2));                          //world
    pt = K_cur * (R_cur_ref * pt + t_cur_ref);                       //cur
    float pt_x = pt.at<float>(0) / pt.at<float>(2);
    float pt_y = pt.at<float>(1) / pt.at<float>(2);

    cv::Mat du = (cv::Mat_<float>(3, 1) << kp_ref.x + 1., kp_ref.y, 1.);  //ref, x axis (1, 0)
    du = K_ref_inv * du;                                                  //world
    du = du * (kp_ref_z / du.at<float>(2));                               //world
    du = K_cur * (R_cur_ref * du + t_cur_ref);                            //cur
    float du_x = du.at<float>(0) / du.at<float>(2);
    float du_y = du.at<float>(1) / du.at<float>(2);

    cv::Mat dv = (cv::Mat_<float>(3, 1) << kp_ref.x, kp_ref.y + 1., 1.);  //ref, y axis (0, 1)
    dv = K_ref_inv * dv;                                                  //world
    dv = dv * (kp_ref_z / dv.at<float>(2));
    dv = K_cur * (R_cur_ref * dv + t_cur_ref);
    float dv_x = dv.at<float>(0) / dv.at<float>(2);
    float dv_y = dv.at<float>(1) / dv.at<float>(2);

    affine = (cv::Mat_<float>(2, 2) << du_x - pt_x, dv_x - pt_x,
              du_y - pt_y, dv_y - pt_y);

    // std::cout<<affine<<std::endl;
    // Affine.at<float>(0, 0) = du_x;
    // Affine.at<float>(0, 1) = dv_x;
    // Affine.at<float>(1, 0) = du_y;
    // Affine.at<float>(1, 1) = dv_y;

    // du = (cv::Mat_<float>(3, 1) << 1., 0, 1.);  //ref, x axis (1, 0)
    // du = K_ref_inv * du;                                               //world
    // du = du * (kp_ref_z / du.at<float>(2));                              //world
    // du = K_cur * (R_cur_ref * du + t_cur_ref);                           //cur
    // du_x = du.at<float>(0) / du.at<float>(2);
    // du_y = du.at<float>(1) / du.at<float>(2);

    // dv = (cv::Mat_<float>(3, 1) << 0, 1., 1.);  //ref, y axis (0, 1)
    // dv = K_ref_inv * dv;                                                    //world
    // dv = dv * (kp_ref_z / dv.at<float>(2));
    // dv = K_cur * (R_cur_ref * dv + t_cur_ref);
    // dv_x = dv.at<float>(0) / dv.at<float>(2);
    // dv_y = dv.at<float>(1) / dv.at<float>(2);

    // std::cout<<du_x<<","<<du_y<<std::endl;
    // std::cout<<dv_x<<","<<dv_y<<std::endl;
}

void KLT::ShowMat(const std::string name, const cv::Mat& src) {
    double min;
    double max;
    cv::minMaxIdx(src, &min, &max);
    cv::Mat adjMap;

    float scale = 255 / (max - min);
    src.convertTo(adjMap, CV_8UC1, scale, -min * scale);

    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_PINK);

    // if (falseColorsMap.cols < 100) {
    // 	cv::resize(falseColorsMap, falseColorsMap, cv::Size(), 100. / falseColorsMap.cols, 100. / falseColorsMap.cols);
    // }

    cv::imshow(name, falseColorsMap);
}

void KLT::calc(InputArray _prevImg, InputArray _nextImg,
               InputArray _prevPts, InputOutputArray _nextPts,
               OutputArray _status, OutputArray _err) {
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    CV_Assert(maxLevel_ >= 0 && winSize_.width > 2 && winSize_.height > 2);

    int level = 0, i, npoints;
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

    if (npoints == 0) {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    if (!(flags_ & OPTFLOW_USE_INITIAL_FLOW))
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert(statusMat.isContinuous());
    uchar* status = statusMat.ptr();
    float* err = 0;

    for (i = 0; i < npoints; i++)
        status[i] = true;

    if (_err.needed()) {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert(errMat.isContinuous());
        err = errMat.ptr<float>();
    }

    // std::vector<Mat> prevPyr_, nextPyr_;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if (_prevImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _prevImg.getMatVector(prevPyr_);

        levels1 = int(prevPyr_.size()) - 1;
        CV_Assert(levels1 >= 0);

        if (levels1 % 2 == 1 && prevPyr_[0].channels() * 2 == prevPyr_[1].channels() && prevPyr_[1].depth() == derivDepth) {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels1 > 0) {
            Size fullSize;
            Point ofs;
            prevPyr_[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + prevPyr_[lvlStep1].cols + winSize_.width <= fullSize.width && ofs.y + prevPyr_[lvlStep1].rows + winSize_.height <= fullSize.height);
        }

        if (levels1 < maxLevel_) {
            maxLevel_ = levels1;
        }
    }

    if (_nextImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _nextImg.getMatVector(nextPyr_);

        levels2 = int(nextPyr_.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr_[0].channels() * 2 == nextPyr_[1].channels() && nextPyr_[1].depth() == derivDepth) {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels2 > 0) {
            Size fullSize;
            Point ofs;
            nextPyr_[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + nextPyr_[lvlStep2].cols + winSize_.width <= fullSize.width && ofs.y + nextPyr_[lvlStep2].rows + winSize_.height <= fullSize.height);
        }

        if (levels2 < maxLevel_)
            maxLevel_ = levels2;
    }

    if (levels1 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_prevImg, prevPyr_, winSize_, maxLevel_, false);
    }

    if (levels2 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_nextImg, nextPyr_, winSize_, maxLevel_, false);
    }

    // dI/dx ~ Ix, dI/dy ~ Iyh
    // Mat derivIBuf_;
    if (lvlStep1 == 1)
        derivIBuf_.create(prevPyr_[0].rows + winSize_.height * 2, prevPyr_[0].cols + winSize_.width * 2, CV_MAKETYPE(derivDepth, prevPyr_[0].channels() * 2));

    for (level = maxLevel_; level >= 0; level--) {
        // std::cout<<"lvlStep1:"<<lvlStep1<<std::endl;
        Mat derivI;
        if (lvlStep1 == 1) {
            Size imgSize = prevPyr_[level * lvlStep1].size();
            Mat _derivI(imgSize.height + winSize_.height * 2,
                        imgSize.width + winSize_.width * 2, derivIBuf_.type(), derivIBuf_.ptr());
            derivI = _derivI(Rect(winSize_.width, winSize_.height, imgSize.width, imgSize.height));
            // eq.19 eq.20
            calcSharrDeriv(prevPyr_[level * lvlStep1], derivI);
            //扩充src边缘，将图像变大，便于处理边界，该函数调用了cv::borderInterpolate
            copyMakeBorder(derivI, _derivI, winSize_.height, winSize_.height, winSize_.width, winSize_.width, BORDER_CONSTANT | BORDER_ISOLATED);
        } else {
            derivI = prevPyr_[level * lvlStep1 + 1];
        }

        CV_Assert(prevPyr_[level * lvlStep1].size() == nextPyr_[level * lvlStep2].size());
        CV_Assert(prevPyr_[level * lvlStep1].type() == nextPyr_[level * lvlStep2].type());

        parallel_for_(Range(0, npoints), LKTrackerInvokerOri(prevPyr_[level * lvlStep1], derivI,
                                                             nextPyr_[level * lvlStep2], prevPts, nextPts,
                                                             status, err,
                                                             winSize_, criteria_, level, maxLevel_,
                                                             flags_, (float)minEigThreshold_));
    }
}

void KLT::calc1D(InputArray _prevImg,
                 InputArray _nextImg,
                 InputArray _prevPts,
                 InputArray _grads,
                 InputOutputArray _nextPts,
                 OutputArray _status,
                 OutputArray _err,
                 bool illumination_adapt,
                 const std::vector<cv::Mat>* _affines) {
    //1 x grads.size, channels 2
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    CV_Assert(maxLevel_ >= 0 && winSize_.width > 2 && winSize_.height > 2);

    int level = 0, i, npoints;
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

    if (npoints == 0) {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    Mat gradsMat = _grads.getMat();
    CV_Assert(gradsMat.checkVector(2, CV_32F, true) == npoints);
    Vec2f* grads = gradsMat.ptr<Vec2f>();

    if (!(flags_ & OPTFLOW_USE_INITIAL_FLOW))
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert(statusMat.isContinuous());
    uchar* status = statusMat.ptr();
    float* err = 0;

    for (i = 0; i < npoints; i++)
        status[i] = true;

    if (_err.needed()) {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert(errMat.isContinuous());
        err = errMat.ptr<float>();
    }

    // std::vector<Mat> prevPyr_, nextPyr_;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if (_prevImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _prevImg.getMatVector(prevPyr_);

        levels1 = int(prevPyr_.size()) - 1;
        CV_Assert(levels1 >= 0);

        if (levels1 % 2 == 1 && prevPyr_[0].channels() * 2 == prevPyr_[1].channels() && prevPyr_[1].depth() == derivDepth) {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels1 > 0) {
            Size fullSize;
            Point ofs;
            prevPyr_[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + prevPyr_[lvlStep1].cols + winSize_.width <= fullSize.width && ofs.y + prevPyr_[lvlStep1].rows + winSize_.height <= fullSize.height);
        }

        if (levels1 < maxLevel_) {
            maxLevel_ = levels1;
        }
    }

    if (_nextImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _nextImg.getMatVector(nextPyr_);

        levels2 = int(nextPyr_.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr_[0].channels() * 2 == nextPyr_[1].channels() && nextPyr_[1].depth() == derivDepth) {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels2 > 0) {
            Size fullSize;
            Point ofs;
            nextPyr_[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + nextPyr_[lvlStep2].cols + winSize_.width <= fullSize.width && ofs.y + nextPyr_[lvlStep2].rows + winSize_.height <= fullSize.height);
        }

        if (levels2 < maxLevel_)
            maxLevel_ = levels2;
    }

    if (levels1 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_prevImg, prevPyr_, winSize_, maxLevel_, false);
    }

    if (levels2 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_nextImg, nextPyr_, winSize_, maxLevel_, false);
    }

    // dI/dx ~ Ix, dI/dy ~ Iyh
    // Mat derivIBuf_;
    if (lvlStep1 == 1)
        derivIBuf_.create(prevPyr_[0].rows + winSize_.height * 2, prevPyr_[0].cols + winSize_.width * 2, CV_MAKETYPE(derivDepth, prevPyr_[0].channels() * 2));

    for (level = maxLevel_; level >= 0; level--) {
        // std::cout<<"lvlStep1:"<<lvlStep1<<std::endl;
        Mat derivI;
        if (lvlStep1 == 1) {
            Size imgSize = prevPyr_[level * lvlStep1].size();
            Mat _derivI(imgSize.height + winSize_.height * 2,
                        imgSize.width + winSize_.width * 2, derivIBuf_.type(), derivIBuf_.ptr());
            derivI = _derivI(Rect(winSize_.width, winSize_.height, imgSize.width, imgSize.height));
            // eq.19 eq.20
            calcSharrDeriv(prevPyr_[level * lvlStep1], derivI);

            //扩充src边缘，将图像变大，便于处理边界，该函数调用了cv::borderInterpolate
            copyMakeBorder(derivI, _derivI, winSize_.height, winSize_.height, winSize_.width, winSize_.width, BORDER_CONSTANT | BORDER_ISOLATED);

            // std::cout<<"level:"<<level<<std::endl;
            // std::vector<Mat> der;
            // split(derivI, der);
            // DetectFeatures::ShowMat(der.at(0), "derivIWinBuf_X");
            // DetectFeatures::ShowMat(der.at(1), "derivIWinBuf_Y");
        } else {
            derivI = prevPyr_[level * lvlStep1 + 1];
        }

        CV_Assert(prevPyr_[level * lvlStep1].size() == nextPyr_[level * lvlStep2].size());
        CV_Assert(prevPyr_[level * lvlStep1].type() == nextPyr_[level * lvlStep2].type());

        parallel_for_(Range(0, npoints), LKTrackerInvoker1D(prevPyr_[level * lvlStep1], derivI,
                                                            nextPyr_[level * lvlStep2], prevPts, grads, nextPts,
                                                            status, err,
                                                            winSize_, criteria_, level, maxLevel_,
                                                            flags_, (float)minEigThreshold_, illumination_adapt, _affines));
    }
}

void KLT::calc2D(InputArray _prevImg, InputArray _nextImg,
                 InputArray _prevPts, InputOutputArray _nextPts,
                 OutputArray _status, OutputArray _err,
                 bool _illumination_adapt,
                 const std::vector<cv::Mat>* _affines) {
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    CV_Assert(maxLevel_ >= 0 && winSize_.width > 2 && winSize_.height > 2);

    int level = 0, i, npoints;
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

    if (npoints == 0) {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    if (!(flags_ & OPTFLOW_USE_INITIAL_FLOW))
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert(statusMat.isContinuous());
    uchar* status = statusMat.ptr();
    float* err = 0;

    for (i = 0; i < npoints; i++)
        status[i] = true;

    if (_err.needed()) {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert(errMat.isContinuous());
        err = errMat.ptr<float>();
    }

    // std::vector<Mat> prevPyr_, nextPyr_;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if (_prevImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _prevImg.getMatVector(prevPyr_);

        levels1 = int(prevPyr_.size()) - 1;
        CV_Assert(levels1 >= 0);

        if (levels1 % 2 == 1 && prevPyr_[0].channels() * 2 == prevPyr_[1].channels() && prevPyr_[1].depth() == derivDepth) {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels1 > 0) {
            Size fullSize;
            Point ofs;
            prevPyr_[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + prevPyr_[lvlStep1].cols + winSize_.width <= fullSize.width && ofs.y + prevPyr_[lvlStep1].rows + winSize_.height <= fullSize.height);
        }

        if (levels1 < maxLevel_) {
            maxLevel_ = levels1;
        }
    }

    if (_nextImg.kind() == _InputArray::STD_VECTOR_MAT) {
        _nextImg.getMatVector(nextPyr_);

        levels2 = int(nextPyr_.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr_[0].channels() * 2 == nextPyr_[1].channels() && nextPyr_[1].depth() == derivDepth) {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has reqired padding
        if (levels2 > 0) {
            Size fullSize;
            Point ofs;
            nextPyr_[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize_.width && ofs.y >= winSize_.height && ofs.x + nextPyr_[lvlStep2].cols + winSize_.width <= fullSize.width && ofs.y + nextPyr_[lvlStep2].rows + winSize_.height <= fullSize.height);
        }

        if (levels2 < maxLevel_)
            maxLevel_ = levels2;
    }

    if (levels1 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_prevImg, prevPyr_, winSize_, maxLevel_, false);
    }

    if (levels2 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(_nextImg, nextPyr_, winSize_, maxLevel_, false);
    }

    // dI/dx ~ Ix, dI/dy ~ Iyh
    // Mat derivIBuf_;
    if (lvlStep1 == 1)
        derivIBuf_.create(prevPyr_[0].rows + winSize_.height * 2, prevPyr_[0].cols + winSize_.width * 2, CV_MAKETYPE(derivDepth, prevPyr_[0].channels() * 2));

    for (level = maxLevel_; level >= 0; level--) {
        // std::cout<<"lvlStep1:"<<lvlStep1<<std::endl;
        Mat derivI;
        if (lvlStep1 == 1) {
            Size imgSize = prevPyr_[level * lvlStep1].size();
            Mat _derivI(imgSize.height + winSize_.height * 2,
                        imgSize.width + winSize_.width * 2, derivIBuf_.type(), derivIBuf_.ptr());
            derivI = _derivI(Rect(winSize_.width, winSize_.height, imgSize.width, imgSize.height));
            // eq.19 eq.20
            calcSharrDeriv(prevPyr_[level * lvlStep1], derivI);
            //扩充src边缘，将图像变大，便于处理边界，该函数调用了cv::borderInterpolate
            copyMakeBorder(derivI, _derivI, winSize_.height, winSize_.height, winSize_.width, winSize_.width, BORDER_CONSTANT | BORDER_ISOLATED);
        } else {
            derivI = prevPyr_[level * lvlStep1 + 1];
        }

        CV_Assert(prevPyr_[level * lvlStep1].size() == nextPyr_[level * lvlStep2].size());
        CV_Assert(prevPyr_[level * lvlStep1].type() == nextPyr_[level * lvlStep2].type());

        // cv::setNumThreads(1);
        parallel_for_(Range(0, npoints), LKTrackerInvoker2D(prevPyr_[level * lvlStep1], derivI,
                                                            nextPyr_[level * lvlStep2], prevPts, nextPts,
                                                            status, err,
                                                            winSize_, criteria_, level, maxLevel_,
                                                            flags_, (float)minEigThreshold_, _illumination_adapt, _affines));
    }
}

void KLT::calc2D(const cv::Mat& prevImg,
                 const cv::Mat& nextImg,
                 const cv::Point2f& prevPt,
                 cv::Point2f& nextPt,
                 uchar& status,
                 float& error,
                 bool illumination_adapt,
                 cv::Mat& affine) {
    // Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<cv::detail::deriv_type>::depth;

    CV_Assert(maxLevel_ >= 0 && winSize_.width > 2 && winSize_.height > 2);

    int level;
    status = true;

    // std::vector<Mat> prevPyr_, nextPyr_;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if (levels1 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(prevImg, prevPyr_, winSize_, maxLevel_, false);
    }

    if (levels2 < 0) {
        maxLevel_ = buildOpticalFlowPyramid(nextImg, nextPyr_, winSize_, maxLevel_, false);
    }

    // dI/dx ~ Ix, dI/dy ~ Iyh
    // Mat derivIBuf_;
    if (lvlStep1 == 1)
        derivIBuf_.create(prevPyr_[0].rows + winSize_.height * 2, prevPyr_[0].cols + winSize_.width * 2, CV_MAKETYPE(derivDepth, prevPyr_[0].channels() * 2));

    for (level = maxLevel_; level >= 0; level--) {
        Mat derivI;
        if (lvlStep1 == 1) {
            Size imgSize = prevPyr_[level * lvlStep1].size();
            Mat _derivI(imgSize.height + winSize_.height * 2,
                        imgSize.width + winSize_.width * 2, derivIBuf_.type(), derivIBuf_.ptr());
            derivI = _derivI(Rect(winSize_.width, winSize_.height, imgSize.width, imgSize.height));
            // eq.19 eq.20
            calcSharrDeriv(prevPyr_[level * lvlStep1], derivI);
            //扩充src边缘，将图像变大，便于处理边界，该函数调用了cv::borderInterpolate
            copyMakeBorder(derivI, _derivI, winSize_.height, winSize_.height, winSize_.width, winSize_.width, BORDER_CONSTANT | BORDER_ISOLATED);
        } else {
            derivI = prevPyr_[level * lvlStep1 + 1];
        }

        CV_Assert(prevPyr_[level * lvlStep1].size() == nextPyr_[level * lvlStep2].size());
        CV_Assert(prevPyr_[level * lvlStep1].type() == nextPyr_[level * lvlStep2].type());

        LKTracker2DSingle(prevPyr_[level * lvlStep1],
                          derivI,
                          nextPyr_[level * lvlStep2],
                          prevPt,
                          nextPt,
                          status,
                          error,
                          level,
                          illumination_adapt,
                          affine);
    }
}

