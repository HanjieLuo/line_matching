#include "klt/klt.h"
using namespace cv;

LKTrackerInvoker1D::LKTrackerInvoker1D(const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                                       const Point2f* _prevPts, const Vec2f* _grads, Point2f* _nextPts,
                                       uchar* _status, float* _err,
                                       Size _winSize, TermCriteria _criteria,
                                       int _level, int _maxLevel, int _flags, float _minEigThreshold,
                                       bool _illumination_adapt,
                                       const std::vector<cv::Mat>* _affines) {
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    grads = _grads;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;

    illumination_adapt = _illumination_adapt;
    affines = _affines;
}

void LKTrackerInvoker1D::operator()(const Range& range) const {
    //range is the index of kp.

    //if winSize is 21*21, halfWin should be 10*10(without the center)
    Point2f halfWin((winSize.width - 1) * 0.5f, (winSize.height - 1) * 0.5f);
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    // std::vector<Mat> der;
    // split(derivI, der);
    // DetectFeatures::ShowMat(der.at(0), "I derivIWinBuf_X");
    // DetectFeatures::ShowMat(der.at(1), "I derivIWinBuf_Y");

    int j, cn = I.channels(), cn2 = cn * 2;
    cv::AutoBuffer<deriv_type> _buf(winSize.area() * (cn2));
    cv::AutoBuffer<float> _buf_float(winSize.area() * (cn));

    int derivDepth = DataType<deriv_type>::depth;

    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
    Mat JWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf + winSize.area() * cn);
    Mat GWinBuf(winSize, CV_MAKETYPE(DataType<float>::depth, cn), (float*)_buf_float);

    for (int ptidx = range.start; ptidx < range.end; ptidx++) {
        Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
        Point2f nextPt;

        float grad_x = grads[ptidx][0];
        float grad_y = grads[ptidx][1];

        if (level == maxLevel) {
            if (flags & OPTFLOW_USE_INITIAL_FLOW) {
                // if (prevPts[ptidx].x == 196 && prevPts[ptidx].y == 160) {
                //     // std::cout<<ptidx<<","<<level<<","<<grads[ptidx]<<std::endl;
                //     std::cout << nextPts[ptidx] << std::endl;
                // }
                nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
            } else {
                nextPt = prevPt;
            }
        } else {
            nextPt = nextPts[ptidx] * 2.f;
        }
        nextPts[ptidx] = nextPt;

        Point2i iprevPt, inextPt;

        //subpixel computation
        //reminder values(0~1)
        float a, b;
        float iG = 0., G, tmp;
        int x, y, ival, ixval, iyval;
        //在计算权重的时候，为了避免浮点计算，会对乘以一定的倍数使用整数运算。
        const int W_BITS = 14, W_BITS1 = 14;

        //00010000 00000000 00000000 = 1048576
        const float FLT_SCALE = 1.f / (1 << 20);

        int iw00, iw01, iw10, iw11;

        int dstep = (int)(derivI.step / derivI.elemSize1());
        int stepI = (int)(I.step / I.elemSize1());
        int stepJ = (int)(J.step / J.elemSize1());

        if (affines == nullptr) {
            //prevPt change to left top
            prevPt -= halfWin;
            //iprevPt is the int location of the left top
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            // not in the range
            if (iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
                iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows) {
                if (level == 0) {
                    if (status)
                        status[ptidx] = false;
                    if (err)
                        err[ptidx] = 0;
                }
                continue;
            }

            //subpixel computation
            //reminder values(0~1)
            a = prevPt.x - iprevPt.x;
            b = prevPt.y - iprevPt.y;

            iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
            iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
            iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            // extract the patch from the first image, compute covariation matrix of derivatives
            // eq.23

            for (y = 0; y < winSize.height; y++) {
                const uchar* src = I.ptr() + (y + iprevPt.y) * stepI + iprevPt.x * cn;
                const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y) * dstep + iprevPt.x * cn2;

                deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                float* Gptr = GWinBuf.ptr<float>(y);

                for (x = 0; x < winSize.width * cn; x++, dsrc += 2) {
                    //interpolation of I(prevPt) , WIDTH 14 BITS
                    ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                                          src[x + stepI] * iw10 + src[x + stepI + cn] * iw11,
                                      W_BITS1 - 5);
                    //interpolation of dIx(prevPt)
                    ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                                           dsrc[dstep] * iw10 + dsrc[dstep + cn2] * iw11,
                                       W_BITS1);
                    //interpolation of dIy(prevPt)
                    iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc[dstep + 1] * iw10 +
                                           dsrc[dstep + cn2 + 1] * iw11,
                                       W_BITS1);

                    Iptr[x] = (short)ival;

                    tmp = ixval * grad_x + iyval * grad_y;
                    Gptr[x] = tmp;
                    iG += tmp * tmp;
                }
            }
        } else {
            // std::cout<<(*affines)[ptidx]<<std::endl;
            float aff00 = ((*affines)[ptidx]).at<float>(0, 0);
            float aff01 = ((*affines)[ptidx]).at<float>(0, 1);
            float aff10 = ((*affines)[ptidx]).at<float>(1, 0);
            float aff11 = ((*affines)[ptidx]).at<float>(1, 1);

            float tmp = aff00 * aff11 - aff01 * aff10;

            if (tmp < 1e-9) {
                if (level == 0) {
                    if (status) {
                        status[ptidx] = false;
                    }
                    if (err) {
                        err[ptidx] = 0;
                    }
                }
                continue;
            }

            float aff00_inv = aff11 / tmp;
            float aff01_inv = -aff01 / tmp;
            float aff10_inv = -aff10 / tmp;
            float aff11_inv = aff00 / tmp;

            float px_offset, py_offset, px_ref, py_ref;
            int ipx_ref, ipy_ref;

            //check in range
            int coner_x[4] = {0, winSize.width * cn - 1, 0, winSize.width * cn - 1};
            int coner_y[4] = {0, 0, winSize.height - 1, winSize.height - 1};

            bool flag = false;

            for (x = 0; x < 4; x++) {
                px_offset = (coner_x[x] - halfWin.x);
                py_offset = (coner_y[x] - halfWin.y);

                px_ref = px_offset * aff00_inv + py_offset * aff01_inv + prevPt.x;
                py_ref = px_offset * aff10_inv + py_offset * aff11_inv + prevPt.y;

                ipx_ref = cvFloor(px_ref);
                ipy_ref = cvFloor(py_ref);

                // std::cout<<px_ref<<","<<py_ref<<std::endl;
                // not in the range

                if (ipx_ref < -winSize.width || ipx_ref >= derivI.cols ||
                    ipy_ref < -winSize.height || ipy_ref >= derivI.rows) {
                    if (level == 0) {
                        if (status)
                            status[ptidx] = false;
                        if (err)
                            err[ptidx] = 0;
                    }
                    flag = true;
                    break;
                }
            }

            if (flag) {
                continue;
            }

            // std::cout<<"OK"<<std::endl;

            for (y = 0; y < winSize.height; y++) {
                deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                float* Gptr = GWinBuf.ptr<float>(y);

                py_offset = (y - halfWin.y);
                for (x = 0; x < winSize.width * cn; x++) {
                    px_offset = (x - halfWin.x);

                    px_ref = px_offset * aff00_inv + py_offset * aff01_inv + prevPt.x;
                    py_ref = px_offset * aff10_inv + py_offset * aff11_inv + prevPt.y;

                    ipx_ref = cvFloor(px_ref);
                    ipy_ref = cvFloor(py_ref);

                    //subpixel computation
                    //reminder values(0~1)
                    a = px_ref - ipx_ref;
                    b = py_ref - ipy_ref;

                    iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
                    iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
                    iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
                    iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                    const uchar* src = I.ptr() + ipy_ref * stepI;
                    const deriv_type* dsrc = derivI.ptr<deriv_type>() + ipy_ref * dstep + ipx_ref * cn2;

                    ival = CV_DESCALE(src[ipx_ref] * iw00 + src[ipx_ref + cn] * iw01 +
                                          src[ipx_ref + stepI] * iw10 + src[ipx_ref + stepI + cn] * iw11,
                                      W_BITS1 - 5);
                    //interpolation of dIx(prevPt)
                    ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                                           dsrc[dstep] * iw10 + dsrc[dstep + cn2] * iw11,
                                       W_BITS1);
                    //interpolation of dIy(prevPt)
                    iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc[dstep + 1] * iw10 +
                                           dsrc[dstep + cn2 + 1] * iw11,
                                       W_BITS1);

                    Iptr[x] = (short)ival;

                    tmp = ixval * grad_x + iyval * grad_y;
                    Gptr[x] = tmp;
                    iG += tmp * tmp;
                }
            }
        }
        // DetectFeatures::ShowMat(GWinBuf, "GWinBuf");
        // std::vector<Mat> der;
        // split(derivIWinBuf, der);
        // DetectFeatures::ShowMat(der.at(0), "derivIWinBuf_X");
        // DetectFeatures::ShowMat(der.at(1), "derivIWinBuf_Y");

        G = iG * FLT_SCALE;

        //det(G)
        // float D = A11 * A22 - A12 * A12;
        //avg of min eigenvalue of G
        // float minEig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) +
        //                                       4.f * A12 * A12)) / (2 * winSize.width * winSize.height);

        //err is the min eigenvalues
        if (err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0) {
            err[ptidx] = (float)G;
        }

        //FLT_EPSILON is min value which is close to 0
        // D FLT_EPSILON means there is no domain direction, means flat
        if (G < minEigThreshold) {
            if (level == 0 && status) {
                status[ptidx] = false;
            }
            continue;
        }

        // D = 1.f / D;

        //nextPt change to left top
        //the nextPt is equal to prevPt or given now
        nextPt -= halfWin;
        Point2f prevDelta;
        // float prevDelta;

        // std::cout<<nextPt<<std::endl;

        // std::cout<<winSize.width<<","<<winSize.height<<std::endl;
        // std::cout<<J.cols<<","<<J.rows<<std::endl<<std::endl;
        // std::cout<<(*affines)[ptidx]<<std::endl;
        // if (debug) {
        //     std::cout<<"level:"<<level<<std::endl;
        //     KLT::ShowMat("IWinBuf", IWinBuf);
        // }
        // KLT::ShowMat("IWinBuf", IWinBuf);
        for (j = 0; j < criteria.maxCount; j++) {
            //inextPt is the int part of nextpt
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if (inextPt.x < -halfWin.x || inextPt.x >= J.cols ||
                inextPt.y < -halfWin.y || inextPt.y >= J.rows) {
                if (level == 0 && status) {
                    status[ptidx] = false;
                }
                break;
            }

            // std::cout<<"level:"<<level<<", j:"<<j<<std::endl;

            // a, b is the float part of nextPt
            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
            iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
            iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            for (y = 0; y < winSize.height; y++) {
                const uchar* dst = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;

                deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

                for (x = 0; x < winSize.width * cn; x++) {
                    Jptr[x] = (short)CV_DESCALE(dst[x] * iw00 + dst[x + cn] * iw01 +
                                                    dst[x + stepJ] * iw10 + dst[x + stepJ + cn] * iw11,
                                                W_BITS1 - 5);
                }
            }

            // std::cout<<nextPt<<std::endl;
            // if (debug) {
            //     KLT::ShowMat("JWinBuf", JWinBuf);
            //     cv::waitKey(0);
            // }
            // std::cout<<level<<","<<j<<std::endl;
            // KLT::ShowMat("JWinBuf", JWinBuf);
            // cv::waitKey(0);

            float alpha, beta;
            if (illumination_adapt) {
                getImageNormParams(IWinBuf, JWinBuf, alpha, beta);
            } else {
                alpha = 1.0;
                beta = 0.0;
            }

            float ib = 0.;
            float b;
            float diff;

            for (y = 0; y < winSize.height; y++) {
                // const uchar* Jptr = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);
                const float* Gptr = GWinBuf.ptr<float>(y);

                for (x = 0; x < winSize.width * cn; x++) {
                    // get the interpolation of J, eq30
                    // int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
                    //                       Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + cn] * iw11,
                    //                       W_BITS1 - 5) - Iptr[x];
                    diff = (float)(alpha * Jptr[x] + beta - Iptr[x]);

                    // image mismatch vector bk, eq29
                    ib += diff * Gptr[x];
                    // std::cout<<"d:"<<diff<<std::endl;
                    // std::cout<<"Gptr:"<<Gptr[x]<<std::endl;
                }
            }

            b = ib * FLT_SCALE;

            // get the inverse of G mulipy the bk, eq28
            // D is 1/det(G) now
            // here is -inverse of G, WHY?
            // it is beacuse delta is in the I, means oppside direction of in the J

            float v = -(b / G);
            // std::cout<<std::endl<<"b:"<<b<<std::endl;
            // std::cout<<"G:"<<G<<std::endl;
            // std::cout<<"v:"<<v<<std::endl;
            Point2f delta((float)(grad_x * v),
                          (float)(grad_y * v));
            // std::cout<<"grad:"<<grads[ptidx]<<std::endl;
            // std::cout<<"delta:"<<delta<<std::endl;
            nextPt += delta;

            //nextPts becomes the center location, eq31
            nextPts[ptidx] = nextPt + halfWin;

            //dot product, get the norm of delta
            if (delta.ddot(delta) <= criteria.epsilon) {
                // std::cout<<"ddot"<<std::endl;
                break;
            }

            //?
            if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                std::abs(delta.y + prevDelta.y) < 0.01) {
                nextPts[ptidx] -= delta * 0.5f;
                // std::cout<<"break"<<std::endl;
                break;
            }
            prevDelta = delta;
        }

        if (j == criteria.maxCount && status && level == 0) {
            // if (debug) {
            //     std::cout<<"FALSE!!"<<std::endl;
            // }
            status[ptidx] = false;
        }

        CV_Assert(status != NULL);
        //calc the error, err = (I -J) / (32 * WINSIZE)
        if (level == 0 && status[ptidx] && err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0) {
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if (inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows) {
                if (status)
                    status[ptidx] = false;
                continue;
            }

            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa) * (1.f - bb) * (1 << W_BITS));
            iw01 = cvRound(aa * (1.f - bb) * (1 << W_BITS));
            iw10 = cvRound((1.f - aa) * bb * (1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            for (y = 0; y < winSize.height; y++) {
                const uchar* dst = J.ptr() + (y + inextPoint.y) * stepJ + inextPoint.x * cn;
                deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

                for (x = 0; x < winSize.width * cn; x++) {
                    Jptr[x] = (short)CV_DESCALE(dst[x] * iw00 + dst[x + cn] * iw01 +
                                                    dst[x + stepJ] * iw10 + dst[x + stepJ + cn] * iw11,
                                                W_BITS1 - 5);
                }
            }

            float alpha, beta;
            if (illumination_adapt) {
                getImageNormParams(IWinBuf, JWinBuf, alpha, beta);
            } else {
                alpha = 1.0;
                beta = 0.0;
            }

            float errval = 0.f;
            for (y = 0; y < winSize.height; y++) {
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

                // the interpolation of J in nextPts - I
                for (x = 0; x < winSize.width * cn; x++) {
                    errval += std::abs((float)(alpha * Jptr[x] + beta - Iptr[x]));
                }
            }
            err[ptidx] = errval * 1.f / (32 * winSize.width * cn * winSize.height);
        }
    }
}
