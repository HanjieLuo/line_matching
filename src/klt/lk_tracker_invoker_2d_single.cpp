#include "klt/klt.h"
using namespace cv;

void KLT::LKTracker2DSingle(const cv::Mat& prevImg,
                            const cv::Mat& prevDeriv,
                            const cv::Mat& nextImg,
                            const cv::Point2f& prevPts,
                            cv::Point2f& nextPts,
                            uchar& status,
                            float& err,
                            int level,
                            bool illumination_adapt,
                            cv::Mat& affines) {
    Point2f halfWin((winSize_.width - 1) * 0.5f, (winSize_.height - 1) * 0.5f);
    const Mat& I = prevImg;
    const Mat& J = nextImg;
    const Mat& derivI = prevDeriv;

    int j, cn = I.channels(), cn2 = cn * 2;
    cv::AutoBuffer<deriv_type> _buf(winSize_.area() * (cn + cn + cn2));
    int derivDepth = DataType<deriv_type>::depth;

    Mat IWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
    Mat JWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf + winSize_.area() * cn);
    Mat derivIWinBuf(winSize_, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf + winSize_.area() * cn2);

    Point2f prevPt = prevPts * (float)(1. / (1 << level));
    Point2f nextPt;
    if (level == maxLevel_) {
        if (flags_ & OPTFLOW_USE_INITIAL_FLOW)
            nextPt = nextPts * (float)(1. / (1 << level));
        else
            nextPt = prevPt;
    } else {
        nextPt = nextPts * 2.f;
    }

    nextPts = nextPt;

    Point2i iprevPt, inextPt;
    //å¨è®¡ç®æéçæ¶åï¼ä¸ºäºé¿åæµ®ç¹è®¡ç®ï¼ä¼å¯¹ä¹ä»¥ä¸å®çåæ°ä½¿ç¨æ´æ°è¿ç®ã
    const int W_BITS = 14, W_BITS1 = 14;

    //00010000 00000000 00000000 = 1048576
    const float FLT_SCALE = 1.f / (1 << 20);

    float a, b;
    int iw00, iw01, iw10, iw11;
    float iA11 = 0, iA12 = 0, iA22 = 0;
    float A11, A12, A22;
    int x, y, ival, ixval, iyval;

    int dstep = (int)(derivI.step / derivI.elemSize1());
    int stepI = (int)(I.step / I.elemSize1());
    int stepJ = (int)(J.step / J.elemSize1());

    if (&affines == NULL) {
        //prevPt change to left top
        prevPt -= halfWin;
        //iprevPt is the int location of the left top
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        // not in the range
        if (iprevPt.x < -winSize_.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize_.height || iprevPt.y >= derivI.rows) {
            if (level == 0) {
                status = false;
                if (&err)
                    err = 0;
            }
            return;
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

        for (y = 0; y < winSize_.height; y++) {
            const uchar* src = I.ptr() + (y + iprevPt.y) * stepI + iprevPt.x * cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y) * dstep + iprevPt.x * cn2;

            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            for (x = 0; x < winSize_.width * cn; x++, dsrc += 2, dIptr += 2) {
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
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

                iA11 += (float)(ixval * ixval);
                iA12 += (float)(ixval * iyval);
                iA22 += (float)(iyval * iyval);
            }
        }
    } else {
        float aff00 = affines.at<float>(0, 0);
        float aff01 = affines.at<float>(0, 1);
        float aff10 = affines.at<float>(1, 0);
        float aff11 = affines.at<float>(1, 1);

        float tmp = aff00 * aff11 - aff01 * aff10;

        if (tmp < 1e-9) {
            if (level == 0) {
                status = false;
                if (&err) {
                    err = 0;
                }
            }
            return;
        }

        float aff00_inv = aff11 / tmp;
        float aff01_inv = -aff01 / tmp;
        float aff10_inv = -aff10 / tmp;
        float aff11_inv = aff00 / tmp;

        float px_offset, py_offset, px_ref, py_ref;
        int ipx_ref, ipy_ref;

        //check in range
        int coner_x[4] = {0, winSize_.width * cn - 1, 0, winSize_.width * cn - 1};
        int coner_y[4] = {0, 0, winSize_.height - 1, winSize_.height - 1};

        bool flag = false;
        for (x = 0; x < 4; x++) {
            px_offset = (coner_x[x] - halfWin.x);
            py_offset = (coner_y[x] - halfWin.y);

            px_ref = px_offset * aff00_inv + py_offset * aff01_inv + prevPt.x;
            py_ref = px_offset * aff10_inv + py_offset * aff11_inv + prevPt.y;

            ipx_ref = cvFloor(px_ref);
            ipy_ref = cvFloor(py_ref);

            // not in the range

            if (ipx_ref < -winSize_.width || ipx_ref >= derivI.cols ||
                ipy_ref < -winSize_.height || ipy_ref >= derivI.rows) {
                if (level == 0) {
                    status = false;
                    if (&err)
                        err = 0;
                }
                flag = true;
                break;
            }
        }

        if (flag) {
            return;
        }

        for (y = 0; y < winSize_.height; y++) {
            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            py_offset = (y - halfWin.y);
            for (x = 0; x < winSize_.width * cn; x++, dIptr += 2) {
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
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

                iA11 += (float)(ixval * ixval);
                iA12 += (float)(ixval * iyval);
                iA22 += (float)(iyval * iyval);
            }
        }
    }

    A11 = iA11 * FLT_SCALE;
    A12 = iA12 * FLT_SCALE;
    A22 = iA22 * FLT_SCALE;

    //det(G)
    float D = A11 * A22 - A12 * A12;
    //avg of min eigenvalue of G
    float minEig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * winSize_.width * winSize_.height);

    //err is the min eigenvalues
    if (&err && (flags_ & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0)
        err = (float)minEig;

    //FLT_EPSILON is min value which is close to 0
    // D FLT_EPSILON means there is no domain direction, means flat
    if (minEig < minEigThreshold_ || D < FLT_EPSILON) {
        if (level == 0)
            status = false;
        return;
    }

    D = 1.f / D;

    //nextPt change to left top
    //the nextPt is equal to prevPt or given now
    nextPt -= halfWin;
    Point2f prevDelta;

    // ShowMat("IWinBuf", IWinBuf);

    for (j = 0; j < criteria_.maxCount; j++) {
        //inextPt is the int part of nextpt
        inextPt.x = cvFloor(nextPt.x);
        inextPt.y = cvFloor(nextPt.y);

        if (inextPt.x < -halfWin.x || inextPt.x >= J.cols ||
            inextPt.y < -halfWin.y || inextPt.y >= J.rows) {
            if (level == 0)
                status = false;
            break;
        }

        // a, b is the float part of nextPt
        a = nextPt.x - inextPt.x;
        b = nextPt.y - inextPt.y;
        iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
        iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
        iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
        iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        for (y = 0; y < winSize_.height; y++) {
            const uchar* dst = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;

            deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

            for (x = 0; x < winSize_.width * cn; x++) {
                Jptr[x] = (short)CV_DESCALE(dst[x] * iw00 + dst[x + cn] * iw01 +
                                                dst[x + stepJ] * iw10 + dst[x + stepJ + cn] * iw11,
                                            W_BITS1 - 5);
            }
        }

        // ShowMat("JWinBuf", JWinBuf);
        // cv::waitKey(0);

        float alpha, beta;
        if (illumination_adapt) {
            getImageNormParams(IWinBuf, JWinBuf, alpha, beta);
        } else {
            alpha = 1.0;
            beta = 0.0;
        }

        float ib1 = 0, ib2 = 0;
        float b1, b2;
        float diff;

        for (y = 0; y < winSize_.height; y++) {
            const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            const deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);
            const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            for (x = 0; x < winSize_.width * cn; x++, dIptr += 2) {
                diff = (float)(alpha * Jptr[x] + beta - Iptr[x]);
                // image mismatch vector bk, eq29
                ib1 += (float)(diff * dIptr[0]);
                ib2 += (float)(diff * dIptr[1]);
            }
        }

        b1 = ib1 * FLT_SCALE;
        b2 = ib2 * FLT_SCALE;

        // get the inverse of G mulipy the bk, eq28
        // D is 1/det(G) now
        // here is -inverse of G, WHY?
        // it is beacuse delta is in the I, means oppside direction of in the J
        Point2f delta((float)((A12 * b2 - A22 * b1) * D),
                      (float)((A12 * b1 - A11 * b2) * D));

        nextPt += delta;
        //nextPts becomes the center location, eq31
        nextPts = nextPt + halfWin;

        //dot product, get the norm of delta
        if (delta.ddot(delta) <= criteria_.epsilon) {
            break;
        }

        //?
        if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
            std::abs(delta.y + prevDelta.y) < 0.01) {
            nextPts -= delta * 0.5f;
            break;
        }
        prevDelta = delta;
    }

    // break_out:

    if (j == criteria_.maxCount && level == 0) {
        status = false;
    }

    // CV_Assert(status != NULL);
    //calc the error, err = (I -J) / (32 * winSize_)

    if (level == 0 && &err && (flags_ & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0) {
        Point2f nextPoint = nextPts - halfWin;
        Point inextPoint;

        inextPoint.x = cvFloor(nextPoint.x);
        inextPoint.y = cvFloor(nextPoint.y);

        if (inextPoint.x < -winSize_.width || inextPoint.x >= J.cols ||
            inextPoint.y < -winSize_.height || inextPoint.y >= J.rows) {
            status = false;
            return;
        }

        float aa = nextPoint.x - inextPoint.x;
        float bb = nextPoint.y - inextPoint.y;
        iw00 = cvRound((1.f - aa) * (1.f - bb) * (1 << W_BITS));
        iw01 = cvRound(aa * (1.f - bb) * (1 << W_BITS));
        iw10 = cvRound((1.f - aa) * bb * (1 << W_BITS));
        iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        for (y = 0; y < winSize_.height; y++) {
            const uchar* dst = J.ptr() + (y + inextPoint.y) * stepJ + inextPoint.x * cn;
            deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

            for (x = 0; x < winSize_.width * cn; x++) {
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
        for (y = 0; y < winSize_.height; y++) {
            const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            const deriv_type* Jptr = JWinBuf.ptr<deriv_type>(y);

            // the interpolation of J in nextPts - I
            for (x = 0; x < winSize_.width * cn; x++) {
                errval += std::abs((float)(alpha * Jptr[x] + beta - Iptr[x]));
            }
        }
        err = errval * 1.f / (32 * winSize_.width * cn * winSize_.height);
    }
}