#include "klt/klt.h"
using namespace cv;

LKTrackerInvokerOri::LKTrackerInvokerOri(const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                                         const Point2f* _prevPts, Point2f* _nextPts,
                                         uchar* _status, float* _err,
                                         Size _winSize, TermCriteria _criteria,
                                         int _level, int _maxLevel, int _flags, float _minEigThreshold) {
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;
}

void LKTrackerInvokerOri::operator()(const Range& range) const {
    //range is the index of kp.

    //if winSize is 21*21, halfWin should be 10*10(without the center)
    Point2f halfWin((winSize.width - 1) * 0.5f, (winSize.height - 1) * 0.5f);
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    int j, cn = I.channels(), cn2 = cn * 2;
    cv::AutoBuffer<deriv_type> _buf(winSize.area() * (cn + cn2));
    int derivDepth = DataType<deriv_type>::depth;

    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf);
    Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf + winSize.area() * cn);

    for (int ptidx = range.start; ptidx < range.end; ptidx++) {
        Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
        Point2f nextPt;
        if (level == maxLevel) {
            if (flags & OPTFLOW_USE_INITIAL_FLOW)
                nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
            else
                nextPt = prevPt;
        } else {
            nextPt = nextPts[ptidx] * 2.f;
            // std::cout<<nextPts[ptidx].x<<","<<nextPts[ptidx].y<<std::endl;
            // std::cout<<nextPt.x<<","<<nextPt.y<<std::endl;
        }
        nextPts[ptidx] = nextPt;

        Point2i iprevPt, inextPt;
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
        float a = prevPt.x - iprevPt.x;
        float b = prevPt.y - iprevPt.y;
        //在计算权重的时候，为了避免浮点计算，会对乘以一定的倍数使用整数运算。
        const int W_BITS = 14, W_BITS1 = 14;

        //00010000 00000000 00000000 = 1048576
        const float FLT_SCALE = 1.f / (1 << 20);

        int iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
        int iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
        int iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        int dstep = (int)(derivI.step / derivI.elemSize1());
        int stepI = (int)(I.step / I.elemSize1());
        int stepJ = (int)(J.step / J.elemSize1());
        float iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

        // extract the patch from the first image, compute covariation matrix of derivatives
        // eq.23
        int x, y;
        for (y = 0; y < winSize.height; y++) {
            const uchar* src = I.ptr() + (y + iprevPt.y) * stepI + iprevPt.x * cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y) * dstep + iprevPt.x * cn2;

            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            for (x = 0; x < winSize.width * cn; x++, dsrc += 2, dIptr += 2) {
                //interpolation of I(prevPt) , WIDTH 14 BITS
                int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                                          src[x + stepI] * iw10 + src[x + stepI + cn] * iw11,
                                      W_BITS1 - 5);
                //interpolation of dIx(prevPt)
                int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                                           dsrc[dstep] * iw10 + dsrc[dstep + cn2] * iw11,
                                       W_BITS1);
                //interpolation of dIy(prevPt)
                int iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc[dstep + 1] * iw10 +
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

        A11 = iA11 * FLT_SCALE;
        A12 = iA12 * FLT_SCALE;
        A22 = iA22 * FLT_SCALE;

        //det(G)
        float D = A11 * A22 - A12 * A12;
        //avg of min eigenvalue of G
        float minEig = (A22 + A11 - std::sqrt((A11 - A22) * (A11 - A22) + 4.f * A12 * A12)) / (2 * winSize.width * winSize.height);

        //err is the min eigenvalues
        if (err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0)
            err[ptidx] = (float)minEig;

        //FLT_EPSILON is min value which is close to 0
        // D FLT_EPSILON means there is no domain direction, means flat
        if (minEig < minEigThreshold || D < FLT_EPSILON) {
            if (level == 0 && status)
                status[ptidx] = false;
            continue;
        }

        D = 1.f / D;

        //nextPt change to left top
        //the nextPt is equal to prevPt or given now
        nextPt -= halfWin;
        Point2f prevDelta;

        for (j = 0; j < criteria.maxCount; j++) {
            //inextPt is the int part of nextpt
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if (inextPt.x < -halfWin.x || inextPt.x >= J.cols ||
                inextPt.y < -halfWin.y || inextPt.y >= J.rows) {
                if (level == 0 && status)
                    status[ptidx] = false;
                break;
            }

            // a, b is the float part of nextPt
            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a) * (1.f - b) * (1 << W_BITS));
            iw01 = cvRound(a * (1.f - b) * (1 << W_BITS));
            iw10 = cvRound((1.f - a) * b * (1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float ib1 = 0, ib2 = 0;
            float b1, b2;

            for (y = 0; y < winSize.height; y++) {
                const uchar* Jptr = J.ptr() + (y + inextPt.y) * stepJ + inextPt.x * cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

                for (x = 0; x < winSize.width * cn; x++, dIptr += 2) {
                    // get the interpolation of J, eq30
                    int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
                                              Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + cn] * iw11,
                                          W_BITS1 - 5) -
                               Iptr[x];

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
            nextPts[ptidx] = nextPt + halfWin;

            // std::cout<< std::fixed<<std::setprecision(12)<<delta.ddot(delta)<<","<<criteria.epsilon <<std::endl;
            //dot product, get the norm of delta
            if (delta.ddot(delta) <= criteria.epsilon) {
                break;
            }

            //?
            // std::cout<<j<<","<<delta.x<<","<<prevDelta.x<<","<<delta.y<<","<<prevDelta.y<<std::endl;
            if (j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
                std::abs(delta.y + prevDelta.y) < 0.01) {
                nextPts[ptidx] -= delta * 0.5f;
                break;
            }
            // std::cout<<j<<","<<nextPts[ptidx].x<<","<<nextPts[ptidx].y<<std::endl;
            prevDelta = delta;
        }

        if (j == criteria.maxCount && status && level == 0) {
            status[ptidx] = false;
        }

        CV_Assert(status != NULL);
        //calc the error, err = (I -J) / (32 * WINSIZE)
        if (status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0) {
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
            float errval = 0.f;

            for (y = 0; y < winSize.height; y++) {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y) * stepJ + inextPoint.x * cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);

                // the interpolation of J in nextPts - I
                for (x = 0; x < winSize.width * cn; x++) {
                    int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
                                              Jptr[x + stepJ] * iw10 + Jptr[x + stepJ + cn] * iw11,
                                          W_BITS1 - 5) -
                               Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            err[ptidx] = errval * 1.f / (32 * winSize.width * cn * winSize.height);
        }
    }
}