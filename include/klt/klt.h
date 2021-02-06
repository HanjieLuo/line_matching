#ifndef MYKLT_H
#define MYKLT_H

#include <float.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <immintrin.h>

#include <unordered_set>

#include <opencv2/opencv.hpp>
#include "lkpyramid.hpp"
#include "opencv2/ts.hpp"
#include "opencv2/video.hpp"

#ifndef CV_CPU_HAS_SUPPORT_SSE2
#define CV_CPU_HAS_SUPPORT_SSE2 (cv::checkHardwareSupport(CV_CPU_SSE2))
#endif
#include "opencv2/core/hal/intrin.hpp"

#include <sys/time.h>

#define CVTE_SSEB true
#define CVTE_SSES true
#define CVTE_SSEG true

#define CVTE_SSEB1D true
#define CVTE_SSEG1D true

#define CVTE_SSEILL12D true
#define CVTE_SSEILL22D true

#define CVTE_SSEILL11D true
#define CVTE_SSEILL21D true

using cv::detail::deriv_type;

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

void getImageNormParams(const cv::Mat &src, const cv::Mat &dst, float &alpha, float &beta);

void getImageNormParams(const float &src_mean, const float &src_sd, const cv::Mat &dst, float &alpha, float &beta);

class KLT : public cv::SparsePyrLKOpticalFlow {
   public:
    /** @brief Computes KLT algorithm.
    @param winSize size of the search window at each pyramid level.
    @param maxLevel 0-based maximal pyramid level number; if set to 0, pyramids are not used (single
    level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
    algorithm will use as many levels as pyramids have but no more than maxLevel.
    @param criteria parameter, specifying the termination criteria of the iterative search algorithm
    (after the specified maximum number of iterations criteria.maxCount or when the search window
    moves by less than criteria.epsilon.
    @param minEigThreshold the algorithm calculates the minimum eigen value of a 2x2 normal matrix of
    optical flow equations (this matrix is called a spatial gradient matrix in @cite Bouguet00), divided
    by number of pixels in a window; if this value is less than minEigThreshold, then a corresponding
    feature is filtered out and its flow is not processed, so it allows to remove bad points and get a
    performance boost.
    @param flags operation flags:
    -   **OPTFLOW_USE_INITIAL_FLOW** uses initial estimations, stored in nextPts; if the flag is
     not set, then prevPts is copied to nextPts and is considered the initial estimate.
    -   **OPTFLOW_LK_GET_MIN_EIGENVALS** use minimum eigen values as an error measure (see
     minEigThreshold description); if the flag is not set, then L1 distance between patches
     around the original and a moved point, divided by number of pixels in a window, is used as a
     error measure. (It is useless within Track1D)
    **/
    KLT(cv::Size winSize = cv::Size(21, 21),
        int maxLevel = 3,
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        double minEigThreshold = 1e-4,
        int flags = cv::OPTFLOW_USE_INITIAL_FLOW);

    ~KLT();

    virtual cv::Size getWinSize() const { return winSize_; }
    virtual void setWinSize(cv::Size winSize) { winSize_ = winSize; }

    virtual int getMaxLevel() const { return maxLevel_; }
    virtual void setMaxLevel(int maxLevel) { maxLevel_ = maxLevel; }

    virtual cv::TermCriteria getTermCriteria() const { return criteria_; }
    virtual void setTermCriteria(cv::TermCriteria &crit) { criteria_ = crit; }

    virtual int getFlags() const { return flags_; }
    virtual void setFlags(int flags) { flags_ = flags; }

    virtual double getMinEigThreshold() const { return minEigThreshold_; }
    virtual void setMinEigThreshold(double minEigThreshold) { minEigThreshold_ = minEigThreshold; }

    /** @brief The Original KLT Algorithm.
    **/
    virtual void calc(cv::InputArray prevImg, cv::InputArray nextImg,
                      cv::InputArray prevPts, cv::InputOutputArray nextPts,
                      cv::OutputArray status,
                      cv::OutputArray err = cv::noArray());

    /** @brief Computes 2D KLT algorithm.
    @param prevImg first 8-bit input image.
    @param nextImg second input image or pyramid of the same size and the same type as prevImg.
    @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
    single-precision floating-point numbers.
    @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
    containing the calculated new positions of input features in the second image; when
    OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
    @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
    the flow for the corresponding features has been found, otherwise, it is set to 0.
    @param errors output vector of errors; each element of the vector is set to an error for the
    corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
    found then the error is not defined (use the status parameter to find such cases).
    @param illumination_adapt Optional, To make KLT tracking more robust with respect those changes due to illumination.
    @param affines Optional affine transformation model for target image(see SVO)
    **/
    void calc2D(cv::InputArray prevImg, cv::InputArray nextImg,
                cv::InputArray prevPts, cv::InputOutputArray nextPts,
                cv::OutputArray status,
                cv::OutputArray errors = cv::noArray(),
                bool illumination_adapt = false,
                const std::vector<cv::Mat> *affines = nullptr);

    void calc2D(const cv::Mat &prevImg,
                const cv::Mat &nextImg,
                const cv::Point2f &prevPt,
                cv::Point2f &nextPt,
                uchar &status,
                float &error = *(float *)NULL,
                bool illumination_adapt = true,
                cv::Mat &affine = *(cv::Mat *)NULL);


    /** @brief Computes 1D KLT algorithm.
    @param prevImg first 8-bit input image.
    @param nextImg second input image or pyramid of the same size and the same type as prevImg.
    @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be
    single-precision floating-point numbers.
    @param grads Search direction of each prevPts.
    @param nextPts output vector of 2D points (with single-precision floating-point coordinates)
    containing the calculated new positions of input features in the second image; when
    OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.
    @param grad_pre vector of 2D norm vector for each keypoint in prevPts.
    @param status output status vector (of unsigned chars); each element of the vector is set to 1 if
    the flow for the corresponding features has been found, otherwise, it is set to 0.
    @param errors output vector of errors; each element of the vector is set to an error for the
    corresponding feature, type of the error measure can be set in flags parameter; if the flow wasn't
    found then the error is not defined (use the status parameter to find such cases).
    @param illumination_adapt To make KLT tracking more robust with respect those changes due to illumination.
    @param affines Affine transformation model for target image(see SVO)
    **/
    void calc1D(cv::InputArray prevImg,
                cv::InputArray nextImg,
                cv::InputArray prevPts,
                cv::InputArray dirs,
                cv::InputOutputArray nextPts,
                cv::OutputArray status,
                cv::OutputArray errors = cv::noArray(),
                bool illumination_adapt = false,
                const std::vector<cv::Mat> *affines = nullptr);

   
    void getAffineModel(const cv::Point2f &kp_ref,
                        const cv::Mat &K_ref,
                        const cv::Mat &K_cur,
                        const cv::Mat &T_cur_ref,
                        const float &kp_ref_z,  // z axix
                        cv::Mat &Affine);

    static void ShowMat(const std::string name, const cv::Mat &src);

   private:
    cv::Size winSize_;
    int maxLevel_;
    cv::TermCriteria criteria_;
    int flags_;
    double minEigThreshold_;
    float maxError_;

    std::vector<cv::Size> winSizes_;
    std::vector<cv::Point2i> mask_pts_;

    std::vector<cv::Mat> prevPyr_;
    std::vector<cv::Mat> nextPyr_;
    cv::Mat derivIBuf_;


    static void calcSharrDeriv(const cv::Mat &src, cv::Mat &dst);

    void LKTracker2DSingle(const cv::Mat &prevImg,
                           const cv::Mat &prevDeriv,
                           const cv::Mat &nextImg,
                           const cv::Point2f &prevPts,
                           cv::Point2f &nextPts,
                           uchar &status,
                           float &err,
                           int level,
                           bool illumination_adapt,
                           cv::Mat &affines);
};

class LKTrackerInvokerOri {
   public:
    LKTrackerInvokerOri(const cv::Mat &_prevImg,
                        const cv::Mat &_prevDeriv,
                        const cv::Mat &_nextImg,
                        const cv::Point2f *_prevPts,
                        cv::Point2f *_nextPts,
                        uchar *_status,
                        float *_err,
                        cv::Size _winSize,
                        cv::TermCriteria _criteria,
                        int _level,
                        int _maxLevel,
                        int _flags,
                        float _minEigThreshold);

    void operator()(const cv::Range &range) const;

   private:
    const cv::Mat *prevImg;
    const cv::Mat *prevDeriv;
    const cv::Mat *nextImg;
    const cv::Point2f *prevPts;
    cv::Point2f *nextPts;
    uchar *status;
    float *err;
    cv::Size winSize;
    cv::TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};

class LKTrackerInvoker1D {
   public:
    LKTrackerInvoker1D(const cv::Mat &_prevImg,
                       const cv::Mat &_prevDeriv,
                       const cv::Mat &_nextImg,
                       const cv::Point2f *_prevPts,
                       const cv::Vec2f *grads,
                       cv::Point2f *_nextPts,
                       uchar *_status,
                       float *_err,
                       cv::Size _winSize,
                       cv::TermCriteria _criteria,
                       int _level,
                       int _maxLevel,
                       int _flags,
                       float _minEigThreshold,
                       bool _illumination_adapt,
                       const std::vector<cv::Mat> *_affines);

    void operator()(const cv::Range &range) const;

   private:
    const cv::Mat *prevImg;
    const cv::Mat *prevDeriv;
    const cv::Mat *nextImg;
    const cv::Point2f *prevPts;
    const cv::Vec2f *grads;
    cv::Point2f *nextPts;
    uchar *status;
    float *err;
    cv::Size winSize;
    cv::TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;

    bool illumination_adapt;
    const std::vector<cv::Mat> *affines;
};


class LKTrackerInvoker2D {
   public:
    LKTrackerInvoker2D(const cv::Mat &_prevImg,
                       const cv::Mat &_prevDeriv,
                       const cv::Mat &_nextImg,
                       const cv::Point2f *_prevPts,
                       cv::Point2f *_nextPts,
                       uchar *_status,
                       float *_err,
                       cv::Size _winSize,
                       cv::TermCriteria _criteria,
                       int _level,
                       int _maxLevel,
                       int _flags,
                       float _minEigThreshold,
                       bool _illumination_adapt,
                       const std::vector<cv::Mat> *_affines);

    void operator()(const cv::Range &range) const;

   private:
    const cv::Mat *prevImg;
    const cv::Mat *prevDeriv;
    const cv::Mat *nextImg;
    const cv::Point2f *prevPts;
    uchar *status;
    float *err;
    cv::Size winSize;
    cv::TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
    cv::Point2f *nextPts;

    bool illumination_adapt;
    const std::vector<cv::Mat> *affines;
};




#endif
