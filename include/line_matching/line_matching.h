#ifndef LINE_MATCHING_H
#define LINE_MATCHING_H

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "edline/edline_detector.h"
#include "klt/klt.h"

class LineMatching {
   public:
    LineMatching(int step = 10,
                 float closest_line_threshold = 1.0,
                 float line_matching_ratio = 0.4,
                 float line_distance_error_ratio = 3,
                 float klt_error_threshold = 40);
    ~LineMatching();

    bool Matching(const cv::Mat &img_ref,
                  const cv::Mat &img_cur,
                  const std::vector<Line> &lines_ref,
                  const std::vector<Line> &lines_cur,
                  std::vector<int> &line_ref_to_line_cur,
                  const cv::Mat &K_ref = *(cv::Mat *)NULL,
                  const cv::Mat &K_cur = *(cv::Mat *)NULL,
                  const cv::Mat &T_cur_ref = *(cv::Mat *)NULL,  // 4*4
                  const bool illumination_adapt = false,
                  const bool topological_filter = true,
                  const int debug_show = 1,
                  const int debug_img_ref_id = 0,
                  const int debug_img_cur_id = 0);

    void LineFilter(std::vector<Line> &lines,
                    float distance_threshold,
                    float parallel_threshold = 0.0348994967);  //sin(3 degree)

    void TopologicalFilter(const std::vector<Line> &lines_ref,
                           const std::vector<Line> &lines_cur,
                           const std::vector<int> &line_ref_to_line_cur,
                           std::vector<bool> &status,
                           const cv::Mat *img_ref_debug = NULL,
                           const cv::Mat *img_cur_debug = NULL,
                           const float distance_threshold = 15,
                           const float line_length_change_tolerate_ratio = 0.2,
                           const float violation_ratio = 0.05);

    //if l2 center point is on the right side of the line l1, output 1
    //if l2 center point if on the left side of the line l1, output -1
    //otherwise output 0
    //distance is from l2 center point to line l1
   //  int SidenessCheck(const Line &l1, const Line &l2, float &distance);
   bool SidenessCheck(const Line &l1_ref, const Line &l2_ref,
                      const Line &l1_cur, const Line &l2_cur,
                      float &distance1, float &distance2);

    void DashLine(cv::Mat &img,
                  const cv::Point2f &p1,
                  const cv::Point2f &p2,
                  const cv::Scalar &bgr,
                  const int interval = 5);

    void DashRotatedRect(cv::Mat &img,
                         const cv::Point2f &center,
                         const float width,
                         const float height,
                         const float angle,
                         const cv::Scalar &bgr,
                         const int interval = 5);

    void getPointMatchResult(std::vector<cv::Point2f> &kps_ref,
                             std::vector<cv::Point2f> &kps_init,
                             std::vector<cv::Vec2f> &dirs,
                             std::vector<cv::Point2f> &kps_cur,
                             std::vector<int> &line_kp_num_ref,
                             std::vector<int> &kp2line_cur);

   private:
    int step_;
    float closest_line_threshold_;
    float klt_error_threshold_;
    float line_matching_ratio_;
    float line_distance_error_ratio_;

    cv::Ptr<KLT> klt_;

    std::vector<cv::Point2f> kps_ref_;
    std::vector<cv::Point2f> kps_cur_;
    std::vector<cv::Point2f> kps_init_;
    std::vector<cv::Vec2f> dirs_;
    std::vector<uchar> status_;
    std::vector<float> errors_;
    std::vector<int> kp2line_ref_;
    std::vector<int> kp2line_cur_;

    std::vector<int> line_kp_num_ref_;

    std::vector<int> topological_filter_cout_;

    void ClosestLine(const std::vector<cv::Point2f> &kps_cur,
                     const std::vector<uchar> &status,
                     const std::vector<float> &errors,
                     const std::vector<Line> &lines_cur,
                     std::vector<int> &kp2line_cur);

    float PointLineDistance(const cv::Point2f &pt,
                            const std::array<float, 4> &line_endpoint);

    float PointLineDistance(const float x, const float y,
                            const std::array<float, 4> &line_endpoint);

    void Point2Line(const std::vector<Line> &lines_ref,
                    const std::vector<Line> &lines_cur,
                    const std::vector<int> &line_kp_num_ref,
                    const std::vector<int> &kp2line_cur,
                    const int lines_cur_num,
                    std::vector<int> &line_ref_to_line_cur);

    bool Anchors(const std::vector<Line> &lines_ref,
                 const cv::Mat &K_ref,
                 const cv::Mat &K_cur,
                 const cv::Mat &T_cur_ref,
                 std::vector<cv::Point2f> &kps_ref,
                 std::vector<cv::Point2f> &kps_cur,
                 std::vector<cv::Vec2f> &dirs,
                 std::vector<int> &kp2line_ref,
                 std::vector<int> &line_kp_num_ref);

   /*
    void Anchors(const std::vector<Line> &lines_ref,
                 const std::vector<Line> &lines_pre,
                 const std::vector<int> &line_ref_to_line_pre,
                 std::vector<cv::Point2f> &kps_ref,
                 std::vector<cv::Point2f> &kps_cur,
                 std::vector<cv::Vec2f> &dirs,
                 std::vector<int> &kp2line_ref,
                 std::vector<int> &line_kp_num_ref);
   */

    void DebugShow(const int &debug_show,
                   const cv::Mat &img_ref,
                   const cv::Mat &img_cur,
                   const std::vector<Line> &lines_ref,
                   const std::vector<Line> &lines_cur,
                   const std::vector<cv::Point2f> &kps_ref,
                   const std::vector<cv::Point2f> &kps_cur,
                   const std::vector<cv::Point2f> &kps_init,
                   const std::vector<cv::Vec2f> &dirs,
                   const std::vector<uchar> &status,
                   const std::vector<int> &kp2line_cur,
                   const std::vector<int> &line_kp_num_ref,
                   const std::vector<int> &line_ref_to_line_cur,
                   const std::vector<int> &line_ref_to_line_cur_before_filiter,
                   const std::vector<bool> &top_filter_status,
                   const int debug_img_ref_id,
                   const int debug_img_cur_id);
};
#endif