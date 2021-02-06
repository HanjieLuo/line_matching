#ifndef LINE_H
#define LINE_H

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

struct Line {
    std::array<float, 4> line_endpoint;
    std::array<double, 3> line_equation;
    std::array<float, 2> center;
    float length;

    std::vector<cv::Point2f> kps;
    std::vector<cv::Point2f> kps_init;
    std::vector<cv::Vec2f> dirs;
};

#endif