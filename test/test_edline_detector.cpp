#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include "edline/edline_detector.h"

void test() {
    // ksize, sigma, gradientThreshold, anchorThreshold, scanIntervals, minLineLen, lineFitErrThreshold
    EDLineParam param = {5, 1.0, 30, 5, 2, 25, 1.8};
    EDLineDetector line_detctor = EDLineDetector(param);
    std::vector<Line> lines;

    cv::Mat img = cv::imread("./../data/mh04/imgs/1.png", 0);

    struct timeval t1, t2;
    int test_times = 100;
    double run_time, all_run_time_sum = 0;

    for (int i = 0; i < test_times; i++) {
        gettimeofday(&t1, NULL);
        line_detctor.EDline(img, lines, false);
        gettimeofday(&t2, NULL);

        run_time = (t2.tv_sec - t1.tv_sec) * 1000. + (t2.tv_usec - t1.tv_usec) / 1000.;
        all_run_time_sum += run_time;
        std::cout<<"run time: "<<run_time<<"ms"<<std::endl;
    }

    int line_num = (int)lines.size();
    std::cout << "Edline parallel version:"<<std::endl; 
    std::cout << "Detect " << line_num << " lines" << std::endl;
    std::cout << "Avg run time:" << all_run_time_sum / test_times << "ms" << std::endl<<std::endl; ;


    cv::Mat img_show;
    srand((unsigned)time(0));
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    unsigned int r, g, b;  //the color of lines
    float mid_x, mid_y, normal_x, normal_y, dir_x, dir_y;
    cv::Point startPoint, endPoint;

    cv::cvtColor(img, img_show, cv::COLOR_GRAY2BGR);

    for (int i = 0; i < line_num; i++) {
        r = lowest + int(rand() % range);
        g = lowest + int(rand() % range);
        b = lowest + int(rand() % range);

        startPoint = cv::Point(lines[i].line_endpoint[0], lines[i].line_endpoint[1]);
        endPoint = cv::Point(lines[i].line_endpoint[2], lines[i].line_endpoint[3]);

        cv::line(img_show, startPoint, endPoint, cv::Scalar(r, g, b), 2, CV_AA);

        mid_x = lines[i].center[0];
        mid_y = lines[i].center[1];

        normal_x = lines[i].line_equation[0];
        normal_y = lines[i].line_equation[1];

        dir_x = -lines[i].line_equation[1];
        dir_y = lines[i].line_equation[0];

        cv::arrowedLine(img_show, cv::Point(mid_x, mid_y), cv::Point(mid_x + 10 * normal_x, mid_y + 10 * normal_y), cv::Scalar(255, 0, 0), 1, CV_AA);
        cv::arrowedLine(img_show, cv::Point(mid_x, mid_y), cv::Point(mid_x + 10 * dir_x, mid_y + 10 * dir_y), cv::Scalar(0, 0, 255), 1, CV_AA);
    }

    cv::imshow("img_show", img_show);
    // cv::imwrite("./../data/edline_result.png", img_show);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    test();
    return 0;
}