#include <sys/time.h>

#include <iostream>

#include "line_matching/line_matching.h"

void test() {
    // provide the reference transform matrix between two frames. It will get better result if given.
    bool use_reference_T = true;
    // 0: close  1: show debug information 2: show more debug information
    int debug_show = 2;
    // illumination adapt for KLT tracker
    bool illumination_adapt = false;
    // remove outlider
    bool topological_filter = true;

    int idx_cout = 0;
    int idx_ref = 1;
    int idx_cur = idx_ref + 1;

    cv::Mat img_ref = cv::imread("./../data/mh04/imgs/" + std::to_string(idx_ref) + ".png", 0);

    cv::Mat K = (cv::Mat_<float>(3, 3) << 436.23459, 0, 364.44122,
                 0, 436.23459, 256.95169,
                 0, 0, 1);

    // ksize, sigma, gradientThreshold, anchorThreshold, scanIntervals, minLineLen, lineFitErrThreshold
    // EDLineParam param = {5, 1.0, 30, 5, 2, 25, 2.0};  //1.4
    EDLineParam param = {5, 1.0, 30, 5, 2, 25, 1.8};
    EDLineDetector line_detctor = EDLineDetector(param);
    std::vector<Line> lines_ref;
    line_detctor.EDline(img_ref, lines_ref, false);

    LineMatching line_matching = LineMatching();
    float filter_distance = 3.0;
    line_matching.LineFilter(lines_ref, filter_distance);

    char key;
    cv::Mat img_cur;
    std::vector<Line> lines_cur;
    std::vector<int> line_ref_to_line_cur;
    cv::Mat T_cur_ref;
    while (idx_cur <= 15) {
        img_cur = cv::imread("./../data/mh04/imgs/" + std::to_string(idx_cur) + ".png", 0);

        line_detctor.EDline(img_cur, lines_cur, false);
        line_matching.LineFilter(lines_cur, filter_distance);

        if (use_reference_T) {
            cv::FileStorage storage("./../data/mh04/T_cur_ref/" + std::to_string(idx_cur) + ".yml", cv::FileStorage::READ);
            storage["T_cur_ref"] >> T_cur_ref;
            storage.release();

            line_matching.Matching(img_ref,
                                   img_cur,
                                   lines_ref,
                                   lines_cur,
                                   line_ref_to_line_cur,
                                   K,
                                   K,
                                   T_cur_ref,
                                   illumination_adapt,
                                   topological_filter,
                                   debug_show,
                                   idx_ref, idx_cur);

        } else {
            line_matching.Matching(img_ref,
                                   img_cur,
                                   lines_ref,
                                   lines_cur,
                                   line_ref_to_line_cur,
                                   K,
                                   K,
                                   *(cv::Mat*)NULL,
                                   illumination_adapt,
                                   topological_filter,
                                   debug_show,
                                   idx_ref, idx_cur);
        }

        if (idx_cur % 5 == 0) {
            img_ref = img_cur;
            idx_ref = idx_cur;

            line_detctor.EDline(img_ref, lines_ref, false);
            line_matching.LineFilter(lines_ref, filter_distance);

            std::cout << "Reset!!!" << std::endl;
        }

        key = cv::waitKey(0);

        if ((key & 0xFF) == 'q') {
            break;
        }

        idx_cur++;
        idx_cout++;
    }
}

int main(int argc, char** argv) {
    test();
    return 0;
}