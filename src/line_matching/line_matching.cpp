#include "line_matching/line_matching.h"

LineMatching::LineMatching(int step,
                           float closest_line_threshold,
                           float line_matching_ratio,
                           float line_distance_error_ratio,
                           float klt_error_threshold) {
    step_ = step;
    closest_line_threshold_ = closest_line_threshold;
    klt_error_threshold_ = klt_error_threshold;
    line_matching_ratio_ = line_matching_ratio;
    line_distance_error_ratio_ = line_distance_error_ratio;

    klt_ = cv::makePtr<KLT>(cv::Size(9, 9), 3, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.001), 1e-4, 0);
}

LineMatching::~LineMatching() {
    klt_.release();
}

float LineMatching::PointLineDistance(const float x, const float y,
                                      const std::array<float, 4> &line_endpoint) {
    float v_x = line_endpoint[2] - line_endpoint[0];
    float v_y = line_endpoint[3] - line_endpoint[1];

    float u_x = line_endpoint[0] - x;
    float u_y = line_endpoint[1] - y;

    float t = -(v_x * u_x + v_y * u_y) / (v_x * v_x + v_y * v_y);

    if (t < 0) {
        t = 0;
    } else if (t > 1) {
        t = 1;
    }

    float d_x = t * v_x + u_x;
    float d_y = t * v_y + u_y;

    return sqrt(d_x * d_x + d_y * d_y);
}

float LineMatching::PointLineDistance(const cv::Point2f &pt,
                                      const std::array<float, 4> &line_endpoint) {
    return PointLineDistance(pt.x, pt.y, line_endpoint);
}

void LineMatching::ClosestLine(const std::vector<cv::Point2f> &kps_cur,
                               const std::vector<uchar> &status,
                               const std::vector<float> &errors,
                               const std::vector<Line> &lines_cur,
                               std::vector<int> &kp2line_cur) {
    int i, j, kps_cur_num, lines_cur_num, min_idx;
    cv::Point2f pt;
    float distance, min_distance;

    kps_cur_num = kps_cur.size();
    lines_cur_num = lines_cur.size();

    kp2line_cur.clear();
    kp2line_cur.assign(kps_cur_num, -1);

    for (i = 0; i < kps_cur_num; i++) {
        if (!status[i] || errors[i] > klt_error_threshold_) continue;

        pt = kps_cur[i];

        min_idx = -1;
        min_distance = 1000000;

        for (j = 0; j < lines_cur_num; j++) {
            distance = PointLineDistance(pt, lines_cur[j].line_endpoint);

            if (distance < min_distance) {
                min_distance = distance;
                min_idx = j;
            }
        }
        if (min_distance < closest_line_threshold_) {
            kp2line_cur[i] = min_idx;
        }
        // PointLineDistance
    }
}

void LineMatching::Point2Line(const std::vector<Line> &lines_ref,
                              const std::vector<Line> &lines_cur,
                              const std::vector<int> &line_kp_num_ref,
                              const std::vector<int> &kp2line_cur,
                              const int lines_num_cur,
                              std::vector<int> &line_ref_to_line_cur) {
    int i, j, kp_num, idx_cur, idx = 0;
    int max_idx = 0, max_value;
    int line_num_ref = line_kp_num_ref.size();

    int *count = new int[lines_num_cur];
    line_ref_to_line_cur.assign(line_num_ref, -1);

    for (i = 0; i < line_num_ref; i++) {
        kp_num = line_kp_num_ref[i];
        // std::cout<<"Line no "<<i<<", num "<<kp_num<<std::endl;
        memset(count, 0, lines_num_cur * sizeof(int));
        for (j = 0; j < kp_num; j++) {
            idx_cur = kp2line_cur[idx];
            if (idx_cur != -1) {
                count[idx_cur]++;
            }
            // std::cout<<"Line "<<i<<" , Point "<<idx<<" Match to Line "<<kp2line_cur[idx]<<std::endl;
            idx++;
        }

        max_value = -1;
        for (j = 0; j < lines_num_cur; j++) {
            if (count[j] > max_value) {
                max_value = count[j];
                max_idx = j;
            }
        }

        if (max_value <= 2 ||
            float(max_value) / kp_num < line_matching_ratio_ ||
            lines_cur[max_idx].length > lines_ref[i].length * line_distance_error_ratio_ ||
            lines_cur[max_idx].length < lines_ref[i].length / line_distance_error_ratio_) {
            continue;
        }

        line_ref_to_line_cur[i] = max_idx;

        // if (max_value != 0)
        // std::cout<<"max_idx:"<<max_idx<<", max_value:"<<max_value<<std::endl;
    }

    delete[] count;
}

void LineMatching::DashLine(cv::Mat &img,
                            const cv::Point2f &p1,
                            const cv::Point2f &p2,
                            const cv::Scalar &bgr,
                            const int interval) {
    cv::LineIterator dash(img, p1, p2, 8);

    for (int i = 0; i < dash.count; i++, dash++) {
        if (i % interval != 0) {
            (*dash)[0] = bgr[0];  // Blue
            (*dash)[1] = bgr[1];  // Green
            (*dash)[2] = bgr[2];  // Red
        }
    }
}

void LineMatching::DashRotatedRect(cv::Mat &img,
                                   const cv::Point2f &center,
                                   const float width,
                                   const float height,
                                   const float angle,
                                   const cv::Scalar &bgr,
                                   const int interval) {
    cv::RotatedRect Rect = cv::RotatedRect(center, cv::Size2f(width, height), angle);
    cv::Point2f vertices[4];  //定义4个点的数组
    Rect.points(vertices);    //将四个点存储到vertices数组中

    for (int i = 0; i < 4; i++) {
        DashLine(img, vertices[i], vertices[(i + 1) % 4], bgr, interval);
    }
}

void LineMatching::LineFilter(std::vector<Line> &lines,
                              float distance_threshold,
                              float parallel_threshold) {
    int i, j, num = lines.size();
    int idx1, idx2;
    float x11, y11, x12, y12;
    float x21, y21, x22, y22;
    float ux, uy, vx, vy;
    float u_dist, v_dist;
    float d1, d2;
    std::vector<size_t> dists_idx(num);

    std::iota(dists_idx.begin(), dists_idx.end(), 0);
    std::sort(dists_idx.begin(), dists_idx.end(),
              [&lines](size_t i1, size_t i2) { return lines[i1].length > lines[i2].length; });

    for (i = 0; i < num; i++) {
        idx1 = dists_idx[i];

        // u_dist = dists[idx1];
        u_dist = lines[idx1].length;

        if (u_dist == -1) continue;

        x11 = lines[idx1].line_endpoint[0];
        y11 = lines[idx1].line_endpoint[1];
        x12 = lines[idx1].line_endpoint[2];
        y12 = lines[idx1].line_endpoint[3];

        ux = x12 - x11;
        uy = y12 - y11;

        for (j = i + 1; j < num; j++) {
            idx2 = dists_idx[j];

            // v_dist = dists[idx2];
            v_dist = lines[idx2].length;

            if (v_dist == -1) continue;

            x21 = lines[idx2].line_endpoint[0];
            y21 = lines[idx2].line_endpoint[1];
            x22 = lines[idx2].line_endpoint[2];
            y22 = lines[idx2].line_endpoint[3];

            vx = x22 - x21;
            vy = y22 - y21;

            if (fabs(ux * vy - vx * uy) > (u_dist * v_dist * parallel_threshold)) continue;

            d1 = PointLineDistance(x21, y21, lines[idx1].line_endpoint);
            d2 = PointLineDistance(x22, y22, lines[idx1].line_endpoint);

            // std::cout<<d1<<","<<d2<<std::endl;

            if (d1 < distance_threshold || d2 < distance_threshold) {
                // dists[idx2] = -1;
                lines[idx2].length = -1;
            }
        }
        // std::cout<<dists[dists_idx[i]]<<std::endl;
    }

    // cv::line(img_show,
    //          cv::Point(x11, y11),
    //          cv::Point(x12, y12),
    //          cv::Scalar(0, 0, 255),
    //          1,
    //          CV_AA);

    // cv::line(img_show,
    //          cv::Point(x21, y21),
    //          cv::Point(x22, y22),
    //          cv::Scalar(255, 0, 0),
    //          1,
    //          CV_AA);

    std::vector<Line> new_lines;
    new_lines.reserve(num);
    for (i = 0; i < num; i++) {
        if (lines[i].length != -1) {
            new_lines.emplace_back(lines[i]);

            // cv::line(img_show,
            //          cv::Point(line_endpoints[i][0], line_endpoints[i][1]),
            //          cv::Point(line_endpoints[i][2], line_endpoints[i][3]),
            //          cv::Scalar(0, 0, 255),
            //          1,
            //          CV_AA);
        }
    }

    // line_endpoints.clear();
    // line_endpoints = new_line_endpoints;
    lines = std::move(new_lines);

    // cv::imshow("img", img_show);
    // cv::waitKey(0);
}

void LineMatching::TopologicalFilter(const std::vector<Line> &lines_ref,
                                     const std::vector<Line> &lines_cur,
                                     const std::vector<int> &line_ref_to_line_cur,
                                     std::vector<bool> &status,
                                     const cv::Mat *img_ref_debug,
                                     const cv::Mat *img_cur_debug,
                                     const float distance_threshold,
                                     const float line_length_change_tolerate_ratio,
                                     const float violation_ratio) {
    int lines_ref_num = lines_ref.size();

    int idx_ref1, idx_ref2, idx_cur1, idx_cur2;
    // int sign_ref, sign_cur;
    bool flag = false;
    float distance_ref, distance_cur;
    int match_num = 0;
    float length_diff_ratio;

    topological_filter_cout_.clear();
    topological_filter_cout_.assign(lines_ref_num, 0);

    for (idx_ref1 = 0; idx_ref1 < lines_ref_num; idx_ref1++) {
        idx_cur1 = line_ref_to_line_cur[idx_ref1];
        if (idx_cur1 != -1) {
            match_num++;
            for (idx_ref2 = 0; idx_ref2 < lines_ref_num; idx_ref2++) {
                if (idx_ref1 != idx_ref2) {
                    idx_cur2 = line_ref_to_line_cur[idx_ref2];
                    if (idx_cur2 != -1) {
                        length_diff_ratio = fabs(lines_ref[idx_ref2].length - lines_cur[idx_cur2].length) / lines_ref[idx_ref2].length;

                        if (length_diff_ratio > line_length_change_tolerate_ratio) continue;

                        // sign_ref = SidenessCheck(lines_ref[idx_ref1], lines_ref[idx_ref2], distance_ref);
                        // sign_cur = SidenessCheck(lines_cur[idx_cur1], lines_cur[idx_cur2], distance_cur);

                        flag = SidenessCheck(lines_ref[idx_ref1], lines_ref[idx_ref2],
                                             lines_cur[idx_cur1], lines_cur[idx_cur2],
                                             distance_ref, distance_cur);

                        // if (sign_ref != sign_cur) {
                        if (!flag &&
                            fabs(distance_ref) > distance_threshold &&
                            fabs(distance_cur) > distance_threshold) {
                            topological_filter_cout_[idx_ref1] += 1;
                            topological_filter_cout_[idx_ref2] += 1;

                            if (img_ref_debug && img_cur_debug) {
                                int height = (*img_ref_debug).rows;
                                int width = (*img_ref_debug).cols;
                                cv::Mat img(height, width * 2, CV_8UC3);

                                cv::cvtColor((*img_ref_debug), img(cv::Rect(0, 0, width, height)), cv::COLOR_GRAY2BGR);
                                cv::cvtColor((*img_cur_debug), img(cv::Rect(width, 0, width, height)), cv::COLOR_GRAY2BGR);

                                std::cout << "Distence ref:" << distance_ref << std::endl;
                                std::cout << "Distence cur:" << distance_cur << std::endl;

                                std::cout << "Equation ref:" << lines_ref[idx_ref1].line_equation[0] << "," << lines_ref[idx_ref1].line_equation[1] << "," << lines_ref[idx_ref1].line_equation[2] << std::endl;
                                std::cout << "center ref:" << lines_ref[idx_ref2].center[0] << "," << lines_ref[idx_ref2].center[1] << std::endl;

                                std::cout << "Equation cur:" << lines_cur[idx_cur1].line_equation[0] << "," << lines_cur[idx_cur1].line_equation[1] << "," << lines_cur[idx_cur1].line_equation[2] << std::endl;
                                std::cout << "center cur:" << lines_cur[idx_cur2].center[0] << "," << lines_cur[idx_cur2].center[1] << std::endl
                                          << std::endl;

                                cv::line(img,
                                         cv::Point(lines_ref[idx_ref1].line_endpoint[0], lines_ref[idx_ref1].line_endpoint[1]),
                                         cv::Point(lines_ref[idx_ref1].line_endpoint[2], lines_ref[idx_ref1].line_endpoint[3]),
                                         cv::Scalar(255, 0, 0),
                                         2,
                                         CV_AA);

                                cv::line(img,
                                         cv::Point(lines_ref[idx_ref2].line_endpoint[0], lines_ref[idx_ref2].line_endpoint[1]),
                                         cv::Point(lines_ref[idx_ref2].line_endpoint[2], lines_ref[idx_ref2].line_endpoint[3]),
                                         cv::Scalar(0, 0, 255),
                                         2,
                                         CV_AA);

                                cv::line(img,
                                         cv::Point(lines_cur[idx_cur1].line_endpoint[0] + width, lines_cur[idx_cur1].line_endpoint[1]),
                                         cv::Point(lines_cur[idx_cur1].line_endpoint[2] + width, lines_cur[idx_cur1].line_endpoint[3]),
                                         cv::Scalar(255, 0, 0),
                                         2,
                                         CV_AA);

                                cv::line(img,
                                         cv::Point(lines_cur[idx_cur2].line_endpoint[0] + width, lines_cur[idx_cur2].line_endpoint[1]),
                                         cv::Point(lines_cur[idx_cur2].line_endpoint[2] + width, lines_cur[idx_cur2].line_endpoint[3]),
                                         cv::Scalar(0, 0, 255),
                                         2,
                                         CV_AA);

                                cv::circle(img, cv::Point(lines_ref[idx_ref2].center[0], lines_ref[idx_ref2].center[1]), 5, cv::Scalar(0, 255, 255), -1);
                                cv::circle(img, cv::Point(lines_cur[idx_cur2].center[0] + width, lines_cur[idx_cur2].center[1]), 5, cv::Scalar(0, 255, 255), -1);

                                cv::imshow("TopologicalFilter", img);
                                cv::waitKey(0);
                            }
                        }
                    }
                }
            }
        }
    }

    // float threshold = 0.15 * match_num;
    float threshold = violation_ratio * (match_num - 1);

    if (threshold < 2) threshold = 2;

    // std::cout << "threshold:" << threshold << std::endl;

    status.clear();
    status.assign(lines_ref_num, true);
    for (idx_ref1 = 0; idx_ref1 < lines_ref_num; idx_ref1++) {
        // std::cout << topological_filter_cout_[idx_ref1] << ", ";
        if (topological_filter_cout_[idx_ref1] > threshold) {
            status[idx_ref1] = false;
        }
    }

    /*
    if (&status) {
        status.assign(lines_ref_num, true);
        for (idx_ref1 = 0; idx_ref1 < lines_ref_num; idx_ref1++) {
            // std::cout << topological_filter_cout_[idx_ref1] << ", ";
            if (topological_filter_cout_[idx_ref1] > threshold) {
                line_ref_to_line_cur[idx_ref1] = -1;
                status[idx_ref1] = false;
            }
        }
    } else {
        for (idx_ref1 = 0; idx_ref1 < lines_ref_num; idx_ref1++) {
            // std::cout << topological_filter_cout_[idx_ref1] << ", ";
            if (topological_filter_cout_[idx_ref1] > threshold) {
                line_ref_to_line_cur[idx_ref1] = -1;
            }
        }
    }
    */

    // std::cout << std::endl;
}

bool LineMatching::SidenessCheck(const Line &l1_ref, const Line &l2_ref,
                                 const Line &l1_cur, const Line &l2_cur,
                                 float &distance1, float &distance2) {
    double a_1 = l1_ref.line_equation[0];
    double b_1 = l1_ref.line_equation[1];
    double c_1 = l1_ref.line_equation[2];

    double px_1 = l2_ref.center[0];
    double py_1 = l2_ref.center[1];

    double a_2 = l1_cur.line_equation[0];
    double b_2 = l1_cur.line_equation[1];
    double c_2 = l1_cur.line_equation[2];

    double px_2 = l2_cur.center[0];
    double py_2 = l2_cur.center[1];

    if ((fabs(a_1 - a_2) + fabs(b_1 - b_2)) > (fabs(a_1 + a_2) + fabs(b_1 + b_2))) {
        a_2 = -a_2;
        b_2 = -b_2;
        c_2 = -c_2;
    }

    distance1 = (px_1 * a_1 + py_1 * b_1 + c_1) / sqrt(a_1 * a_1 + b_1 * b_1);
    distance2 = (px_2 * a_2 + py_2 * b_2 + c_2) / sqrt(a_2 * a_2 + b_2 * b_2);

    // std::cout << a_1 << "," << b_1 << "," << c_1 << std::endl;
    // std::cout << a_2 << "," << b_2 << "," << c_2 << std::endl;
    // std::cout << distance1 << std::endl;
    // std::cout << distance2 << std::endl;

    if (distance1 * distance2 < 0) return false;

    return true;
}

/*
int LineMatching::SidenessCheck(const Line &l1, const Line &l2, float &distance) {
    //if l2 is on the right side of the l1, output 1
    //if l2 if on the left side of the l1, output -1
    //otherwise output 0

    double a_1 = l1.line_equation[0];
    double b_1 = l1.line_equation[1];
    double c_1 = l1.line_equation[2];

    // double a_2 = l2.line_equation[0];
    // double b_2 = l2.line_equation[1];
    // double c_2 = l2.line_equation[2];

    if (a_1 < 0) {
        a_1 = - a_1;
        b_1 = - b_1;
        c_1 = - c_1;
    }

    // double s_x = l1.line_endpoint[0];
    // double s_y = l1.line_endpoint[1];
    // double e_x = l1.line_endpoint[2];
    // double e_y = l1.line_endpoint[3];

    // double px = l2.center[0];
    // double py = l2.center[1];

    // double side = (py - s_y) * (e_x - s_x) - (px - s_x) * (e_y - s_y);

    // std::cout<<"============="<<std::endl;
    // std::cout<<side / l1.length<<std::endl;


    // std::cout<<"============="<<std::endl;
    // std::cout<< (- a * l2.center[0] - c) / b<<std::endl;
    // std::cout<< l2.center[1]<<std::endl;
    // std::cout<<l2.center[0] * a + l2.center[1] * b + c<<std::endl;
    // std::cout<<l2.center[0] * a + l2.center[1] * b + c<<std::endl;
    // std::cout<<"============="<<std::endl;

    // double sign = py + (a * px + c) / b;

    // double slope = - a / b;
    // double sign;
    // if (slope > 1 || slope < -1) {
    //     sign = px + (b * py + c) / a;
    // } else {
    //     sign = py + (a * px + c) / b;
    // }

    // std::cout<<"============="<<std::endl;
    // std::cout<<a<<","<<b<<","<<c<<std::endl;
    // std::cout<<px<<","<<py<<std::endl;
    // std::cout<<"slop:"<<-a/b<<std::endl;
    // std::cout<<(-a * px - c) / b<<std::endl;
    // std::cout<<(-b * py - c) / a<<std::endl;
    // std::cout<<sign<<std::endl;
    // std::cout<<"============="<<std::endl;




    // std::cout<<a<<", "<<b<<", "<<c<<std::endl;
    // std::cout<<-a / b<<std::endl;

    // float sideness = l2.center[0] * a + l2.center[1] * b + c;
    // distance = fabs(sideness) / sqrt(a * a + b * b);
    // if (sideness > 0) return 1;
    // if (sideness < 0) return -1;

    ////////////////

    distance = (l2.center[0] * a_1 + l2.center[1] * b_1 + c_1) / sqrt( a_1 * a_1 + b_1 * b_1);
    // std::cout<<distance<<std::endl;
    // std::cout<<"============="<<std::endl;
    ////////////////

    if (distance > 0) return 1;
    if (distance < 0) return -1;
    return 0;
}
*/

bool LineMatching::Anchors(const std::vector<Line> &lines_ref,
                           const cv::Mat &K_ref,
                           const cv::Mat &K_cur,
                           const cv::Mat &T_cur_ref,
                           std::vector<cv::Point2f> &kps_ref,
                           std::vector<cv::Point2f> &kps_cur,
                           std::vector<cv::Vec2f> &dirs,
                           std::vector<int> &kp2line_ref,
                           std::vector<int> &line_kp_num_ref) {
    int i, j, iter_num;
    float pt_ref_x1, pt_ref_y1, pt_ref_x2, pt_ref_y2;
    float length_ref;
    float pt_ref_x, pt_ref_y;
    float dir_ref_x, dir_ref_y;
    float delta_ref_x, delta_ref_y;
    cv::Mat epip_line;
    cv::Mat kp_ref;
    cv::Mat kp_cur_infinity;
    int lines_ref_num = lines_ref.size();

    kps_ref.clear();
    kps_cur.clear();
    dirs.clear();
    kp2line_ref.clear();
    line_kp_num_ref.clear();

    line_kp_num_ref.resize(lines_ref_num);
    // kps_ref.reserve(10 * lines_ref_num);

    bool has_T = false;
    cv::Mat F;
    cv::Mat K_R_K_inv;
    if (&K_ref != NULL && &K_cur != NULL && &T_cur_ref != NULL &&
        !K_ref.empty() && !K_cur.empty() && !T_cur_ref.empty()) {
        cv::Mat R = T_cur_ref(cv::Rect(0, 0, 3, 3));
        cv::Mat t = T_cur_ref(cv::Rect(3, 0, 1, 3));

        cv::Mat K_T = K_cur * t;
        cv::Mat K_T_skew = (cv::Mat_<float>(3, 3) << 0, -K_T.at<float>(2), K_T.at<float>(1),
                            K_T.at<float>(2), 0, -K_T.at<float>(0),
                            -K_T.at<float>(1), K_T.at<float>(0), 0);
        K_R_K_inv = K_cur * R * K_ref.inv();
        F = K_T_skew * K_R_K_inv;

        // cv::Mat K_inv = K.inv();
        // cv::Mat R = T_cur_ref(cv::Rect(0, 0, 3, 3));
        // cv::Mat t_skew = (cv::Mat_<float>(3, 3) << 0, -T_cur_ref.at<float>(2, 3), T_cur_ref.at<float>(1, 3),
        //                   T_cur_ref.at<float>(2, 3), 0, -T_cur_ref.at<float>(0, 3),
        //                   -T_cur_ref.at<float>(1, 3), T_cur_ref.at<float>(0, 3), 0);
        // F = K_inv.t() * t_skew * R * K_inv;
        has_T = true;
    }

    for (i = 0; i < lines_ref_num; i++) {
        pt_ref_x = pt_ref_x1 = lines_ref[i].line_endpoint[0];
        pt_ref_y = pt_ref_y1 = lines_ref[i].line_endpoint[1];
        pt_ref_x2 = lines_ref[i].line_endpoint[2];
        pt_ref_y2 = lines_ref[i].line_endpoint[3];

        length_ref = lines_ref[i].length;

        dir_ref_x = (pt_ref_x2 - pt_ref_x1) / length_ref;
        dir_ref_y = (pt_ref_y2 - pt_ref_y1) / length_ref;

        delta_ref_x = step_ * dir_ref_x;
        delta_ref_y = step_ * dir_ref_y;

        iter_num = int(length_ref / step_);

        cv::Vec2f dir = cv::Vec2f(-dir_ref_y, dir_ref_x);

        for (j = 0; j <= iter_num; j++) {
            if (has_T) {
                kp_ref = (cv::Mat_<float>(3, 1) << pt_ref_x, pt_ref_y, 1.);
                epip_line = F * kp_ref;
                cv::Vec2f dir_epip = cv::Vec2f(epip_line.at<float>(1), -epip_line.at<float>(0));
                dirs.emplace_back(dir_epip / cv::norm(dir_epip));

                kp_cur_infinity = K_R_K_inv * kp_ref;
                kps_cur.push_back(cv::Point2f(kp_cur_infinity.at<float>(0) / kp_cur_infinity.at<float>(2),
                                              kp_cur_infinity.at<float>(1) / kp_cur_infinity.at<float>(2)));
            } else {
                dirs.emplace_back(dir);
            }

            kps_ref.emplace_back(cv::Point2f(pt_ref_x, pt_ref_y));
            kp2line_ref.emplace_back(i);

            pt_ref_x += delta_ref_x;
            pt_ref_y += delta_ref_y;
        }

        if (has_T) {
            kp_ref = (cv::Mat_<float>(3, 1) << pt_ref_x2, pt_ref_y2, 1.);
            epip_line = F * kp_ref;
            cv::Vec2f dir_epip = cv::Vec2f(epip_line.at<float>(1), -epip_line.at<float>(0));
            dirs.emplace_back(dir_epip / cv::norm(dir_epip));

            kp_cur_infinity = K_R_K_inv * kp_ref;
            kps_cur.push_back(cv::Point2f(kp_cur_infinity.at<float>(0) / kp_cur_infinity.at<float>(2),
                                          kp_cur_infinity.at<float>(1) / kp_cur_infinity.at<float>(2)));

        } else {
            dirs.emplace_back(dir);
        }

        kps_ref.emplace_back(cv::Point2f(pt_ref_x2, pt_ref_y2));
        kp2line_ref.emplace_back(i);

        line_kp_num_ref[i] = iter_num + 2;
    }

    return has_T;
}


bool LineMatching::Matching(const cv::Mat &img_ref,
                            const cv::Mat &img_cur,
                            const std::vector<Line> &lines_ref,
                            const std::vector<Line> &lines_cur,
                            std::vector<int> &line_ref_to_line_cur,
                            const cv::Mat &K_ref,
                            const cv::Mat &K_cur,
                            const cv::Mat &T_cur_ref,
                            const bool illumination_adapt,
                            const bool topological_filter,
                            const int debug_show,
                            const int debug_img_ref_id,
                            const int debug_img_cur_id) {
    int lines_ref_num = lines_ref.size();
    int lines_cur_num = lines_cur.size();

    if (lines_ref_num == 0 || lines_cur_num == 0) return false;

    bool has_T = Anchors(lines_ref,
                         K_ref,
                         K_cur,
                         T_cur_ref,
                         kps_ref_,
                         kps_cur_,
                         dirs_,
                         kp2line_ref_,
                         line_kp_num_ref_);

    if (kps_ref_.size() == 0) return false;

    if (has_T) {
        klt_->setFlags(cv::OPTFLOW_USE_INITIAL_FLOW);
        klt_->setWinSize(cv::Size(9, 9));
    } else {
        klt_->setFlags(0);
        klt_->setWinSize(cv::Size(13, 13));
    }

    // std::vector<cv::Point2f> kps_init;

    if (has_T) {
        kps_init_ = kps_cur_;
    } else {
        kps_init_ = kps_ref_;
    }

    status_.clear();
    errors_.clear();

    if (has_T) {
        klt_->calc1D(img_ref, img_cur, kps_ref_, dirs_, kps_cur_, status_, errors_, illumination_adapt);
    } else {
        klt_->calc2D(img_ref, img_cur, kps_ref_, kps_cur_, status_, errors_, illumination_adapt);
    }

    ClosestLine(kps_cur_,
                status_,
                errors_,
                // line_endpoints_cur,
                lines_cur,
                kp2line_cur_);

    line_ref_to_line_cur.clear();

    Point2Line(lines_ref,
               lines_cur,
               line_kp_num_ref_,
               kp2line_cur_,
               lines_cur_num,
               line_ref_to_line_cur);

    std::vector<int> line_ref_to_line_cur_before_filiter;
    std::vector<bool> topological_filter_status;

    if (topological_filter) {
        line_ref_to_line_cur_before_filiter = line_ref_to_line_cur;
        TopologicalFilter(lines_ref, lines_cur, line_ref_to_line_cur, topological_filter_status);

        for (int i = 0; i < int(line_ref_to_line_cur.size()); i++) {
            if (!topological_filter_status[i]) {
                line_ref_to_line_cur[i] = -1;
            }
        }
    }

    // std::cout<<"debug_show:"<<debug_show<<std::endl;
    if (debug_show > 0) {
        // std::vector<int> line_ref_to_line_cur_before_filiter = line_ref_to_line_cur;
        // std::vector<bool> top_filter_status;
        // TopologicalFilter(lines_ref, lines_cur, line_ref_to_line_cur, &top_filter_status, &img_show_tmp);
        // TopologicalFilter(lines_ref, lines_cur, line_ref_to_line_cur, top_filter_status);

        DebugShow(debug_show,
                  img_ref,
                  img_cur,
                  lines_ref,
                  lines_cur,
                  kps_ref_,
                  kps_cur_,
                  kps_init_,
                  dirs_,
                  status_,
                  kp2line_cur_,
                  line_kp_num_ref_,
                  line_ref_to_line_cur,
                  line_ref_to_line_cur_before_filiter,
                  topological_filter_status,
                  debug_img_ref_id,
                  debug_img_cur_id);

        // cv::Point startPoint;
        // cv::Point endPoint;
        // int width = img_ref.cols;
        // int height = img_ref.rows;
        // cv::Mat img_show(height, width * 2, CV_8UC3);
        // cv::cvtColor(img_ref, img_show(cv::Rect(0, 0, width, height)), cv::COLOR_GRAY2BGR);
        // cv::cvtColor(img_cur, img_show(cv::Rect(width, 0, width, height)), cv::COLOR_GRAY2BGR);
        // for (int j = 0; j < int(kps_init.size()); j++) {
        //     startPoint = cv::Point2f(kps_init[j].x + width, kps_init[j].y);
        //     endPoint = cv::Point2f(kps_init[j].x + dirs_[j][0] * 15 + width, kps_init[j].y + dirs_[j][1] * 15);

        //     cv::line(img_show, startPoint, endPoint, cv::Scalar(0, 0, 255), 1, CV_AA);
        // }
        // cv::imshow("line", img_show);
        // cv::waitKey(0);
    }

    return true;
}

void LineMatching::getPointMatchResult(std::vector<cv::Point2f> &kps_ref,
                                       std::vector<cv::Point2f> &kps_init,
                                       std::vector<cv::Vec2f> &dirs,
                                       std::vector<cv::Point2f> &kps_cur,
                                       std::vector<int> &line_kp_num_ref,
                                       std::vector<int> &kp2line_cur) {
    kps_ref = kps_ref_;
    kps_init = kps_init_;
    dirs = dirs_;
    kps_cur = kps_cur_;
    line_kp_num_ref = line_kp_num_ref_;
    kp2line_cur = kp2line_cur_;
}

void LineMatching::DebugShow(const int &debug_show,
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
                             const int debug_img_cur_id) {
    int lines_ref_num = lines_ref.size();
    int lines_cur_num = lines_cur.size();
    int width = img_ref.cols;
    int height = img_ref.rows;
    cv::Mat img_show(height, width * 2, CV_8UC3);
    cv::Point startPoint;
    cv::Point endPoint;
    srand(0);
    cv::Scalar bgr;
    int match_idx;
    int i, j, idx = 0;
    int lowest = 0, highest = 255;
    int range = (highest - lowest) + 1;
    bool has_top_filter = true;

    if (top_filter_status.size() == 0) has_top_filter = false;

    cv::cvtColor(img_ref, img_show(cv::Rect(0, 0, width, height)), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_cur, img_show(cv::Rect(width, 0, width, height)), cv::COLOR_GRAY2BGR);

    // cv::Mat img_show_tmp = img_show.clone();
    for (i = 0; i < lines_cur_num; i++) {
        startPoint = cv::Point(int(lines_cur[i].line_endpoint[0]) + width, int(lines_cur[i].line_endpoint[1]));
        endPoint = cv::Point(int(lines_cur[i].line_endpoint[2]) + width, int(lines_cur[i].line_endpoint[3]));

        DashLine(img_show, startPoint, endPoint, cv::Scalar(255, 112, 132));

        // bgr = cv::Scalar(lowest + int(rand() % range),
        //                  lowest + int(rand() % range),
        //                  lowest + int(rand() % range));
        // DashLine(img_show, startPoint, endPoint, bgr);
    }

    for (i = 0; i < lines_ref_num; i++) {
        bgr = cv::Scalar(lowest + int(rand() % range),
                         lowest + int(rand() % range),
                         lowest + int(rand() % range));

        startPoint = cv::Point(int(lines_ref[i].line_endpoint[0]), int(lines_ref[i].line_endpoint[1]));
        endPoint = cv::Point(int(lines_ref[i].line_endpoint[2]), int(lines_ref[i].line_endpoint[3]));

        match_idx = line_ref_to_line_cur[i];
        if (match_idx != -1) {
            cv::line(img_show, startPoint, endPoint, bgr, 2, CV_AA);

            startPoint = cv::Point(int(lines_cur[match_idx].line_endpoint[0]) + width, int(lines_cur[match_idx].line_endpoint[1]));
            endPoint = cv::Point(int(lines_cur[match_idx].line_endpoint[2]) + width, int(lines_cur[match_idx].line_endpoint[3]));
            cv::line(img_show, startPoint, endPoint, bgr, 2, CV_AA);
        } else {
            DashLine(img_show, startPoint, endPoint, cv::Scalar(255, 112, 132));
            // DashLine(img_show, startPoint, endPoint, bgr);
        }

        if (has_top_filter && !top_filter_status[i]) {
            match_idx = line_ref_to_line_cur_before_filiter[i];

            DashLine(img_show, cv::Point(lines_ref[i].center[0], lines_ref[i].center[1]), cv::Point(lines_cur[match_idx].center[0] + width, lines_cur[match_idx].center[1]), cv::Scalar(0, 0, 255));

            DashRotatedRect(img_show,
                            cv::Point(lines_ref[i].center[0], lines_ref[i].center[1]),
                            lines_ref[i].length + 4,
                            7,
                            atan(-lines_ref[i].line_equation[0] / lines_ref[i].line_equation[1]) * 57.2957795131,
                            cv::Scalar(0, 0, 255));

            DashRotatedRect(img_show,
                            cv::Point(lines_cur[match_idx].center[0] + width, lines_cur[match_idx].center[1]),
                            lines_cur[match_idx].length + 4,
                            7,
                            atan(-lines_cur[match_idx].line_equation[0] / lines_cur[match_idx].line_equation[1]) * 57.2957795131,
                            cv::Scalar(0, 0, 255));
        }

        if (debug_show > 1) {
            for (j = 0; j < line_kp_num_ref[i]; j++) {
                // std::cout<<j<<std::endl;
                // cv::arrowedLine(img_show, kps_ref_[idx], cv::Point2f(kps_ref_[idx].x + dirs_[idx][0] * 10, kps_ref_[idx].y + dirs_[idx][1] * 10), bgr, 1, CV_AA, 0, 0.1);
                cv::circle(img_show, kps_ref[idx], 2, bgr, -1);

                cv::circle(img_show, cv::Point2f(kps_init[idx].x + width, kps_init[idx].y), 1, bgr, -1);

                if (status[idx]) {
                    DashRotatedRect(img_show,
                                    cv::Point2f(kps_cur[idx].x + width, kps_cur[idx].y),
                                    4,
                                    4,
                                    0,
                                    bgr,
                                    8);
                }

                if (kp2line_cur[idx] != -1) {
                    cv::arrowedLine(img_show, cv::Point2f(kps_init[idx].x + width, kps_init[idx].y), cv::Point2f(kps_cur[idx].x + width, kps_cur[idx].y), bgr, 1, CV_AA, 0, 0.05);
                    cv::circle(img_show, cv::Point2f(kps_cur[idx].x + width, kps_cur[idx].y), 2, bgr, 1);
                } else {
                    DashLine(img_show, cv::Point2f(kps_init[idx].x + width, kps_init[idx].y), cv::Point2f(kps_init[idx].x + dirs[idx][0] * 15 + width, kps_init[idx].y + dirs[idx][1] * 15), bgr, 2);
                }

                idx++;
            }
        }
    }

    int match_cout = 0;
    for (int i = 0; i < int(lines_ref.size()); i++) {
        if (line_ref_to_line_cur[i] != -1) match_cout++;
    }

    cv::addWeighted(img_show(cv::Rect(0, 0, width * 2, height * 0.06)), 0.3,
                    cv::Mat::zeros(height * 0.06, width * 2, CV_8UC3), 0.7, 0.0,
                    img_show(cv::Rect(0, 0, width * 2, height * 0.06)));

    cv::putText(img_show, "Ref. Img No." + std::to_string(debug_img_ref_id) + ", Line num: " + std::to_string(lines_ref.size()), cv::Point(10, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 255, 255), 1.8, 8, 0);
    cv::putText(img_show, "Cur. Img No." + std::to_string(debug_img_cur_id) + ", Line num: " + std::to_string(lines_cur.size()) + ", Match num: " + std::to_string(match_cout), cv::Point(width + 10, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 255, 255), 1.8, 8, 0);

    // std::cout<<debug_img_ref_id<<std::endl;
    // std::cout<<debug_img_cur_id<<std::endl;
    cv::imshow("Line Matching", img_show);
    // cv::imwrite("result.png", img_show);
    // cv::waitKey(0);
}