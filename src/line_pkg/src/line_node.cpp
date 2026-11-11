#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include "tnn/core/tnn.h"

static void onTrackbar(int, void*) {}

class LineNode : public rclcpp::Node {
public:
    LineNode() : Node("line_node"), mission_state_(1), new_frame_received_(false)
    {
        param("image_topic",           image_topic_,           std::string("/image_jpeg/compressed"));
        param("offset_topic",          offset_topic_,          std::string("/line_offset"));
        param("mission_topic",         mission_topic_,         std::string("/mission_state"));
        param("yellow_detected_topic", yellow_detected_topic_, std::string("/yellow_detected"));
        param("line_image_bev_topic",  line_image_bev_topic_,  std::string("/line/image/bev"));
        param("line_image_sliding_topic", line_image_sliding_topic_, std::string("/line/image/sliding"));
        param("show_debug",            show_debug_,            false);
        param("debug_mode",            debug_mode_,            false);
        param("debug_scale",           debug_scale_,           1.0);
        param("publish_debug_images",  publish_debug_images_,  true);
        param("debug_image_quality",   debug_image_quality_,   80);

        // 슬라이딩 윈도우 파라미터
        param("n_windows",           n_windows_,           7);
        param("win_half_width",      win_half_width_,      50);
        param("min_pix",             min_pix_,             70);
        param("bottom_lane_windows", bottom_lane_windows_, 1);

        // 초기 탐색 영역 범위 (감지 전까지만 적용, 0.0~1.0)
        param("left_search_start",  left_search_start_,  0.0);
        param("left_search_end",    left_search_end_,    0.35);
        param("right_search_start", right_search_start_, 0.65);
        param("right_search_end",   right_search_end_,   1.0);

        // offset 기준 높이
        param("offset_height_ratio", offset_height_ratio_, 0.7);

        // 노란색/흰색 픽셀 임계값 (제어팀에 bool 전달용)
        param("yellow_detect_thresh", yellow_detect_thresh_, 500);

        // 노란색 검출 ROI (원본 이미지 기준 직사각형, 0이면 전체 이미지 사용)
        param("yellow_roi_x",      yellow_roi_x_,      0);
        param("yellow_roi_y",      yellow_roi_y_,      0);
        param("yellow_roi_width",  yellow_roi_width_,  0);
        param("yellow_roi_height", yellow_roi_height_, 0);

        // BEV 변환 좌표 (src: 원본 이미지 4점, dst: 변환 후 4점)
        // bl=하단왼쪽, br=하단오른쪽, tl=상단왼쪽, tr=상단오른쪽
        param("bev_src_bl_x", bev_src_bl_x_, 0.0);
        param("bev_src_bl_y", bev_src_bl_y_, 118.0);
        param("bev_src_br_x", bev_src_br_x_, 320.0);
        param("bev_src_br_y", bev_src_br_y_, 118.0);
        param("bev_src_tl_x", bev_src_tl_x_, 90.0);
        param("bev_src_tl_y", bev_src_tl_y_, 80.0);
        param("bev_src_tr_x", bev_src_tr_x_, 230.0);
        param("bev_src_tr_y", bev_src_tr_y_, 80.0);

        param("bev_dst_bl_x", bev_dst_bl_x_, 75.0);
        param("bev_dst_bl_y", bev_dst_bl_y_, 120.0);
        param("bev_dst_br_x", bev_dst_br_x_, 245.0);
        param("bev_dst_br_y", bev_dst_br_y_, 120.0);
        param("bev_dst_tl_x", bev_dst_tl_x_, 75.0);
        param("bev_dst_tl_y", bev_dst_tl_y_, 0.0);
        param("bev_dst_tr_x", bev_dst_tr_x_, 245.0);
        param("bev_dst_tr_y", bev_dst_tr_y_, 0.0);

        // HSV 흰색
        param("white_h_min", white_h_min_, 0);
        param("white_h_max", white_h_max_, 180);
        param("white_s_min", white_s_min_, 0);
        param("white_s_max", white_s_max_, 55);
        param("white_v_min", white_v_min_, 150);
        param("white_v_max", white_v_max_, 255);

        // HSV 노란색
        param("yellow_h_min", yellow_h_min_, 15);
        param("yellow_h_max", yellow_h_max_, 30);
        param("yellow_s_min", yellow_s_min_, 70);
        param("yellow_s_max", yellow_s_max_, 255);
        param("yellow_v_min", yellow_v_min_, 120);
        param("yellow_v_max", yellow_v_max_, 255);

        // UFLD Parameters
        param("ufld_proto_path", ufld_proto_path_, std::string("/home/topst/ufld_deploy/models/ufld_dracer_160x320_fp16.opt.tnnproto"));
        param("ufld_model_path", ufld_model_path_, std::string("/home/topst/ufld_deploy/models/ufld_dracer_160x320_fp16.opt.tnnmodel"));
        param("ufld_cls_num_per_lane", ufld_cls_num_per_lane_, 56);
        param("ufld_use_gpu",    ufld_use_gpu_,    true);
        param("ufld_enabled",    ufld_enabled_,    true);
        param("ufld_viz_enabled", ufld_viz_enabled_, true);
        param("ufld_viz_topic",   ufld_viz_topic_,   std::string("/ufld/image/compressed"));
        param("ufld_viz_quality", ufld_viz_quality_, 80);
        param("ufld_direct_offset_enabled", ufld_direct_offset_enabled_, true);
        param("ufld_offset_y_ratio", ufld_offset_y_ratio_, 0.78);
        param("ufld_offset_band_ratio", ufld_offset_band_ratio_, 0.12);
        param("ufld_expected_lane_width_px", ufld_expected_lane_width_px_, 170.0);
        param("ufld_offset_filter_alpha", ufld_offset_filter_alpha_, 0.35);
        param("ufld_offset_max_step_px", ufld_offset_max_step_px_, 35.0);

        if (ufld_enabled_) {
            RCLCPP_INFO(get_logger(), "Initializing UFLD TNN from %s...", ufld_proto_path_.c_str());
            if (!initUFLD()) {
                RCLCPP_ERROR(get_logger(), "Failed to initialize UFLD TNN! Falling back to HSV.");
                ufld_enabled_ = false;
            } else {
                RCLCPP_INFO(get_logger(), "UFLD TNN initialized successfully on %s.", ufld_use_gpu_ ? "GPU (OpenCL)" : "CPU");
            }
        }

        image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(

            image_topic_, 1, [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) { imageCallback(msg); });
        mission_sub_ = create_subscription<std_msgs::msg::Int32>(
            mission_topic_, 10, [this](const std_msgs::msg::Int32::SharedPtr msg) { missionCallback(msg); });
        line_tuning_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            "/line_tuning", 10, [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                // [0-5]  : white HSV  (h_min, h_max, s_min, s_max, v_min, v_max)
                // [6-11] : yellow HSV (h_min, h_max, s_min, s_max, v_min, v_max)
                // [12-19]: bev_src    (bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y)
                // [20-27]: bev_dst    (bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y)
                // [28-30]: ufld direct offset (offset_y_ratio, expected_lane_width_px, filter_alpha)
                // [31-32]: ufld direct offset stabilizer (band_ratio, max_step_px)
                if (msg->data.size() >= 20) {
                    white_h_min_ = static_cast<int>(msg->data[0]);
                    white_h_max_ = static_cast<int>(msg->data[1]);
                    white_s_min_ = static_cast<int>(msg->data[2]);
                    white_s_max_ = static_cast<int>(msg->data[3]);
                    white_v_min_ = static_cast<int>(msg->data[4]);
                    white_v_max_ = static_cast<int>(msg->data[5]);
                    yellow_h_min_ = static_cast<int>(msg->data[6]);
                    yellow_h_max_ = static_cast<int>(msg->data[7]);
                    yellow_s_min_ = static_cast<int>(msg->data[8]);
                    yellow_s_max_ = static_cast<int>(msg->data[9]);
                    yellow_v_min_ = static_cast<int>(msg->data[10]);
                    yellow_v_max_ = static_cast<int>(msg->data[11]);
                    bev_src_bl_x_ = msg->data[12];
                    bev_src_bl_y_ = msg->data[13];
                    bev_src_br_x_ = msg->data[14];
                    bev_src_br_y_ = msg->data[15];
                    bev_src_tl_x_ = msg->data[16];
                    bev_src_tl_y_ = msg->data[17];
                    bev_src_tr_x_ = msg->data[18];
                    bev_src_tr_y_ = msg->data[19];
                    if (msg->data.size() >= 28) {
                        bev_dst_bl_x_ = msg->data[20];
                        bev_dst_bl_y_ = msg->data[21];
                        bev_dst_br_x_ = msg->data[22];
                        bev_dst_br_y_ = msg->data[23];
                        bev_dst_tl_x_ = msg->data[24];
                        bev_dst_tl_y_ = msg->data[25];
                        bev_dst_tr_x_ = msg->data[26];
                        bev_dst_tr_y_ = msg->data[27];
                    }
                    if (msg->data.size() >= 31) {
                        ufld_offset_y_ratio_ = std::max(0.1, std::min(0.98, static_cast<double>(msg->data[28])));
                        ufld_expected_lane_width_px_ = std::max(20.0, std::min(640.0, static_cast<double>(msg->data[29])));
                        ufld_offset_filter_alpha_ = std::max(0.01, std::min(1.0, static_cast<double>(msg->data[30])));
                    }
                    if (msg->data.size() >= 33) {
                        ufld_offset_band_ratio_ = std::max(0.0, std::min(0.5, static_cast<double>(msg->data[31])));
                        ufld_offset_max_step_px_ = std::max(0.0, std::min(320.0, static_cast<double>(msg->data[32])));
                    }
                    computeBirdEyeMatrix();
                }
            });
        offset_pub_ = create_publisher<std_msgs::msg::Float32>(offset_topic_, 10);
        yellow_detected_pub_ = create_publisher<std_msgs::msg::Bool>(yellow_detected_topic_, 10);
        line_bev_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(line_image_bev_topic_, 1);
        line_sliding_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(line_image_sliding_topic_, 1);
        if (ufld_viz_enabled_) {
            ufld_viz_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(ufld_viz_topic_, 1);
        }

        computeBirdEyeMatrix();

        if (show_debug_ && debug_mode_) {
            cv::namedWindow("Trackbar - White",  cv::WINDOW_NORMAL);
            cv::namedWindow("Trackbar - Yellow", cv::WINDOW_NORMAL);
            cv::namedWindow("01 Original",         cv::WINDOW_NORMAL);
            cv::namedWindow("02 HSV Mask",         cv::WINDOW_NORMAL);
            cv::namedWindow("03 ROI",              cv::WINDOW_NORMAL);
            cv::namedWindow("04 BEV",              cv::WINDOW_NORMAL);
            cv::namedWindow("05 Sliding Window",   cv::WINDOW_NORMAL);
            cv::namedWindow("06 Yellow Threshold", cv::WINDOW_NORMAL);

            createTrackbars();
        }

        timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() { run(); });

        RCLCPP_INFO(get_logger(), "[line_node] Started. state management by control_pkg.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr line_tuning_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr yellow_detected_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr ufld_viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr line_bev_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr line_sliding_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string image_topic_, offset_topic_, mission_topic_;
    std::string yellow_detected_topic_;
    std::string line_image_bev_topic_, line_image_sliding_topic_;
    cv::Mat frame_;
    int     mission_state_;
    bool    show_debug_;
    bool    debug_mode_;
    bool    publish_debug_images_;
    double  debug_scale_;
    int     debug_image_quality_;

    int    n_windows_, win_half_width_, min_pix_;
    int    bottom_lane_windows_;
    double offset_height_ratio_;
    double left_search_start_, left_search_end_;
    double right_search_start_, right_search_end_;

    int yellow_detect_thresh_;
    int yellow_roi_x_, yellow_roi_y_, yellow_roi_width_, yellow_roi_height_;

    int white_h_min_, white_h_max_;
    int white_s_min_, white_s_max_;
    int white_v_min_, white_v_max_;
    int yellow_h_min_, yellow_h_max_;
    int yellow_s_min_, yellow_s_max_;
    int yellow_v_min_, yellow_v_max_;

    double bev_src_bl_x_, bev_src_bl_y_, bev_src_br_x_, bev_src_br_y_;
    double bev_src_tl_x_, bev_src_tl_y_, bev_src_tr_x_, bev_src_tr_y_;
    double bev_dst_bl_x_, bev_dst_bl_y_, bev_dst_br_x_, bev_dst_br_y_;
    double bev_dst_tl_x_, bev_dst_tl_y_, bev_dst_tr_x_, bev_dst_tr_y_;

    cv::Mat M_;

    // UFLD Member Variables
    bool ufld_enabled_;
    bool ufld_use_gpu_;
    std::string ufld_proto_path_;
    std::string ufld_model_path_;
    std::string ufld_viz_topic_;
    int ufld_cls_num_per_lane_;
    int ufld_viz_quality_;
    bool ufld_viz_enabled_;
    bool ufld_direct_offset_enabled_;
    double ufld_offset_y_ratio_;
    double ufld_offset_band_ratio_;
    double ufld_expected_lane_width_px_;
    double ufld_offset_filter_alpha_;
    double ufld_offset_max_step_px_;
    bool ufld_direct_offset_valid_ = false;
    float ufld_direct_offset_ = 0.0f;
    bool ufld_direct_center_initialized_ = false;
    double ufld_direct_lane_center_ = 0.0;
    int ufld_direct_sample_y_ = 0;
    std::shared_ptr<TNN_NS::TNN> tnn_;
    std::shared_ptr<TNN_NS::Instance> instance_;
    bool new_frame_received_;

    template <typename T>
    void param(const std::string& name, T& value, const T& default_value) {
        value = declare_parameter<T>(name, default_value);
    }

    void computeBirdEyeMatrix() {
        cv::Point2f src[4] = {
            {(float)bev_src_bl_x_, (float)bev_src_bl_y_},
            {(float)bev_src_br_x_, (float)bev_src_br_y_},
            {(float)bev_src_tl_x_, (float)bev_src_tl_y_},
            {(float)bev_src_tr_x_, (float)bev_src_tr_y_}
        };
        cv::Point2f dst[4] = {
            {(float)bev_dst_bl_x_, (float)bev_dst_bl_y_},
            {(float)bev_dst_br_x_, (float)bev_dst_br_y_},
            {(float)bev_dst_tl_x_, (float)bev_dst_tl_y_},
            {(float)bev_dst_tr_x_, (float)bev_dst_tr_y_}
        };
        M_ = cv::getPerspectiveTransform(src, dst);
    }

    void createTrackbars() {
        cv::createTrackbar("H min", "Trackbar - White", &white_h_min_, 180, onTrackbar);
        cv::createTrackbar("H max", "Trackbar - White", &white_h_max_, 180, onTrackbar);
        cv::createTrackbar("S min", "Trackbar - White", &white_s_min_, 255, onTrackbar);
        cv::createTrackbar("S max", "Trackbar - White", &white_s_max_, 255, onTrackbar);
        cv::createTrackbar("V min", "Trackbar - White", &white_v_min_, 255, onTrackbar);
        cv::createTrackbar("V max", "Trackbar - White", &white_v_max_, 255, onTrackbar);

        cv::createTrackbar("H min", "Trackbar - Yellow", &yellow_h_min_, 180, onTrackbar);
        cv::createTrackbar("H max", "Trackbar - Yellow", &yellow_h_max_, 180, onTrackbar);
        cv::createTrackbar("S min", "Trackbar - Yellow", &yellow_s_min_, 255, onTrackbar);
        cv::createTrackbar("S max", "Trackbar - Yellow", &yellow_s_max_, 255, onTrackbar);
        cv::createTrackbar("V min", "Trackbar - Yellow", &yellow_v_min_, 255, onTrackbar);
        cv::createTrackbar("V max", "Trackbar - Yellow", &yellow_v_max_, 255, onTrackbar);
    }

    cv::Mat scaleForDebug(const cv::Mat& image) {
        if (debug_scale_ <= 0.0 || debug_scale_ == 1.0) {
            return image;
        }

        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(), debug_scale_, debug_scale_, cv::INTER_NEAREST);
        return scaled;
    }

    void showDebugImage(const std::string& name, const cv::Mat& image) {
        cv::imshow(name, scaleForDebug(image));
    }

    cv::Mat maskToBgr(const cv::Mat& mask) {
        cv::Mat bgr;
        cv::cvtColor(mask, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    void publishCompressedImage(
        const rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr& publisher,
        const cv::Mat& image,
        const std::string& frame_id) {
        if (!publisher || publisher->get_subscription_count() == 0 || image.empty()) {
            return;
        }

        std::vector<uchar> encoded;
        std::vector<int> params = {
            cv::IMWRITE_JPEG_QUALITY,
            std::max(1, std::min(100, debug_image_quality_))
        };
        if (!cv::imencode(".jpg", image, encoded, params)) {
            return;
        }

        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = now();
        msg.header.frame_id = frame_id;
        msg.format = "jpeg";
        msg.data.assign(encoded.begin(), encoded.end());
        publisher->publish(msg);
    }

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            frame_ = cv::imdecode(msg->data, cv::IMREAD_COLOR);
            new_frame_received_ = true;
        } catch (const cv::Exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "[line_node] Image decode error: %s", e.what());
        }
    }

    void missionCallback(const std_msgs::msg::Int32::SharedPtr msg) {
        mission_state_ = msg->data;
    }

    // ── state별 흰색/노란색 ON/OFF ────────────────
    // state 0          : 상관없음 (정지)
    // state 1          : 흰색만
    // state 2,3,4,5    : 노란색만
    // state 6          : 흰색+노란색
    // state 7,8,9      : 흰색만
    // state 10         : 상관없음 (정지)
    cv::Mat buildLaneMask(const cv::Mat& hsv) {
        cv::Mat mask_white, mask_yellow, result;

        bool use_white  = (mission_state_ == 1 ||
                           mission_state_ == 6 ||
                          (mission_state_ >= 7 && mission_state_ <= 9));
        bool use_yellow = (mission_state_ >= 2 && mission_state_ <= 6);

        if (use_white) {
            cv::inRange(hsv,
                cv::Scalar(white_h_min_, white_s_min_, white_v_min_),
                cv::Scalar(white_h_max_, white_s_max_, white_v_max_),
                mask_white);
        }
        if (use_yellow) {
            cv::inRange(hsv,
                cv::Scalar(yellow_h_min_, yellow_s_min_, yellow_v_min_),
                cv::Scalar(yellow_h_max_, yellow_s_max_, yellow_v_max_),
                mask_yellow);
        }

        if (use_white && use_yellow)  cv::bitwise_or(mask_white, mask_yellow, result);
        else if (use_white)           result = mask_white;
        else if (use_yellow)          result = mask_yellow;
        else                          result = cv::Mat::zeros(hsv.size(), CV_8UC1);
        return result;
    }


    float slidingWindowOffset(const cv::Mat& binary, cv::Mat& vis_out) {
        const bool need_visual = debug_mode_ || publish_debug_images_;
        if (need_visual) {
            cv::cvtColor(binary, vis_out, cv::COLOR_GRAY2BGR);
        }

        int img_h = binary.rows;
        int img_w = binary.cols;
        int offset_y = static_cast<int>(img_h * offset_height_ratio_);

        cv::Mat lower = binary(cv::Range(img_h / 2, img_h), cv::Range::all());
        cv::Mat hist;
        cv::reduce(lower, hist, 0, cv::REDUCE_SUM, CV_32S);
        int mid = img_w / 2;
        int* hdata = reinterpret_cast<int*>(hist.data);
        int lbase = static_cast<int>(std::distance(hdata, std::max_element(hdata, hdata + mid)));
        int rbase = static_cast<int>(std::distance(hdata, std::max_element(hdata + mid, hdata + img_w)));

        int win_h = std::max(1, img_h / std::max(1, n_windows_));
        int lcur = lbase, rcur = rbase;

        const int left_x0  = static_cast<int>(img_w * left_search_start_);
        const int left_x1  = static_cast<int>(img_w * left_search_end_);
        const int right_x0 = static_cast<int>(img_w * right_search_start_);
        const int right_x1 = static_cast<int>(img_w * right_search_end_);
        bool left_detected  = false;
        bool right_detected = false;

        const int deadzone_margin = static_cast<int>(img_w * 0.05);
        const int left_wall  = mid - deadzone_margin;
        const int right_wall = mid + deadzone_margin;

        if (need_visual) {
            cv::line(vis_out, cv::Point(left_wall,  0), cv::Point(left_wall,  img_h), cv::Scalar(0, 0, 150), 1);
            cv::line(vis_out, cv::Point(right_wall, 0), cv::Point(right_wall, img_h), cv::Scalar(0, 0, 150), 1);
        }

        bool left_enabled  = true;
        bool right_enabled = true;
        bool left_bottom_seen  = false;
        bool right_bottom_seen = false;
        const int bottom_limit = std::max(0, std::min(bottom_lane_windows_, n_windows_));

        bool   lx_found = false, rx_found = false;
        double lx_at_offset = 0.0, rx_at_offset = 0.0;

        for (int win = 0; win < n_windows_; ++win) {
            int ylow  = std::max(0, img_h - (win + 1) * win_h);
            int yhigh = std::min(img_h, img_h - win * win_h);
            if (ylow >= yhigh) continue;

            int lx0 = left_detected  ? std::max(0,       lcur - win_half_width_)
                                     : std::max(left_x0,  lcur - win_half_width_);
            int lx1 = left_detected  ? std::min(img_w,   lcur + win_half_width_)
                                     : std::min(std::min(left_x1, left_wall), lcur + win_half_width_);
            int rx0 = right_detected ? std::max(0,        rcur - win_half_width_)
                                     : std::max(std::max(right_x0, right_wall), rcur - win_half_width_);
            int rx1 = right_detected ? std::min(img_w,   rcur + win_half_width_)
                                     : std::min(right_x1, rcur + win_half_width_);

            if (need_visual) {
                cv::rectangle(vis_out, cv::Point(lx0, ylow), cv::Point(lx1, yhigh), cv::Scalar(0, 255, 0), 1);
                cv::rectangle(vis_out, cv::Point(rx0, ylow), cv::Point(rx1, yhigh), cv::Scalar(0, 255, 0), 1);
            }

            cv::Mat win_roi = binary(cv::Range(ylow, yhigh), cv::Range::all());
            if (cv::countNonZero(win_roi) == 0) continue;

            std::vector<cv::Point> nz;
            cv::findNonZero(win_roi, nz);

            std::vector<cv::Point> gl, gr;
            for (auto& p : nz) {
                if (left_enabled  && p.x >= lx0 && p.x < lx1) gl.push_back(p);
                if (right_enabled && p.x >= rx0 && p.x < rx1) gr.push_back(p);
            }

            if (need_visual) {
                for (auto& p : gl) vis_out.at<cv::Vec3b>(p.y + ylow, p.x) = cv::Vec3b(0, 0, 255);
                for (auto& p : gr) vis_out.at<cv::Vec3b>(p.y + ylow, p.x) = cv::Vec3b(255, 0, 0);
            }

            bool left_valid = false, right_valid = false;

            if (gl.size() > static_cast<size_t>(min_pix_)) {
                lcur = std::accumulate(gl.begin(), gl.end(), 0,
                    [](int s, const cv::Point& p){ return s + p.x; }) / (int)gl.size();
                left_valid    = true;
                left_detected = true;
            }
            if (gr.size() > static_cast<size_t>(min_pix_)) {
                rcur = std::accumulate(gr.begin(), gr.end(), 0,
                    [](int s, const cv::Point& p){ return s + p.x; }) / (int)gr.size();
                right_valid    = true;
                right_detected = true;
            }

            if (win < bottom_limit) {
                left_bottom_seen  = left_bottom_seen  || left_valid;
                right_bottom_seen = right_bottom_seen || right_valid;
            }
            if (bottom_limit > 0 && win == bottom_limit - 1) {
                if (!left_bottom_seen)  left_enabled  = false;
                if (!right_bottom_seen) right_enabled = false;
            }

            if (offset_y >= ylow && offset_y < yhigh) {
                lx_at_offset = static_cast<double>(lcur);
                rx_at_offset = static_cast<double>(rcur);
                lx_found = left_valid;
                rx_found = right_valid;
            }
        }

        if (need_visual) {
            cv::line(vis_out, cv::Point(0, offset_y), cv::Point(img_w, offset_y), cv::Scalar(0, 255, 255), 1);
        }

        double image_center = img_w / 2.0;
        double lane_center  = 0.0;
        float  offset       = 0.0f;

        if (lx_found && rx_found) {
            lane_center = (lx_at_offset + rx_at_offset) / 2.0;
            offset = static_cast<float>(lane_center - image_center);
        } else if (lx_found) {
            lane_center = lx_at_offset + win_half_width_;
            offset = static_cast<float>(lane_center - image_center);
        } else if (rx_found) {
            lane_center = rx_at_offset - win_half_width_;
            offset = static_cast<float>(lane_center - image_center);
        }

        if (need_visual) {
            cv::circle(vis_out, cv::Point(static_cast<int>(image_center), offset_y), 5, cv::Scalar(255, 255, 0), -1);
            if (lane_center > 0)
                cv::circle(vis_out, cv::Point(static_cast<int>(lane_center), offset_y), 5, cv::Scalar(0, 165, 255), -1);
        }

        return offset;
    }

    bool sampleLaneXAtY(const std::vector<cv::Point>& pts, int target_y, double& out_x) {
        if (pts.empty()) return false;

        std::vector<cv::Point> sorted = pts;
        std::sort(sorted.begin(), sorted.end(),
                  [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const cv::Point& p0 = sorted[i];
            const cv::Point& p1 = sorted[i + 1];
            const int y_min = std::min(p0.y, p1.y);
            const int y_max = std::max(p0.y, p1.y);
            if (target_y >= y_min && target_y <= y_max && p0.y != p1.y) {
                double t = static_cast<double>(target_y - p0.y) / static_cast<double>(p1.y - p0.y);
                out_x = p0.x + t * static_cast<double>(p1.x - p0.x);
                return true;
            }
        }

        const cv::Point* nearest = &sorted.front();
        int best_dy = std::abs(sorted.front().y - target_y);
        for (const auto& p : sorted) {
            int dy = std::abs(p.y - target_y);
            if (dy < best_dy) {
                best_dy = dy;
                nearest = &p;
            }
        }

        const int max_dy = std::max(12, frame_.rows / 6);
        if (best_dy > max_dy) return false;
        out_x = nearest->x;
        return true;
    }

    void computeUFLDDirectOffset(const std::vector<std::vector<cv::Point>>& lane_points,
                                 const std::vector<int>& active_classes,
                                 cv::Mat& viz) {
        ufld_direct_offset_valid_ = false;
        if (!ufld_direct_offset_enabled_ || frame_.empty()) return;

        const int img_w = frame_.cols;
        const int img_h = frame_.rows;
        const double image_center = img_w / 2.0;
        const int target_y = std::max(0, std::min(img_h - 1, static_cast<int>(img_h * ufld_offset_y_ratio_)));
        ufld_direct_sample_y_ = target_y;
        const int band_half = static_cast<int>(img_h * ufld_offset_band_ratio_ * 0.5);
        const int y_min = std::max(0, target_y - band_half);
        const int y_max = std::min(img_h - 1, target_y + band_half);

        std::vector<double> left_xs;
        std::vector<double> right_xs;
        for (int lane_idx : active_classes) {
            if (lane_idx < 0 || lane_idx >= static_cast<int>(lane_points.size())) continue;

            std::vector<double> lane_samples;
            for (const auto& p : lane_points[lane_idx]) {
                if (p.y >= y_min && p.y <= y_max) {
                    lane_samples.push_back(static_cast<double>(p.x));
                }
            }
            if (lane_samples.empty()) {
                double x = 0.0;
                if (!sampleLaneXAtY(lane_points[lane_idx], target_y, x)) continue;
                lane_samples.push_back(x);
            }

            if (lane_idx == 0 || lane_idx == 1) {
                left_xs.insert(left_xs.end(), lane_samples.begin(), lane_samples.end());
            } else {
                right_xs.insert(right_xs.end(), lane_samples.begin(), lane_samples.end());
            }
        }

        const auto robustMean = [](std::vector<double> values) {
            std::sort(values.begin(), values.end());
            if (values.size() >= 5) {
                const size_t trim = values.size() / 5;
                values.erase(values.begin(), values.begin() + trim);
                values.erase(values.end() - trim, values.end());
            }
            return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
        };

        bool have_left = !left_xs.empty();
        bool have_right = !right_xs.empty();
        if (!have_left && !have_right) return;

        double lane_center = image_center;
        if (have_left && have_right) {
            lane_center = (robustMean(left_xs) + robustMean(right_xs)) * 0.5;
        } else if (have_left) {
            lane_center = robustMean(left_xs) + ufld_expected_lane_width_px_ * 0.5;
        } else {
            lane_center = robustMean(right_xs) - ufld_expected_lane_width_px_ * 0.5;
        }

        if (ufld_direct_center_initialized_ && ufld_offset_max_step_px_ > 0.0) {
            const double delta = lane_center - ufld_direct_lane_center_;
            if (std::fabs(delta) > ufld_offset_max_step_px_) {
                lane_center = ufld_direct_lane_center_ + std::copysign(ufld_offset_max_step_px_, delta);
            }
        }

        ufld_direct_lane_center_ = lane_center;
        ufld_direct_center_initialized_ = true;
        ufld_direct_offset_ = static_cast<float>(lane_center - image_center);
        ufld_direct_offset_valid_ = true;

        if (!viz.empty()) {
            cv::rectangle(viz, cv::Point(0, y_min), cv::Point(img_w - 1, y_max), cv::Scalar(80, 80, 80), 1);
            cv::line(viz, cv::Point(0, target_y), cv::Point(img_w, target_y), cv::Scalar(255, 255, 0), 1);
            cv::circle(viz, cv::Point(static_cast<int>(image_center), target_y), 5, cv::Scalar(255, 255, 0), -1);
            cv::circle(viz, cv::Point(static_cast<int>(lane_center), target_y), 5, cv::Scalar(0, 165, 255), -1);
            char offset_text[160];
            std::snprintf(offset_text, sizeof(offset_text), "direct offset %.1f y=%.2f band=%.2f w=%.0f",
                          ufld_direct_offset_, ufld_offset_y_ratio_,
                          ufld_offset_band_ratio_, ufld_expected_lane_width_px_);
            cv::putText(viz, offset_text, cv::Point(10, 108), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 255, 255), 1);
        }
    }

    float filterOffset(float measured_offset) {
        static float ema_offset = 0.0f;
        const double alpha = (ufld_enabled_ && ufld_direct_offset_valid_)
            ? ufld_offset_filter_alpha_
            : 0.20;
        ema_offset = static_cast<float>(ema_offset * (1.0 - alpha) + measured_offset * alpha);
        return ema_offset;
    }

public:
    void run() {
        if (!new_frame_received_ || frame_.empty()) return;
        new_frame_received_ = false;

        cv::Mat hsv;
        cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);

        // ── 색깔 검출 (state 1에서만 노란색 감지 → 제어팀에 bool 전달) ──
        cv::Mat mask_yellow;
        cv::inRange(hsv,
            cv::Scalar(yellow_h_min_, yellow_s_min_, yellow_v_min_),
            cv::Scalar(yellow_h_max_, yellow_s_max_, yellow_v_max_), mask_yellow);

        // ROI 적용 (yellow_roi_width_>0 이면 ROI 사용, 아니면 전체 이미지)
        int yellow_px = 0;
        cv::Mat mask_yellow_roi = mask_yellow;
        cv::Mat roi_mask_vis;
        if (debug_mode_) {
            roi_mask_vis = cv::Mat::zeros(mask_yellow.size(), mask_yellow.type());
        }
        if (yellow_roi_width_ > 0 && yellow_roi_height_ > 0) {
            int x = std::max(0, yellow_roi_x_);
            int y = std::max(0, yellow_roi_y_);
            int w = std::min(yellow_roi_width_,  frame_.cols - x);
            int h = std::min(yellow_roi_height_, frame_.rows - y);
            if (w > 0 && h > 0) {
                cv::Rect roi(x, y, w, h);
                mask_yellow_roi = mask_yellow(roi);
                if (debug_mode_) {
                    mask_yellow(roi).copyTo(roi_mask_vis(roi));
                }
            }
        } else {
            if (debug_mode_) {
                mask_yellow.copyTo(roi_mask_vis);
            }
        }
        yellow_px = cv::countNonZero(mask_yellow_roi);

        // state 1일 때만 yellow_detected pub
        if (mission_state_ == 1) {
            std_msgs::msg::Bool yellow_msg;
            yellow_msg.data = (yellow_px >= yellow_detect_thresh_);
            yellow_detected_pub_->publish(yellow_msg);
        }

        // ── 색깔 검출 시각화 창 ───────────────────
        cv::Mat yellow_threshold_vis;
        if (debug_mode_) {
            yellow_threshold_vis = maskToBgr(mask_yellow);
            cv::putText(yellow_threshold_vis,
                "Y:" + std::to_string(yellow_px) +
                (mission_state_ == 1 ? (yellow_px >= yellow_detect_thresh_ ? " [ON]" : " [off]") : " [inactive]"),
                cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // ── 차선 인식 (UFLD 또는 기존 HSV 적용) ──
        cv::Mat lane_mask;
        if (ufld_enabled_) {
            lane_mask = buildUFLDLaneMask();
        } else {
            lane_mask = buildLaneMask(hsv);
        }
        cv::Mat roi_vis = cv::Mat::zeros(lane_mask.size(), lane_mask.type());
        std::vector<cv::Point> bev_src_points = {
            cv::Point(static_cast<int>(bev_src_bl_x_), static_cast<int>(bev_src_bl_y_)),
            cv::Point(static_cast<int>(bev_src_br_x_), static_cast<int>(bev_src_br_y_)),
            cv::Point(static_cast<int>(bev_src_tr_x_), static_cast<int>(bev_src_tr_y_)),
            cv::Point(static_cast<int>(bev_src_tl_x_), static_cast<int>(bev_src_tl_y_))
        };
        std::vector<std::vector<cv::Point>> roi_polygons = {bev_src_points};
        cv::fillPoly(roi_vis, roi_polygons, cv::Scalar(255));
        cv::bitwise_and(lane_mask, roi_vis, roi_vis);

        cv::Mat blurred, binary;
        cv::GaussianBlur(roi_vis, blurred, cv::Size(5, 5), 0);
        cv::threshold(blurred, binary, 127, 255, cv::THRESH_BINARY);

        cv::Mat lane_warped;
        cv::warpPerspective(binary, lane_warped, M_, binary.size());

        cv::Mat sliding_vis;
        float offset = slidingWindowOffset(lane_warped, sliding_vis);
        if (ufld_enabled_ && ufld_direct_offset_valid_) {
            offset = ufld_direct_offset_;
        }

        const float filtered_offset = filterOffset(offset);

        std_msgs::msg::Float32 offset_msg;
        offset_msg.data = filtered_offset;
        offset_pub_->publish(offset_msg);

        if (publish_debug_images_) {
            publishCompressedImage(line_bev_pub_, maskToBgr(lane_warped), "line_bev");
            publishCompressedImage(line_sliding_pub_, sliding_vis, "line_sliding");
        }

        if (debug_mode_) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
                "[line_node] state=%d  offset=%.2f  yellow=%d%s",
                mission_state_, filtered_offset,
                yellow_px,
                mission_state_ == 1 ? (yellow_px >= yellow_detect_thresh_ ? " [ON]" : " [off]") : " [inactive]");
        }

        if (show_debug_ && debug_mode_) {
            showDebugImage("01 Original", frame_);
            showDebugImage("02 HSV Mask", maskToBgr(lane_mask));
            showDebugImage("03 ROI", maskToBgr(roi_vis));
            showDebugImage("04 BEV", maskToBgr(lane_warped));
            showDebugImage("05 Sliding Window", sliding_vis);
            showDebugImage("06 Yellow Threshold", yellow_threshold_vis);
            cv::waitKey(1);
        }
    }

    bool initUFLD() {
        std::ifstream proto_stream(ufld_proto_path_);
        if (!proto_stream.is_open()) {
            RCLCPP_ERROR(get_logger(), "Failed to open proto file: %s", ufld_proto_path_.c_str());
            return false;
        }
        std::string proto_content((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());

        std::ifstream model_stream(ufld_model_path_, std::ios::binary);
        if (!model_stream.is_open()) {
            RCLCPP_ERROR(get_logger(), "Failed to open model file: %s", ufld_model_path_.c_str());
            return false;
        }
        std::stringstream model_content_stream;
        model_content_stream << model_stream.rdbuf();
        std::string model_content = model_content_stream.str();

        TNN_NS::ModelConfig model_config;
        model_config.model_type = TNN_NS::MODEL_TYPE_TNN;
        model_config.params.push_back(proto_content);
        model_config.params.push_back(model_content);

        tnn_ = std::make_shared<TNN_NS::TNN>();
        TNN_NS::Status status = tnn_->Init(model_config);
        if (status != TNN_NS::TNN_OK) {
            RCLCPP_ERROR(get_logger(), "TNN Init failed: %s", status.description().c_str());
            return false;
        }

        TNN_NS::NetworkConfig net_config;
        if (ufld_use_gpu_) {
            net_config.device_type = TNN_NS::DEVICE_OPENCL;
            net_config.enable_tune_kernel = true;
            net_config.cache_path = "/home/topst/D-Racer-Kit/.shaders";
            net_config.precision = TNN_NS::PRECISION_AUTO;
        } else {
            net_config.device_type = TNN_NS::DEVICE_ARM;
            net_config.precision = TNN_NS::PRECISION_AUTO;
        }

        TNN_NS::Status error;
        instance_ = tnn_->CreateInst(net_config, error);
        if (error != TNN_NS::TNN_OK) {
            RCLCPP_ERROR(get_logger(), "TNN CreateInst failed: %s", error.description().c_str());
            return false;
        }

        return true;
    }

    cv::Mat buildUFLDLaneMask() {
        auto start_total = std::chrono::high_resolution_clock::now();
        cv::Mat mask = cv::Mat::zeros(frame_.size(), CV_8UC1);
        ufld_direct_offset_valid_ = false;
        if (!instance_) return mask;

        // Resize image to 160x320
        cv::Mat resized;
        cv::resize(frame_, resized, cv::Size(320, 160));

        // Prepare input Mat
        TNN_NS::DimsVector input_dims = {1, 3, 160, 320};
        auto input_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, input_dims, resized.data);

        TNN_NS::MatConvertParam input_param;
        input_param.scale = {0.01712475f, 0.01750700f, 0.01742919f, 0.0f};
        input_param.bias = {-2.11790393f, -2.03571429f, -1.80444444f, 0.0f};
        input_param.reverse_channel = true;

        TNN_NS::Status status = instance_->SetInputMat(input_mat, input_param, "input");
        if (status != TNN_NS::TNN_OK) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "UFLD SetInputMat failed: %s", status.description().c_str());
            return mask;
        }

        // Forward
        auto start_infer = std::chrono::high_resolution_clock::now();
        status = instance_->Forward();
        auto end_infer = std::chrono::high_resolution_clock::now();
        if (status != TNN_NS::TNN_OK) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "UFLD Forward failed: %s", status.description().c_str());
            return mask;
        }

        // Get output Mat
        std::shared_ptr<TNN_NS::Mat> output_mat;
        status = instance_->GetOutputMat(output_mat, TNN_NS::MatConvertParam(), "output", TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT);
        if (status != TNN_NS::TNN_OK) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "UFLD GetOutputMat failed: %s", status.description().c_str());
            return mask;
        }

        float* data = reinterpret_cast<float*>(output_mat->GetData());
        if (!data) return mask;

        // Row anchors in 288 scale
        static const std::vector<int> tusimple_row_anchor_56 = {
            64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284
        };
        static const std::vector<int> tusimple_row_anchor_36 = {
            64, 72, 76, 84, 88, 96, 100, 108, 116, 120, 128, 132, 140, 144,
            152, 160, 164, 172, 176, 184, 188, 196, 204, 208, 216, 220, 228,
            232, 240, 248, 252, 260, 264, 272, 276, 284
        };
        const std::vector<int>& tusimple_row_anchor = (ufld_cls_num_per_lane_ == 36) ? tusimple_row_anchor_36 : tusimple_row_anchor_56;

        int img_w = frame_.cols;
        int img_h = frame_.rows;
        cv::Mat viz;
        if (ufld_viz_pub_ && ufld_viz_pub_->get_subscription_count() > 0) {
            viz = frame_.clone();
        }

        bool use_white  = (mission_state_ == 1 ||
                           mission_state_ == 6 ||
                           (mission_state_ >= 7 && mission_state_ <= 9));
        bool use_yellow = (mission_state_ >= 2 && mission_state_ <= 6);

        std::vector<int> active_classes;
        if (use_white) {
            active_classes.push_back(0);
            active_classes.push_back(3);
        }
        if (use_yellow) {
            active_classes.push_back(1);
            active_classes.push_back(2);
        }

        const std::vector<cv::Scalar> lane_colors = {
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 0)
        };
        const std::vector<std::string> lane_names = {
            "white_left", "yellow_left", "yellow_right", "white_right"
        };
        const std::vector<int> viz_classes = {0, 1, 2, 3};
        const std::vector<int>& classes_to_process = viz.empty() ? active_classes : viz_classes;

        // Keep the driving mask mission-gated, but draw every lane class in debug visualization.
        int num_anchors = ufld_cls_num_per_lane_;
        int stride_c = num_anchors * 4;
        std::vector<std::vector<cv::Point>> lane_points(4);
        for (int l : classes_to_process) {
            bool class_is_active = std::find(active_classes.begin(), active_classes.end(), l) != active_classes.end();
            std::vector<cv::Point> pts;
            for (int r = 0; r < num_anchors; ++r) {
                // Find argmax over 101 classes
                int argmax_c = 0;
                float max_val = data[(num_anchors - 1 - r) * 4 + l]; // for c=0
                for (int c = 1; c < 101; ++c) {
                    float val = data[c * stride_c + (num_anchors - 1 - r) * 4 + l];
                    if (val > max_val) {
                        max_val = val;
                        argmax_c = c;
                    }
                }

                if (argmax_c < 100) {
                    // Compute Softmax over 0..99
                    float max_val_100 = data[(num_anchors - 1 - r) * 4 + l];
                    for (int c = 1; c < 100; ++c) {
                        float val = data[c * stride_c + (num_anchors - 1 - r) * 4 + l];
                        if (val > max_val_100) max_val_100 = val;
                    }

                    float sum_exp = 0.0f;
                    float sum_val = 0.0f;
                    for (int c = 0; c < 100; ++c) {
                        float val = data[c * stride_c + (num_anchors - 1 - r) * 4 + l];
                        float ev = std::exp(val - max_val_100);
                        sum_exp += ev;
                        sum_val += ev * (c + 1);
                    }

                    float grid_x = sum_val / sum_exp;
                    int x_coord = static_cast<int>(grid_x * img_w / 100.0) - 1;
                    int anchor_idx = num_anchors - 1 - r;
                    int y_coord = static_cast<int>(img_h * (tusimple_row_anchor[anchor_idx] / 288.0)) - 1;

                    pts.push_back(cv::Point(x_coord, y_coord));
                }
            }
            if (l >= 0 && l < static_cast<int>(lane_points.size())) {
                lane_points[l] = pts;
            }

            if (pts.size() >= 2) {
                for (size_t i = 0; i < pts.size() - 1; ++i) {
                    if (class_is_active) {
                        cv::line(mask, pts[i], pts[i+1], cv::Scalar(255), 10);
                    }
                    if (!viz.empty()) {
                        cv::line(viz, pts[i], pts[i+1], lane_colors[l], 3);
                    }
                }
                if (!viz.empty()) {
                    for (const auto& p : pts) {
                        cv::circle(viz, p, 3, lane_colors[l], -1);
                    }
                    cv::Point label_pt = pts[pts.size() / 2];
                    label_pt.y = std::max(20, label_pt.y - 8);
                    cv::putText(viz, lane_names[l], label_pt, cv::FONT_HERSHEY_SIMPLEX,
                                0.45, lane_colors[l], 1);
                }
            }
        }
        computeUFLDDirectOffset(lane_points, active_classes, viz);

        auto end_total = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_infer - start_infer).count();
        double total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_total - start_total).count();

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "[UFLD Latency] Inference: %.2f ms (%.1f FPS) | Total: %.2f ms (%.1f FPS)",
                             infer_ms, 1000.0 / infer_ms, total_ms, 1000.0 / total_ms);

        if (!viz.empty()) {
            char text[128];
            std::snprintf(text, sizeof(text), "TOPST UFLD %.1f ms %.1f FPS",
                          infer_ms, 1000.0 / infer_ms);
            cv::putText(viz, text, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
                        0.65, cv::Scalar(30, 255, 30), 2);
            publishCompressedImage(ufld_viz_pub_, viz, "camera");
        }

        return mask;
    }
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LineNode>());
    rclcpp::shutdown();
    return 0;
}
