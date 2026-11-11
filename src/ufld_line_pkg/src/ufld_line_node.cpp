#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ufld_line_pkg/tnn_ufld_runner.hpp"
#include "ufld_line_pkg/ufld_decoder.hpp"

namespace ufld_line_pkg {

class UfldLineNode : public rclcpp::Node {
public:
    UfldLineNode()
        : Node("ufld_line_node"),
          decoder_(UfldConfig{}),
          mission_state_(1) {
        config_ = readUfldConfig();
        decoder_ = UfldDecoder(config_);

        image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image/compressed");
        offset_topic_ = declare_parameter<std::string>("offset_topic", "/line_offset");
        heading_topic_ = declare_parameter<std::string>("heading_topic", "/line_heading");
        curvature_topic_ = declare_parameter<std::string>("curvature_topic", "/line_curvature");
        valid_topic_ = declare_parameter<std::string>("valid_topic", "/line_valid");
        tuning_topic_ = declare_parameter<std::string>("tuning_topic", "/line_tuning");
        debug_metrics_topic_ = declare_parameter<std::string>("debug_metrics_topic", "/ufld_debug");
        mission_topic_ = declare_parameter<std::string>("mission_topic", "/mission_state");
        lane_mode_topic_ = declare_parameter<std::string>("lane_mode_topic", "/mission_lane_mode");
        white_lane_detected_topic_ = declare_parameter<std::string>("white_lane_detected_topic", "/white_lane_detected");
        yellow_lane_detected_topic_ = declare_parameter<std::string>("yellow_lane_detected_topic", "/yellow_lane_detected");
        yellow_detected_topic_ = declare_parameter<std::string>("yellow_detected_topic", "/yellow_detected");
        debug_image_topic_ = declare_parameter<std::string>("debug_image_topic", "/ufld_line/debug_image/compressed");
        line_image_bev_topic_ = declare_parameter<std::string>("line_image_bev_topic", "/line/image/bev");
        line_image_overlay_topic_ = declare_parameter<std::string>("line_image_overlay_topic", "/line/image/sliding");

        proto_path_ = declare_parameter<std::string>("proto_path", "");
        model_path_ = declare_parameter<std::string>("model_path", "");
        input_name_ = declare_parameter<std::string>("input_name", "input");
        output_name_ = declare_parameter<std::string>("output_name", "output");

        use_opencl_ = declare_parameter<bool>("use_opencl", true);
        cpu_threads_ = declare_parameter<int>("cpu_threads", 4);
        show_debug_ = declare_parameter<bool>("show_debug", false);
        publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
        publish_zero_on_fail_ = declare_parameter<bool>("publish_zero_on_fail", false);
        offset_filter_alpha_ = declare_parameter<double>("offset_filter_alpha", 0.20);
        log_period_sec_ = declare_parameter<double>("log_period_sec", 1.0);
        dynamic_ld_enabled_ = declare_parameter<bool>("dynamic_ld_enabled", true);
        dynamic_ld_heading_start_ = declare_parameter<double>("dynamic_ld_heading_start", 0.18);
        dynamic_ld_heading_full_ = declare_parameter<double>("dynamic_ld_heading_full", 0.55);
        dynamic_ld_y_add_ = declare_parameter<double>("dynamic_ld_y_add", 0.10);
        heading_filter_alpha_ = declare_parameter<double>("heading_filter_alpha", 0.35);
        curvature_filter_alpha_ = declare_parameter<double>("curvature_filter_alpha", 0.25);
        use_bev_offset_ = declare_parameter<bool>("use_bev_offset", true);
        publish_bev_debug_image_ = declare_parameter<bool>("publish_bev_debug_image", true);
        bev_debug_quality_ = declare_parameter<int>("bev_debug_quality", 80);
        bev_debug_quality_ = std::max(1, std::min(100, bev_debug_quality_));

        bev_src_bl_x_ = declare_parameter<double>("bev_src_bl_x", 0.0);
        bev_src_bl_y_ = declare_parameter<double>("bev_src_bl_y", 118.0);
        bev_src_br_x_ = declare_parameter<double>("bev_src_br_x", 320.0);
        bev_src_br_y_ = declare_parameter<double>("bev_src_br_y", 118.0);
        bev_src_tl_x_ = declare_parameter<double>("bev_src_tl_x", 90.0);
        bev_src_tl_y_ = declare_parameter<double>("bev_src_tl_y", 80.0);
        bev_src_tr_x_ = declare_parameter<double>("bev_src_tr_x", 230.0);
        bev_src_tr_y_ = declare_parameter<double>("bev_src_tr_y", 80.0);
        bev_dst_bl_x_ = declare_parameter<double>("bev_dst_bl_x", 75.0);
        bev_dst_bl_y_ = declare_parameter<double>("bev_dst_bl_y", 120.0);
        bev_dst_br_x_ = declare_parameter<double>("bev_dst_br_x", 245.0);
        bev_dst_br_y_ = declare_parameter<double>("bev_dst_br_y", 120.0);
        bev_dst_tl_x_ = declare_parameter<double>("bev_dst_tl_x", 75.0);
        bev_dst_tl_y_ = declare_parameter<double>("bev_dst_tl_y", 0.0);
        bev_dst_tr_x_ = declare_parameter<double>("bev_dst_tr_x", 245.0);
        bev_dst_tr_y_ = declare_parameter<double>("bev_dst_tr_y", 0.0);
        computeBevMatrix();

        yellow_h_min_ = declare_parameter<int>("yellow_h_min", 15);
        yellow_h_max_ = declare_parameter<int>("yellow_h_max", 30);
        yellow_s_min_ = declare_parameter<int>("yellow_s_min", 70);
        yellow_s_max_ = declare_parameter<int>("yellow_s_max", 255);
        yellow_v_min_ = declare_parameter<int>("yellow_v_min", 120);
        yellow_v_max_ = declare_parameter<int>("yellow_v_max", 255);
        yellow_detect_thresh_ = declare_parameter<int>("yellow_detect_thresh", 99999);
        yellow_roi_x_ = declare_parameter<int>("yellow_roi_x", 0);
        yellow_roi_y_ = declare_parameter<int>("yellow_roi_y", 0);
        yellow_roi_width_ = declare_parameter<int>("yellow_roi_width", 0);
        yellow_roi_height_ = declare_parameter<int>("yellow_roi_height", 0);

        // Yellow X-cross detector, from UFLD geometry only: in normal driving the
        // yellow_left/yellow_right boundaries sit ~2*half_lane_width apart in BEV;
        // where they cross (X zone) their separation collapses toward zero or
        // inverts. Fires when the minimum same-height separation drops below
        // yellow_cross_width_px.
        yellow_cross_topic_ = declare_parameter<std::string>("yellow_cross_topic", "/yellow_cross_detected");
        yellow_cross_width_px_ = declare_parameter<double>("yellow_cross_width_px", 60.0);
        yellow_cross_y_tol_px_ = std::max(1.0, declare_parameter<double>("yellow_cross_y_tol_px", 12.0));
        yellow_cross_stable_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("yellow_cross_stable_ticks", 2)));
        // Asymmetric debounce: the cross flag only clears after this many
        // consecutive non-cross frames, so one physical X crossing produces a
        // single clean pulse instead of chattering rising edges.
        yellow_cross_clear_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("yellow_cross_clear_ticks", 6)));

        // Warning zone: red-painted road section between the active lane
        // boundaries, checked in BEV. Red wraps around hue 0, so two HSV bands
        // ([0..h1_max] and [h2_min..180]) are OR-ed. Raw per-frame flag; the
        // mission planner debounces and drops speed to SLOW while inside.
        warning_zone_enable_ = declare_parameter<bool>("warning_zone_enable", true);
        warning_zone_topic_ = declare_parameter<std::string>("warning_zone_topic", "/warning_zone_detected");
        warning_red_h1_max_ = declare_parameter<int>("warning_red_h1_max", 10);
        warning_red_h2_min_ = declare_parameter<int>("warning_red_h2_min", 170);
        warning_red_s_min_ = declare_parameter<int>("warning_red_s_min", 120);
        warning_red_v_min_ = declare_parameter<int>("warning_red_v_min", 70);
        warning_red_ratio_thresh_ = declare_parameter<double>("warning_red_ratio_thresh", 0.35);
        // Keep the sample window off the lane paint itself.
        warning_inset_px_ = std::max(0.0, declare_parameter<double>("warning_inset_px", 12.0));

        std::string init_error;
        runner_ready_ = runner_.init(
            proto_path_, model_path_, input_name_, output_name_,
            config_, use_opencl_, cpu_threads_, &init_error);
        if (!runner_ready_) {
            RCLCPP_ERROR(get_logger(), "[ufld_line] TNN init failed: %s", init_error.c_str());
        }

        image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            image_topic_, 1, [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) { imageCb(msg); });
        mission_sub_ = create_subscription<std_msgs::msg::Int32>(
            mission_topic_, 10, [this](const std_msgs::msg::Int32::SharedPtr msg) { mission_state_ = msg->data; });
        lane_mode_sub_ = create_subscription<std_msgs::msg::Int32>(
            lane_mode_topic_, 10, [this](const std_msgs::msg::Int32::SharedPtr msg) { lane_mode_ = msg->data; });
        tuning_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            tuning_topic_, 10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) { tuningCb(msg); });

        offset_pub_ = create_publisher<std_msgs::msg::Float32>(offset_topic_, 10);
        heading_pub_ = create_publisher<std_msgs::msg::Float32>(heading_topic_, 10);
        curvature_pub_ = create_publisher<std_msgs::msg::Float32>(curvature_topic_, 10);
        valid_pub_ = create_publisher<std_msgs::msg::Bool>(valid_topic_, 10);
        debug_metrics_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(debug_metrics_topic_, 10);
        yellow_detected_pub_ = create_publisher<std_msgs::msg::Bool>(yellow_detected_topic_, 10);
        white_lane_detected_pub_ = create_publisher<std_msgs::msg::Bool>(white_lane_detected_topic_, 10);
        yellow_lane_detected_pub_ = create_publisher<std_msgs::msg::Bool>(yellow_lane_detected_topic_, 10);
        yellow_cross_pub_ = create_publisher<std_msgs::msg::Bool>(yellow_cross_topic_, 10);
        warning_zone_pub_ = create_publisher<std_msgs::msg::Bool>(warning_zone_topic_, 10);
        debug_image_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(debug_image_topic_, 1);
        line_bev_image_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(line_image_bev_topic_, 1);
        line_overlay_image_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(line_image_overlay_topic_, 1);

        RCLCPP_INFO(get_logger(),
                    "[ufld_line] Started. image=%s offset=%s heading=%s yellow=%s model=%s opencl=%s bev_offset=%s",
                    image_topic_.c_str(), offset_topic_.c_str(), heading_topic_.c_str(),
                    yellow_detected_topic_.c_str(),
                    model_path_.c_str(), use_opencl_ ? "true" : "false",
                    use_bev_offset_ ? "true" : "false");
        RCLCPP_INFO(get_logger(),
                    "[ufld_line] tuning=%s target_y=%.2f band=%.2f max_step=%.1f half_lane_width=%.1f offset_alpha=%.2f",
                    tuning_topic_.c_str(), config_.near_band, config_.offset_band_ratio,
                    config_.offset_max_step_px, config_.half_lane_width, offset_filter_alpha_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr lane_mode_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr tuning_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr heading_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr curvature_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr valid_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_metrics_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr yellow_detected_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr white_lane_detected_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr yellow_lane_detected_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr yellow_cross_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr line_bev_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr line_overlay_image_pub_;

    std::string image_topic_;
    std::string offset_topic_;
    std::string heading_topic_;
    std::string curvature_topic_;
    std::string valid_topic_;
    std::string tuning_topic_;
    std::string debug_metrics_topic_;
    std::string mission_topic_;
    std::string lane_mode_topic_;
    std::string white_lane_detected_topic_;
    std::string yellow_lane_detected_topic_;
    std::string yellow_detected_topic_;
    std::string debug_image_topic_;
    std::string line_image_bev_topic_;
    std::string line_image_overlay_topic_;
    std::string proto_path_;
    std::string model_path_;
    std::string input_name_;
    std::string output_name_;

    TnnUfldRunner runner_;
    UfldDecoder decoder_;
    UfldConfig config_;
    int mission_state_;
    // /mission_lane_mode: 0 white, 1 yellow, 2 yellow-right-only, 3 yellow-left-only.
    // -1 (nothing received / no planner running) = use all four lane classes.
    int lane_mode_ = -1;
    bool runner_ready_ = false;
    bool use_opencl_ = true;
    bool show_debug_ = false;
    bool publish_debug_image_ = true;
    bool publish_zero_on_fail_ = false;
    int cpu_threads_ = 4;
    double offset_filter_alpha_ = 0.20;
    double log_period_sec_ = 1.0;
    bool dynamic_ld_enabled_ = true;
    double dynamic_ld_heading_start_ = 0.18;
    double dynamic_ld_heading_full_ = 0.55;
    double dynamic_ld_y_add_ = 0.10;
    double heading_filter_alpha_ = 0.35;
    double curvature_filter_alpha_ = 0.25;
    bool use_bev_offset_ = true;
    bool publish_bev_debug_image_ = true;
    int bev_debug_quality_ = 80;
    double bev_src_bl_x_ = 0.0;
    double bev_src_bl_y_ = 118.0;
    double bev_src_br_x_ = 320.0;
    double bev_src_br_y_ = 118.0;
    double bev_src_tl_x_ = 90.0;
    double bev_src_tl_y_ = 80.0;
    double bev_src_tr_x_ = 230.0;
    double bev_src_tr_y_ = 80.0;
    double bev_dst_bl_x_ = 75.0;
    double bev_dst_bl_y_ = 120.0;
    double bev_dst_br_x_ = 245.0;
    double bev_dst_br_y_ = 120.0;
    double bev_dst_tl_x_ = 75.0;
    double bev_dst_tl_y_ = 0.0;
    double bev_dst_tr_x_ = 245.0;
    double bev_dst_tr_y_ = 0.0;
    cv::Mat bev_matrix_;
    float filtered_offset_ = 0.0f;
    float filtered_heading_ = 0.0f;
    float filtered_curvature_ = 0.0f;
    bool has_filtered_heading_ = false;
    bool has_filtered_offset_ = false;
    bool has_filtered_curvature_ = false;
    float last_lane_center_x_ = 0.0f;
    bool has_last_lane_center_x_ = false;
    float last_raw_offset_ = 0.0f;
    float last_clamped_offset_ = 0.0f;
    float last_raw_lane_center_x_ = 0.0f;
    float last_raw_heading_ = 0.0f;
    float last_filtered_heading_ = 0.0f;
    float last_raw_curvature_ = 0.0f;
    float last_dynamic_target_y_ = 0.0f;
    float last_dynamic_factor_ = 0.0f;
    bool last_step_clamped_ = false;
    float last_step_delta_ = 0.0f;
    float last_lane_l0_count_ = 0.0f;
    float last_lane_l1_count_ = 0.0f;
    float last_lane_l2_count_ = 0.0f;
    float last_lane_l3_count_ = 0.0f;
    size_t frame_count_ = 0;
    size_t success_count_ = 0;
    rclcpp::Time last_log_time_;

    int yellow_h_min_, yellow_h_max_;
    int yellow_s_min_, yellow_s_max_;
    int yellow_v_min_, yellow_v_max_;
    int yellow_detect_thresh_;
    int yellow_roi_x_, yellow_roi_y_, yellow_roi_width_, yellow_roi_height_;

    std::string yellow_cross_topic_;
    double yellow_cross_width_px_;
    double yellow_cross_y_tol_px_;
    int yellow_cross_stable_ticks_;
    int yellow_cross_clear_ticks_;
    int yellow_cross_streak_ = 0;
    int yellow_cross_miss_streak_ = 0;
    float last_yellow_min_sep_px_ = 0.0f;

    bool warning_zone_enable_ = true;
    std::string warning_zone_topic_;
    int warning_red_h1_max_ = 10;
    int warning_red_h2_min_ = 170;
    int warning_red_s_min_ = 120;
    int warning_red_v_min_ = 70;
    double warning_red_ratio_thresh_ = 0.35;
    double warning_inset_px_ = 12.0;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr warning_zone_pub_;
    // Per-frame BEV lane bounds from applyBevControlGeometry; consumed by
    // detectWarningZone in the same frame only.
    float bev_left_bound_x_ = 0.0f;
    float bev_right_bound_x_ = 0.0f;
    bool has_bev_bounds_ = false;
    float last_warning_red_ratio_ = 0.0f;
    bool last_warning_zone_ = false;

    void computeBevMatrix() {
        const std::vector<cv::Point2f> src = {
            cv::Point2f(static_cast<float>(bev_src_bl_x_), static_cast<float>(bev_src_bl_y_)),
            cv::Point2f(static_cast<float>(bev_src_br_x_), static_cast<float>(bev_src_br_y_)),
            cv::Point2f(static_cast<float>(bev_src_tl_x_), static_cast<float>(bev_src_tl_y_)),
            cv::Point2f(static_cast<float>(bev_src_tr_x_), static_cast<float>(bev_src_tr_y_)),
        };
        const std::vector<cv::Point2f> dst = {
            cv::Point2f(static_cast<float>(bev_dst_bl_x_), static_cast<float>(bev_dst_bl_y_)),
            cv::Point2f(static_cast<float>(bev_dst_br_x_), static_cast<float>(bev_dst_br_y_)),
            cv::Point2f(static_cast<float>(bev_dst_tl_x_), static_cast<float>(bev_dst_tl_y_)),
            cv::Point2f(static_cast<float>(bev_dst_tr_x_), static_cast<float>(bev_dst_tr_y_)),
        };
        bev_matrix_ = cv::getPerspectiveTransform(src, dst);
    }

    static float medianValue(std::vector<float> values) {
        if (values.empty()) return 0.0f;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    static float laneHeadingDxDy(std::vector<cv::Point2f> lane) {
        if (lane.size() < 2) return 0.0f;
        std::sort(lane.begin(), lane.end(),
                  [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

        const size_t top_idx = lane.size() / 4;
        const size_t bottom_idx = lane.size() - 1 - lane.size() / 4;
        const float dy = lane[bottom_idx].y - lane[top_idx].y;
        if (std::abs(dy) < 1.0f) return 0.0f;
        return (lane[bottom_idx].x - lane[top_idx].x) / dy;
    }

    // Least-squares fit of x = a*y^2 + b*y + c; 'a' captures how fast the lane bends,
    // which the single secant slope used for heading cannot see.
    static float laneCurvatureQuad(const std::vector<cv::Point2f>& lane, int min_points) {
        if (static_cast<int>(lane.size()) < std::max(3, min_points)) return 0.0f;

        double Sy1 = 0.0, Sy2 = 0.0, Sy3 = 0.0, Sy4 = 0.0;
        double Sx0 = 0.0, Sx1 = 0.0, Sx2 = 0.0;
        for (const auto& p : lane) {
            const double y = p.y;
            const double x = p.x;
            const double y2 = y * y;
            Sy1 += y;
            Sy2 += y2;
            Sy3 += y2 * y;
            Sy4 += y2 * y2;
            Sx0 += x;
            Sx1 += x * y;
            Sx2 += x * y2;
        }
        const double n = static_cast<double>(lane.size());

        const double m[3][3] = {
            {Sy4, Sy3, Sy2},
            {Sy3, Sy2, Sy1},
            {Sy2, Sy1, n},
        };
        const double v[3] = {Sx2, Sx1, Sx0};

        const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if (std::abs(det) < 1e-6) return 0.0f;

        const double det_a = v[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                             m[0][1] * (v[1] * m[2][2] - m[1][2] * v[2]) +
                             m[0][2] * (v[1] * m[2][1] - m[1][1] * v[2]);

        return static_cast<float>(det_a / det);
    }

    // Lane classes by mission lane mode. UFLD class order:
    // 0 white_left, 1 yellow_left, 2 yellow_right, 3 white_right.
    void activeLaneClasses(std::vector<int>& left, std::vector<int>& right) const {
        switch (lane_mode_) {
        case 0:  left = {0};    right = {3};    break;  // LANE_WHITE
        case 1:  left = {1};    right = {2};    break;  // LANE_YELLOW
        case 2:  left = {};     right = {2};    break;  // LANE_YELLOW_RIGHT_ONLY
        case 3:  left = {1};    right = {};     break;  // LANE_YELLOW_LEFT_ONLY
        default: left = {0, 1}; right = {2, 3}; break;  // no planner: use everything
        }
    }

    std::vector<std::vector<cv::Point2f>> transformLanesToBev(const LaneDetection& detection) const {
        std::vector<std::vector<cv::Point2f>> bev_lanes(detection.lanes.size());
        if (bev_matrix_.empty()) return bev_lanes;

        for (size_t i = 0; i < detection.lanes.size(); ++i) {
            if (detection.lanes[i].empty()) continue;
            cv::perspectiveTransform(detection.lanes[i], bev_lanes[i], bev_matrix_);
        }
        return bev_lanes;
    }

    bool applyBevControlGeometry(LaneDetection& detection, int image_width, int image_height) {
        has_bev_bounds_ = false;
        if (!use_bev_offset_ || !detection.valid || image_width <= 0 || image_height <= 0) {
            return false;
        }

        const auto bev_lanes = transformLanesToBev(detection);
        if (bev_lanes.empty()) return false;

        const float target_y = static_cast<float>(image_height) * config_.near_band;
        const float band_half = static_cast<float>(image_height) *
                                std::max(0.0f, config_.offset_band_ratio) * 0.5f;
        const float y_min = std::max(0.0f, target_y - band_half);
        const float y_max = std::min(static_cast<float>(image_height - 1), target_y + band_half);

        std::vector<int> left_classes, right_classes;
        activeLaneClasses(left_classes, right_classes);

        std::vector<float> left_xs;
        std::vector<float> right_xs;
        for (int lane_idx : left_classes) {
            if (lane_idx >= static_cast<int>(bev_lanes.size())) continue;
            for (const auto& p : bev_lanes[lane_idx]) {
                if (p.y >= y_min && p.y <= y_max) left_xs.push_back(p.x);
            }
        }
        for (int lane_idx : right_classes) {
            if (lane_idx >= static_cast<int>(bev_lanes.size())) continue;
            for (const auto& p : bev_lanes[lane_idx]) {
                if (p.y >= y_min && p.y <= y_max) right_xs.push_back(p.x);
            }
        }

        const bool have_left = !left_xs.empty();
        const bool have_right = !right_xs.empty();
        if (!have_left && !have_right) return false;

        const float image_center = static_cast<float>(image_width) * 0.5f;
        float lane_center = image_center;
        if (have_left && have_right) {
            bev_left_bound_x_ = medianValue(left_xs);
            bev_right_bound_x_ = medianValue(right_xs);
            lane_center = 0.5f * (bev_left_bound_x_ + bev_right_bound_x_);
        } else if (have_left) {
            bev_left_bound_x_ = medianValue(left_xs);
            bev_right_bound_x_ = bev_left_bound_x_ + 2.0f * config_.half_lane_width;
            lane_center = bev_left_bound_x_ + config_.half_lane_width;
        } else {
            bev_right_bound_x_ = medianValue(right_xs);
            bev_left_bound_x_ = bev_right_bound_x_ - 2.0f * config_.half_lane_width;
            lane_center = bev_right_bound_x_ - config_.half_lane_width;
        }
        has_bev_bounds_ = true;

        // heading/curvature also only from the mission-active lane classes, so an
        // ignored lane color can't skew the geometry during one-sided missions.
        std::vector<int> active_classes = left_classes;
        active_classes.insert(active_classes.end(), right_classes.begin(), right_classes.end());

        std::vector<float> headings;
        std::vector<float> curvatures;
        for (int lane_idx : active_classes) {
            if (lane_idx >= static_cast<int>(bev_lanes.size())) continue;
            const auto& lane = bev_lanes[lane_idx];
            if (lane.size() >= static_cast<size_t>(config_.min_lane_points)) {
                headings.push_back(laneHeadingDxDy(lane));
            }
            if (lane.size() >= static_cast<size_t>(config_.curvature_min_points)) {
                curvatures.push_back(laneCurvatureQuad(lane, config_.curvature_min_points));
            }
        }
        if (!headings.empty()) {
            detection.heading_error = medianValue(headings);
            const float heading_max_abs = std::max(0.0f, config_.heading_max_abs);
            detection.heading_error = std::max(-heading_max_abs, std::min(heading_max_abs, detection.heading_error));
        }
        if (!curvatures.empty()) {
            detection.curvature = medianValue(curvatures);
            const float curvature_max_abs = std::max(0.0f, config_.curvature_max_abs);
            detection.curvature = std::max(-curvature_max_abs, std::min(curvature_max_abs, detection.curvature));
        }

        detection.lane_center_x = lane_center;
        detection.offset = lane_center - image_center;
        return true;
    }

    UfldConfig readUfldConfig() {
        UfldConfig cfg;
        cfg.input_width = declare_parameter<int>("input_width", 320);
        cfg.input_height = declare_parameter<int>("input_height", 160);
        cfg.num_lanes = declare_parameter<int>("num_lanes", 4);
        cfg.griding_num = declare_parameter<int>("griding_num", 100);
        cfg.cls_num_per_lane = declare_parameter<int>("cls_num_per_lane", 36);
        cfg.near_band = declare_parameter<double>("near_band", 0.70);
        cfg.offset_band_ratio = declare_parameter<double>("offset_band_ratio", 0.16);
        cfg.offset_max_step_px = declare_parameter<double>("offset_max_step_px", 35.0);
        cfg.half_lane_width = declare_parameter<double>("half_lane_width", 70.0);
        cfg.min_lane_points = declare_parameter<int>("min_lane_points", 4);
        cfg.min_vertical_span_ratio = declare_parameter<double>("min_vertical_span_ratio", 0.08);
        cfg.heading_min_vertical_span_ratio = declare_parameter<double>("heading_min_vertical_span_ratio", 0.18);
        cfg.heading_max_abs = declare_parameter<double>("heading_max_abs", 0.50);
        cfg.dedupe_max_dx_px = declare_parameter<double>("dedupe_max_dx_px", 28.0);
        cfg.dedupe_min_overlap = declare_parameter<double>("dedupe_min_overlap", 0.45);
        cfg.curvature_min_points = declare_parameter<int>("curvature_min_points", 6);
        cfg.curvature_max_abs = declare_parameter<double>("curvature_max_abs", 0.02);
        config_ = cfg;
        return cfg;
    }

    float scheduledTargetY() {
        last_dynamic_factor_ = 0.0f;
        last_dynamic_target_y_ = config_.near_band;
        return config_.near_band;
    }

    void tuningCb(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        // ufld_view.py publishes legacy tuning arrays:
        // [12-19] BEV source points: bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y,
        // [20-27] BEV destination points: bl_x, bl_y, br_x, br_y, tl_x, tl_y, tr_x, tr_y,
        // [28] target Y ratio, [29] full lane width for one-lane fallback,
        // [30] offset EMA alpha, [31] target Y band, [32] max center step,
        // [33] dynamic LD enabled, [34] heading start, [35] heading full,
        // [36] dynamic target Y add, [37] heading EMA alpha, [38] BEV offset enabled,
        // [39] curvature EMA alpha,
        // [40] yellow cross width px, [41] yellow cross y tol px,
        // [42] yellow cross stable ticks, [43] yellow cross clear ticks.
        if (msg->data.size() < 31) {
            return;
        }

        const auto clamp_x = [](float v) { return static_cast<double>(std::clamp(v, 0.0f, 320.0f)); };
        const auto clamp_y = [](float v) { return static_cast<double>(std::clamp(v, 0.0f, 160.0f)); };
        const double bev_src_bl_x = msg->data.size() >= 28 ? clamp_x(msg->data[12]) : bev_src_bl_x_;
        const double bev_src_bl_y = msg->data.size() >= 28 ? clamp_y(msg->data[13]) : bev_src_bl_y_;
        const double bev_src_br_x = msg->data.size() >= 28 ? clamp_x(msg->data[14]) : bev_src_br_x_;
        const double bev_src_br_y = msg->data.size() >= 28 ? clamp_y(msg->data[15]) : bev_src_br_y_;
        const double bev_src_tl_x = msg->data.size() >= 28 ? clamp_x(msg->data[16]) : bev_src_tl_x_;
        const double bev_src_tl_y = msg->data.size() >= 28 ? clamp_y(msg->data[17]) : bev_src_tl_y_;
        const double bev_src_tr_x = msg->data.size() >= 28 ? clamp_x(msg->data[18]) : bev_src_tr_x_;
        const double bev_src_tr_y = msg->data.size() >= 28 ? clamp_y(msg->data[19]) : bev_src_tr_y_;
        const double bev_dst_bl_x = msg->data.size() >= 28 ? clamp_x(msg->data[20]) : bev_dst_bl_x_;
        const double bev_dst_bl_y = msg->data.size() >= 28 ? clamp_y(msg->data[21]) : bev_dst_bl_y_;
        const double bev_dst_br_x = msg->data.size() >= 28 ? clamp_x(msg->data[22]) : bev_dst_br_x_;
        const double bev_dst_br_y = msg->data.size() >= 28 ? clamp_y(msg->data[23]) : bev_dst_br_y_;
        const double bev_dst_tl_x = msg->data.size() >= 28 ? clamp_x(msg->data[24]) : bev_dst_tl_x_;
        const double bev_dst_tl_y = msg->data.size() >= 28 ? clamp_y(msg->data[25]) : bev_dst_tl_y_;
        const double bev_dst_tr_x = msg->data.size() >= 28 ? clamp_x(msg->data[26]) : bev_dst_tr_x_;
        const double bev_dst_tr_y = msg->data.size() >= 28 ? clamp_y(msg->data[27]) : bev_dst_tr_y_;
        const float near_band = std::clamp(msg->data[28], 0.10f, 0.98f);
        const float lane_width = std::clamp(msg->data[29], 40.0f, 320.0f);
        const float half_lane_width = lane_width * 0.5f;
        const double offset_alpha = static_cast<double>(std::clamp(msg->data[30], 0.01f, 1.0f));
        const float offset_band_ratio = msg->data.size() >= 32
            ? std::clamp(msg->data[31], 0.0f, 0.50f)
            : config_.offset_band_ratio;
        const float offset_max_step_px = msg->data.size() >= 33
            ? std::clamp(msg->data[32], 0.0f, 320.0f)
            : config_.offset_max_step_px;
        const bool dynamic_ld_enabled = msg->data.size() >= 34
            ? msg->data[33] >= 0.5f
            : dynamic_ld_enabled_;
        const double dynamic_ld_heading_start = msg->data.size() >= 35
            ? static_cast<double>(std::clamp(msg->data[34], 0.0f, 2.0f))
            : dynamic_ld_heading_start_;
        const double dynamic_ld_heading_full = msg->data.size() >= 36
            ? static_cast<double>(std::clamp(msg->data[35], 0.0f, 2.0f))
            : dynamic_ld_heading_full_;
        const double dynamic_ld_y_add = msg->data.size() >= 37
            ? static_cast<double>(std::clamp(msg->data[36], -0.5f, 0.5f))
            : dynamic_ld_y_add_;
        const double heading_filter_alpha = msg->data.size() >= 38
            ? static_cast<double>(std::clamp(msg->data[37], 0.01f, 1.0f))
            : heading_filter_alpha_;
        const bool use_bev_offset = msg->data.size() >= 39
            ? msg->data[38] >= 0.5f
            : use_bev_offset_;
        const double curvature_filter_alpha = msg->data.size() >= 40
            ? static_cast<double>(std::clamp(msg->data[39], 0.01f, 1.0f))
            : curvature_filter_alpha_;
        const double yellow_cross_width_px = msg->data.size() >= 41
            ? static_cast<double>(std::clamp(msg->data[40], 5.0f, 320.0f))
            : yellow_cross_width_px_;
        const double yellow_cross_y_tol_px = msg->data.size() >= 42
            ? static_cast<double>(std::clamp(msg->data[41], 1.0f, 80.0f))
            : yellow_cross_y_tol_px_;
        const int yellow_cross_stable_ticks = msg->data.size() >= 43
            ? std::clamp(static_cast<int>(msg->data[42] + 0.5f), 1, 30)
            : yellow_cross_stable_ticks_;
        const int yellow_cross_clear_ticks = msg->data.size() >= 44
            ? std::clamp(static_cast<int>(msg->data[43] + 0.5f), 1, 60)
            : yellow_cross_clear_ticks_;

        const bool changed =
            std::abs(config_.near_band - near_band) > 0.001f ||
            std::abs(config_.offset_band_ratio - offset_band_ratio) > 0.001f ||
            std::abs(config_.offset_max_step_px - offset_max_step_px) > 0.5f ||
            std::abs(config_.half_lane_width - half_lane_width) > 0.5f ||
            std::abs(offset_filter_alpha_ - offset_alpha) > 0.001 ||
            dynamic_ld_enabled_ != dynamic_ld_enabled ||
            std::abs(dynamic_ld_heading_start_ - dynamic_ld_heading_start) > 0.001 ||
            std::abs(dynamic_ld_heading_full_ - dynamic_ld_heading_full) > 0.001 ||
            std::abs(dynamic_ld_y_add_ - dynamic_ld_y_add) > 0.001 ||
            std::abs(heading_filter_alpha_ - heading_filter_alpha) > 0.001 ||
            std::abs(curvature_filter_alpha_ - curvature_filter_alpha) > 0.001 ||
            std::abs(yellow_cross_width_px_ - yellow_cross_width_px) > 0.5 ||
            std::abs(yellow_cross_y_tol_px_ - yellow_cross_y_tol_px) > 0.5 ||
            yellow_cross_stable_ticks_ != yellow_cross_stable_ticks ||
            yellow_cross_clear_ticks_ != yellow_cross_clear_ticks ||
            use_bev_offset_ != use_bev_offset ||
            std::abs(bev_src_bl_x_ - bev_src_bl_x) > 0.5 ||
            std::abs(bev_src_bl_y_ - bev_src_bl_y) > 0.5 ||
            std::abs(bev_src_br_x_ - bev_src_br_x) > 0.5 ||
            std::abs(bev_src_br_y_ - bev_src_br_y) > 0.5 ||
            std::abs(bev_src_tl_x_ - bev_src_tl_x) > 0.5 ||
            std::abs(bev_src_tl_y_ - bev_src_tl_y) > 0.5 ||
            std::abs(bev_src_tr_x_ - bev_src_tr_x) > 0.5 ||
            std::abs(bev_src_tr_y_ - bev_src_tr_y) > 0.5 ||
            std::abs(bev_dst_bl_x_ - bev_dst_bl_x) > 0.5 ||
            std::abs(bev_dst_bl_y_ - bev_dst_bl_y) > 0.5 ||
            std::abs(bev_dst_br_x_ - bev_dst_br_x) > 0.5 ||
            std::abs(bev_dst_br_y_ - bev_dst_br_y) > 0.5 ||
            std::abs(bev_dst_tl_x_ - bev_dst_tl_x) > 0.5 ||
            std::abs(bev_dst_tl_y_ - bev_dst_tl_y) > 0.5 ||
            std::abs(bev_dst_tr_x_ - bev_dst_tr_x) > 0.5 ||
            std::abs(bev_dst_tr_y_ - bev_dst_tr_y) > 0.5;
        if (!changed) {
            return;
        }

        bev_src_bl_x_ = bev_src_bl_x;
        bev_src_bl_y_ = bev_src_bl_y;
        bev_src_br_x_ = bev_src_br_x;
        bev_src_br_y_ = bev_src_br_y;
        bev_src_tl_x_ = bev_src_tl_x;
        bev_src_tl_y_ = bev_src_tl_y;
        bev_src_tr_x_ = bev_src_tr_x;
        bev_src_tr_y_ = bev_src_tr_y;
        bev_dst_bl_x_ = bev_dst_bl_x;
        bev_dst_bl_y_ = bev_dst_bl_y;
        bev_dst_br_x_ = bev_dst_br_x;
        bev_dst_br_y_ = bev_dst_br_y;
        bev_dst_tl_x_ = bev_dst_tl_x;
        bev_dst_tl_y_ = bev_dst_tl_y;
        bev_dst_tr_x_ = bev_dst_tr_x;
        bev_dst_tr_y_ = bev_dst_tr_y;
        computeBevMatrix();
        config_.near_band = near_band;
        config_.offset_band_ratio = offset_band_ratio;
        config_.offset_max_step_px = offset_max_step_px;
        config_.half_lane_width = half_lane_width;
        offset_filter_alpha_ = offset_alpha;
        dynamic_ld_enabled_ = dynamic_ld_enabled;
        dynamic_ld_heading_start_ = dynamic_ld_heading_start;
        dynamic_ld_heading_full_ = dynamic_ld_heading_full;
        heading_filter_alpha_ = heading_filter_alpha;
        curvature_filter_alpha_ = curvature_filter_alpha;
        dynamic_ld_y_add_ = dynamic_ld_y_add;
        use_bev_offset_ = use_bev_offset;
        yellow_cross_width_px_ = yellow_cross_width_px;
        yellow_cross_y_tol_px_ = yellow_cross_y_tol_px;
        yellow_cross_stable_ticks_ = yellow_cross_stable_ticks;
        yellow_cross_clear_ticks_ = yellow_cross_clear_ticks;
        decoder_ = UfldDecoder(config_);

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "[ufld_line] tuning applied: target_y=%.2f band=%.2f max_step=%.1f lane_width=%.1f half_lane_width=%.1f "
            "offset_alpha=%.2f dyn=%d h_start=%.2f h_full=%.2f y_add=%.2f h_alpha=%.2f c_alpha=%.2f bev=%d "
            "xw=%.0f xytol=%.0f xticks=%d xclear=%d",
            config_.near_band, config_.offset_band_ratio, config_.offset_max_step_px,
            lane_width, config_.half_lane_width, offset_filter_alpha_, dynamic_ld_enabled_ ? 1 : 0,
            dynamic_ld_heading_start_, dynamic_ld_heading_full_, dynamic_ld_y_add_, heading_filter_alpha_,
            curvature_filter_alpha_, use_bev_offset_ ? 1 : 0,
            yellow_cross_width_px_, yellow_cross_y_tol_px_, yellow_cross_stable_ticks_,
            yellow_cross_clear_ticks_);
    }

    void imageCb(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        ++frame_count_;

        cv::Mat frame;
        try {
            frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        } catch (const cv::Exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "[ufld_line] Image decode error: %s", e.what());
            return;
        }
        if (frame.empty()) return;

        publishYellow(frame);

        bool valid = false;
        double infer_ms = 0.0;
        LaneDetection detection;
        std::string error;
        if (runner_ready_) {
            UfldOutput output;
            if (runner_.infer(frame, &output, &infer_ms, &error)) {
                UfldConfig frame_config = config_;
                frame_config.near_band = scheduledTargetY();
                UfldDecoder frame_decoder(frame_config);
                detection = frame_decoder.decode(output, frame.cols, frame.rows);
                valid = detection.valid;
                if (valid) {
                    applyBevControlGeometry(detection, frame.cols, frame.rows);
                }
            } else {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                     "[ufld_line] infer failed: %s", error.c_str());
            }
        }

        if (valid) {
            updateDetectionMetrics(detection);
            clampDetectionStep(detection, frame.cols);
            updateHeadingFilter(detection.heading_error);
            updateCurvatureFilter(detection.curvature);
            ++success_count_;
            if (!has_filtered_offset_) {
                filtered_offset_ = detection.offset;
                has_filtered_offset_ = true;
            } else {
                const float alpha = static_cast<float>(std::max(0.0, std::min(1.0, offset_filter_alpha_)));
                filtered_offset_ = filtered_offset_ * (1.0f - alpha) + detection.offset * alpha;
            }
            std_msgs::msg::Float32 offset_msg;
            offset_msg.data = filtered_offset_;
            offset_pub_->publish(offset_msg);
            std_msgs::msg::Float32 heading_msg;
            heading_msg.data = filtered_heading_;
            heading_pub_->publish(heading_msg);
            std_msgs::msg::Float32 curvature_msg;
            curvature_msg.data = filtered_curvature_;
            curvature_pub_->publish(curvature_msg);
        } else {
            last_step_clamped_ = false;
            last_step_delta_ = 0.0f;
            updateDetectionMetrics(detection);
        }
        publishLaneColorFlags(detection);
        publishYellowCross(detection);

        if (warning_zone_enable_) {
            last_warning_zone_ = valid && detectWarningZone(frame);
            std_msgs::msg::Bool warning_msg;
            warning_msg.data = last_warning_zone_;
            warning_zone_pub_->publish(warning_msg);
        }

        std_msgs::msg::Bool valid_msg;
        valid_msg.data = valid;
        valid_pub_->publish(valid_msg);

        if (!valid) {
            last_raw_offset_ = detection.offset;
            last_clamped_offset_ = detection.offset;
            if (publish_zero_on_fail_) {
                std_msgs::msg::Float32 offset_msg;
                offset_msg.data = 0.0f;
                offset_pub_->publish(offset_msg);
            }
        }

        if (publish_debug_image_) {
            publishDebug(frame, detection, valid, infer_ms);
        }

        logStats(valid, infer_ms);
    }

    void updateDetectionMetrics(const LaneDetection& detection) {
        last_raw_offset_ = detection.offset;
        last_clamped_offset_ = detection.offset;
        last_raw_lane_center_x_ = detection.lane_center_x;
        last_raw_heading_ = detection.heading_error;
        last_raw_curvature_ = detection.curvature;
        last_lane_l0_count_ = detection.lanes.size() > 0 ? static_cast<float>(detection.lanes[0].size()) : 0.0f;
        last_lane_l1_count_ = detection.lanes.size() > 1 ? static_cast<float>(detection.lanes[1].size()) : 0.0f;
        last_lane_l2_count_ = detection.lanes.size() > 2 ? static_cast<float>(detection.lanes[2].size()) : 0.0f;
        last_lane_l3_count_ = detection.lanes.size() > 3 ? static_cast<float>(detection.lanes[3].size()) : 0.0f;
    }

    void updateHeadingFilter(float measured_heading) {
        const float alpha = static_cast<float>(std::max(0.0, std::min(1.0, heading_filter_alpha_)));
        if (!has_filtered_heading_) {
            filtered_heading_ = measured_heading;
            has_filtered_heading_ = true;
        } else {
            filtered_heading_ = filtered_heading_ * (1.0f - alpha) + measured_heading * alpha;
        }
        last_filtered_heading_ = filtered_heading_;
    }

    void updateCurvatureFilter(float measured_curvature) {
        const float alpha = static_cast<float>(std::max(0.0, std::min(1.0, curvature_filter_alpha_)));
        if (!has_filtered_curvature_) {
            filtered_curvature_ = measured_curvature;
            has_filtered_curvature_ = true;
        } else {
            filtered_curvature_ = filtered_curvature_ * (1.0f - alpha) + measured_curvature * alpha;
        }
    }

    void clampDetectionStep(LaneDetection& detection, int image_width) {
        if (!detection.valid || image_width <= 0) return;

        last_step_clamped_ = false;
        last_step_delta_ = has_last_lane_center_x_ ? detection.lane_center_x - last_lane_center_x_ : 0.0f;
        if (has_last_lane_center_x_ && config_.offset_max_step_px > 0.0f) {
            const float delta = detection.lane_center_x - last_lane_center_x_;
            if (std::abs(delta) > config_.offset_max_step_px) {
                detection.lane_center_x = last_lane_center_x_ +
                    std::copysign(config_.offset_max_step_px, delta);
                detection.offset = detection.lane_center_x - static_cast<float>(image_width) * 0.5f;
                last_step_clamped_ = true;
            }
        }

        last_clamped_offset_ = detection.offset;
        last_lane_center_x_ = detection.lane_center_x;
        has_last_lane_center_x_ = true;
    }

    // Raw per-frame lane-color presence for the mission planner (which does its
    // own debouncing). Classes: 0/3 white, 1/2 yellow.
    void publishLaneColorFlags(const LaneDetection& detection) {
        const size_t min_pts = static_cast<size_t>(std::max(1, config_.min_lane_points));
        const auto laneHas = [&](size_t idx) {
            return detection.lanes.size() > idx && detection.lanes[idx].size() >= min_pts;
        };

        std_msgs::msg::Bool white_msg;
        white_msg.data = laneHas(0) || laneHas(3);
        white_lane_detected_pub_->publish(white_msg);

        std_msgs::msg::Bool yellow_msg;
        yellow_msg.data = laneHas(1) || laneHas(2);
        yellow_lane_detected_pub_->publish(yellow_msg);
    }

    // Sample window between the active BEV lane bounds, limited to the BEV dst
    // road strip and inset off the lane paint. Empty rect when bounds are
    // missing or the window collapses.
    cv::Rect warningZoneRect(const cv::Size& size) const {
        if (!has_bev_bounds_) return cv::Rect();
        int y_top = static_cast<int>(std::min(bev_dst_tl_y_, bev_dst_tr_y_));
        int y_bot = static_cast<int>(std::max(bev_dst_bl_y_, bev_dst_br_y_));
        y_top = std::max(0, std::min(size.height - 1, y_top));
        y_bot = std::max(y_top + 1, std::min(size.height, y_bot));
        int x0 = static_cast<int>(bev_left_bound_x_ + warning_inset_px_);
        int x1 = static_cast<int>(bev_right_bound_x_ - warning_inset_px_);
        x0 = std::max(0, std::min(size.width - 1, x0));
        x1 = std::max(0, std::min(size.width, x1));
        if (x1 - x0 < 8 || y_bot - y_top < 8) return cv::Rect();
        return cv::Rect(x0, y_top, x1 - x0, y_bot - y_top);
    }

    bool detectWarningZone(const cv::Mat& frame) {
        last_warning_red_ratio_ = 0.0f;
        if (!warning_zone_enable_ || bev_matrix_.empty() || !has_bev_bounds_) return false;
        const cv::Rect rect = warningZoneRect(frame.size());
        if (rect.empty()) return false;

        cv::Mat bev;
        cv::warpPerspective(frame, bev, bev_matrix_, frame.size());
        cv::Mat hsv;
        cv::cvtColor(bev(rect), hsv, cv::COLOR_BGR2HSV);
        // Red wraps hue 0: [0..h1_max] OR [h2_min..180].
        cv::Mat red_lo, red_hi;
        cv::inRange(hsv, cv::Scalar(0, warning_red_s_min_, warning_red_v_min_),
                    cv::Scalar(warning_red_h1_max_, 255, 255), red_lo);
        cv::inRange(hsv, cv::Scalar(warning_red_h2_min_, warning_red_s_min_, warning_red_v_min_),
                    cv::Scalar(180, 255, 255), red_hi);
        const double red_pixels = cv::countNonZero(red_lo) + cv::countNonZero(red_hi);
        last_warning_red_ratio_ = static_cast<float>(red_pixels / static_cast<double>(rect.area()));
        return last_warning_red_ratio_ >= warning_red_ratio_thresh_;
    }

    void publishYellow(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask_yellow;
        cv::inRange(hsv,
                    cv::Scalar(yellow_h_min_, yellow_s_min_, yellow_v_min_),
                    cv::Scalar(yellow_h_max_, yellow_s_max_, yellow_v_max_),
                    mask_yellow);

        cv::Mat roi = mask_yellow;
        if (yellow_roi_width_ > 0 && yellow_roi_height_ > 0) {
            const int x = std::max(0, yellow_roi_x_);
            const int y = std::max(0, yellow_roi_y_);
            const int w = std::min(yellow_roi_width_, frame.cols - x);
            const int h = std::min(yellow_roi_height_, frame.rows - y);
            if (w > 0 && h > 0) {
                roi = mask_yellow(cv::Rect(x, y, w, h));
            }
        }

        if (mission_state_ == 1) {
            std_msgs::msg::Bool yellow_msg;
            yellow_msg.data = cv::countNonZero(roi) >= yellow_detect_thresh_;
            yellow_detected_pub_->publish(yellow_msg);
        }
    }

    // Yellow X-cross from UFLD lane geometry: minimum BEV separation between
    // yellow_left(1) and yellow_right(2) at matching heights. Normal lane keeps
    // ~2*half_lane_width; an X crossing collapses it below yellow_cross_width_px
    // (or inverts it, which is also below threshold). A few consecutive frames
    // are required so one noisy inference can't fire the mission.
    void publishYellowCross(const LaneDetection& detection) {
        bool cross_now = false;
        // -1 = no yellow L/R pair matched this frame (distinguishes "unknown"
        // from a real 0px separation in /ufld_debug).
        last_yellow_min_sep_px_ = -1.0f;

        const size_t min_pts = static_cast<size_t>(std::max(1, config_.min_lane_points));
        if (detection.valid && detection.lanes.size() >= 3 && !bev_matrix_.empty() &&
            detection.lanes[1].size() >= min_pts && detection.lanes[2].size() >= min_pts) {
            std::vector<cv::Point2f> left_bev, right_bev;
            cv::perspectiveTransform(detection.lanes[1], left_bev, bev_matrix_);
            cv::perspectiveTransform(detection.lanes[2], right_bev, bev_matrix_);

            float min_sep = std::numeric_limits<float>::max();
            for (const auto& lp : left_bev) {
                for (const auto& rp : right_bev) {
                    if (std::fabs(lp.y - rp.y) <= static_cast<float>(yellow_cross_y_tol_px_)) {
                        min_sep = std::min(min_sep, rp.x - lp.x);
                    }
                }
            }

            if (min_sep < std::numeric_limits<float>::max()) {
                last_yellow_min_sep_px_ = min_sep;
                cross_now = min_sep < static_cast<float>(yellow_cross_width_px_);
            }
        }

        if (cross_now) {
            yellow_cross_streak_ = std::min(1000, yellow_cross_streak_ + 1);
            yellow_cross_miss_streak_ = 0;
        } else {
            yellow_cross_miss_streak_ = std::min(1000, yellow_cross_miss_streak_ + 1);
            if (yellow_cross_miss_streak_ >= yellow_cross_clear_ticks_) {
                yellow_cross_streak_ = 0;
            }
        }

        std_msgs::msg::Bool cross_msg;
        cross_msg.data = yellow_cross_streak_ >= yellow_cross_stable_ticks_;
        yellow_cross_pub_->publish(cross_msg);
    }

    void publishCompressedImage(
        const rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr& publisher,
        const cv::Mat& image,
        const std::string& frame_id) {
        if (!publisher || publisher->get_subscription_count() == 0 || image.empty()) {
            return;
        }

        std::vector<uchar> encoded;
        if (!cv::imencode(".jpg", image, encoded, {cv::IMWRITE_JPEG_QUALITY, bev_debug_quality_})) return;

        sensor_msgs::msg::CompressedImage msg;
        msg.header.stamp = get_clock()->now();
        msg.header.frame_id = frame_id;
        msg.format = "jpeg";
        msg.data.assign(encoded.begin(), encoded.end());
        publisher->publish(msg);
    }

    void publishDebug(const cv::Mat& frame, const LaneDetection& detection, bool valid, double infer_ms) {
        cv::Mat vis = frame.clone();
        static const cv::Scalar colors[] = {
            cv::Scalar(0, 255, 0),      // white_left (Green)
            cv::Scalar(0, 0, 255),      // yellow_left (Red)
            cv::Scalar(0, 255, 255),    // yellow_right (Yellow)
            cv::Scalar(255, 0, 0)       // white_right (Blue)
        };

        for (size_t i = 0; i < detection.lanes.size(); ++i) {
            const auto& lane = detection.lanes[i];
            for (const auto& p : lane) {
                cv::circle(vis, p, 2, colors[i % 4], -1);
            }
            for (size_t j = 1; j < lane.size(); ++j) {
                cv::line(vis, lane[j - 1], lane[j], colors[i % 4], 1);
            }
        }
        cv::line(vis, cv::Point(frame.cols / 2, 0), cv::Point(frame.cols / 2, frame.rows), cv::Scalar(255, 255, 0), 1);
        if (valid) {
            cv::circle(vis, cv::Point(static_cast<int>(detection.lane_center_x), static_cast<int>(frame.rows * 0.75f)),
                       4, cv::Scalar(0, 165, 255), -1);
        }
        publishCompressedImage(debug_image_pub_, vis, "camera");
        publishCompressedImage(line_overlay_image_pub_, vis, "line_overlay");
        if (publish_bev_debug_image_) {
            publishBevDebug(frame, detection, valid);
        }

        if (show_debug_) {
            cv::imshow("UFLD Line", vis);
            cv::waitKey(1);
        }
    }

    void publishBevDebug(const cv::Mat& frame, const LaneDetection& detection, bool valid) {
        if (!line_bev_image_pub_ || line_bev_image_pub_->get_subscription_count() == 0 || bev_matrix_.empty()) {
            return;
        }

        cv::Mat bev;
        cv::warpPerspective(frame, bev, bev_matrix_, frame.size());
        const auto bev_lanes = transformLanesToBev(detection);

        static const cv::Scalar colors[] = {
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 0)
        };
        for (size_t i = 0; i < bev_lanes.size(); ++i) {
            const auto& lane = bev_lanes[i];
            for (const auto& p : lane) {
                cv::circle(bev, p, 2, colors[i % 4], -1);
            }
            for (size_t j = 1; j < lane.size(); ++j) {
                cv::line(bev, lane[j - 1], lane[j], colors[i % 4], 1);
            }
        }

        if (warning_zone_enable_ && has_bev_bounds_) {
            const cv::Rect warn_rect = warningZoneRect(frame.size());
            if (!warn_rect.empty()) {
                const cv::Scalar warn_color = last_warning_zone_
                    ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 255);
                cv::rectangle(bev, warn_rect, warn_color, 1);
                char warn_buf[32];
                std::snprintf(warn_buf, sizeof(warn_buf), "red %.2f", last_warning_red_ratio_);
                cv::putText(bev, warn_buf,
                            cv::Point(warn_rect.x + 2, std::max(10, warn_rect.y + 12)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, warn_color, 1);
            }
        }

        const int center_x = frame.cols / 2;
        const int target_y = std::max(0, std::min(frame.rows - 1, static_cast<int>(frame.rows * config_.near_band)));
        cv::line(bev, cv::Point(center_x, 0), cv::Point(center_x, frame.rows), cv::Scalar(255, 255, 0), 1);
        cv::line(bev, cv::Point(0, target_y), cv::Point(frame.cols, target_y), cv::Scalar(255, 255, 0), 1);
        if (valid) {
            cv::circle(bev, cv::Point(static_cast<int>(detection.lane_center_x), target_y),
                       4, cv::Scalar(0, 165, 255), -1);
        }
        publishCompressedImage(line_bev_image_pub_, bev, "line_bev");
    }

    void logStats(bool valid, double infer_ms) {
        const auto now = get_clock()->now();
        publishDebugMetrics(valid, infer_ms);
        if (last_log_time_.nanoseconds() == 0) {
            last_log_time_ = now;
            return;
        }
        if ((now - last_log_time_).seconds() < log_period_sec_) return;

        const double success_ratio = frame_count_ > 0 ? 100.0 * static_cast<double>(success_count_) / frame_count_ : 0.0;
        RCLCPP_INFO(get_logger(),
                    "[ufld_dbg] frames=%zu success=%.1f%% last=%s raw=%.2f clamp=%.2f filt=%.2f "
                    "heading_raw=%.3f heading_filt=%.3f dyn_y=%.2f dyn_factor=%.2f center_raw=%.2f "
                    "step_delta=%.2f step_clamp=%d alpha=%.2f target_y=%.2f band=%.2f max_step=%.1f "
                    "half_w=%.1f lanes=[%.0f,%.0f,%.0f,%.0f] infer=%.1fms",
                    frame_count_, success_ratio, valid ? "OK" : "MISS",
                    last_raw_offset_, last_clamped_offset_, filtered_offset_,
                    last_raw_heading_, last_filtered_heading_, last_dynamic_target_y_, last_dynamic_factor_,
                    last_raw_lane_center_x_, last_step_delta_, last_step_clamped_ ? 1 : 0,
                    offset_filter_alpha_, config_.near_band, config_.offset_band_ratio,
                    config_.offset_max_step_px, config_.half_lane_width,
                    last_lane_l0_count_, last_lane_l1_count_, last_lane_l2_count_, last_lane_l3_count_,
                    infer_ms);
        last_log_time_ = now;
    }

    void publishDebugMetrics(bool valid, double infer_ms) {
        if (!debug_metrics_pub_) return;

        const double success_ratio = frame_count_ > 0 ? 100.0 * static_cast<double>(success_count_) / frame_count_ : 0.0;
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {
            valid ? 1.0f : 0.0f,
            last_raw_offset_,
            last_clamped_offset_,
            filtered_offset_,
            last_raw_heading_,
            last_filtered_heading_,
            last_dynamic_target_y_,
            last_dynamic_factor_,
            last_raw_lane_center_x_,
            last_step_delta_,
            last_step_clamped_ ? 1.0f : 0.0f,
            static_cast<float>(offset_filter_alpha_),
            static_cast<float>(config_.near_band),
            static_cast<float>(config_.offset_band_ratio),
            static_cast<float>(config_.offset_max_step_px),
            static_cast<float>(config_.half_lane_width),
            last_lane_l0_count_,
            last_lane_l1_count_,
            last_lane_l2_count_,
            last_lane_l3_count_,
            static_cast<float>(infer_ms),
            static_cast<float>(frame_count_),
            static_cast<float>(success_ratio),
            last_raw_curvature_,
            filtered_curvature_,
            static_cast<float>(lane_mode_),
            last_yellow_min_sep_px_,
            static_cast<float>(yellow_cross_streak_),
            yellow_cross_streak_ >= yellow_cross_stable_ticks_ ? 1.0f : 0.0f,
            static_cast<float>(yellow_cross_width_px_),
        };
        debug_metrics_pub_->publish(msg);
    }
};

}  // namespace ufld_line_pkg

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ufld_line_pkg::UfldLineNode>());
    rclcpp::shutdown();
    return 0;
}
