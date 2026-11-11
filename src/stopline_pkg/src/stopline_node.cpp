#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

static void onTrackbar(int, void*) {}

class StoplineNode : public rclcpp::Node {
public:
    StoplineNode()
        : Node("stopline_node"),
          flag_published_(false),
          in_cooldown_(false),
          pulse_publish_remaining_(0),
          pulse_reset_pending_(false),
          stopline_count_(0),
          mission_state_(0)
    {
        // ── ROS 파라미터 ──────────────────────────
        param("image_topic",         image_topic_,         std::string("/image_jpeg/compressed"));
        param("stopline_flag_topic", stopline_flag_topic_, std::string("/stopline_flag"));
        param("mission_topic",       mission_topic_,       std::string("/mission_state"));
        param("debug_mode",          debug_mode_,          false);

        param("roi_height",       roi_height_,       80);
        param("row_pixel_ratio",  row_pixel_ratio_,  0.25);
        param("detect_row_count", detect_row_count_, 25);
        param("cooldown_sec",     cooldown_sec_,     3.0);
        param("flag_pulse_count", flag_pulse_count_, 10);
        flag_pulse_count_ = std::max(1, flag_pulse_count_);

        // HSV 흰색
        param("white_h_min", white_h_min_, 0);
        param("white_h_max", white_h_max_, 180);
        param("white_s_min", white_s_min_, 0);
        param("white_s_max", white_s_max_, 30);
        param("white_v_min", white_v_min_, 200);
        param("white_v_max", white_v_max_, 255);

        // HSV 노란색
        param("yellow_h_min", yellow_h_min_, 20);
        param("yellow_h_max", yellow_h_max_, 35);
        param("yellow_s_min", yellow_s_min_, 50);
        param("yellow_s_max", yellow_s_max_, 255);
        param("yellow_v_min", yellow_v_min_, 100);
        param("yellow_v_max", yellow_v_max_, 255);

        // ── Sub / Pub ─────────────────────────────
        image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            image_topic_, 1, [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) { imageCallback(msg); });
        mission_sub_ = create_subscription<std_msgs::msg::Int32>(
            mission_topic_, 10, [this](const std_msgs::msg::Int32::SharedPtr msg) { missionCallback(msg); });
        stopline_flag_pub_ = create_publisher<std_msgs::msg::Bool>(stopline_flag_topic_, 10);

        if (debug_mode_) {
            // ── 트랙바 윈도우 ─────────────────────────
            cv::namedWindow("Trackbar - SL White",  cv::WINDOW_NORMAL);
            cv::namedWindow("Trackbar - SL Yellow", cv::WINDOW_NORMAL);
            cv::namedWindow("SL 01 Original",       cv::WINDOW_NORMAL);
            cv::namedWindow("SL 02 HSV Mask",       cv::WINDOW_NORMAL);
            cv::namedWindow("SL 03 ROI Debug",      cv::WINDOW_NORMAL);
            createTrackbars();
        }

        timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() { run(); });

        RCLCPP_INFO(get_logger(), "[stopline_node] Started. roi_height=%d cooldown=%.1fs",
                    roi_height_, cooldown_sec_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mission_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stopline_flag_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string image_topic_, stopline_flag_topic_, mission_topic_;
    cv::Mat frame_;
    int     mission_state_;

    bool      flag_published_;
    bool      in_cooldown_;
    int       pulse_publish_remaining_;
    bool      pulse_reset_pending_;
    rclcpp::Time cooldown_start_;

    int       stopline_count_;

    int    roi_height_;
    double row_pixel_ratio_;
    int    detect_row_count_;
    int    flag_pulse_count_;
    double cooldown_sec_;

    int white_h_min_, white_h_max_;
    int white_s_min_, white_s_max_;
    int white_v_min_, white_v_max_;
    int yellow_h_min_, yellow_h_max_;
    int yellow_s_min_, yellow_s_max_;
    int yellow_v_min_, yellow_v_max_;

    bool debug_mode_;

    template <typename T>
    void param(const std::string& name, T& value, const T& default_value) {
        value = declare_parameter<T>(name, default_value);
    }

    // ── 트랙바 ───────────────────────────────────
    void createTrackbars() {
        cv::createTrackbar("H min", "Trackbar - SL White", &white_h_min_, 180, onTrackbar);
        cv::createTrackbar("H max", "Trackbar - SL White", &white_h_max_, 180, onTrackbar);
        cv::createTrackbar("S min", "Trackbar - SL White", &white_s_min_, 255, onTrackbar);
        cv::createTrackbar("S max", "Trackbar - SL White", &white_s_max_, 255, onTrackbar);
        cv::createTrackbar("V min", "Trackbar - SL White", &white_v_min_, 255, onTrackbar);
        cv::createTrackbar("V max", "Trackbar - SL White", &white_v_max_, 255, onTrackbar);

        cv::createTrackbar("H min", "Trackbar - SL Yellow", &yellow_h_min_, 180, onTrackbar);
        cv::createTrackbar("H max", "Trackbar - SL Yellow", &yellow_h_max_, 180, onTrackbar);
        cv::createTrackbar("S min", "Trackbar - SL Yellow", &yellow_s_min_, 255, onTrackbar);
        cv::createTrackbar("S max", "Trackbar - SL Yellow", &yellow_s_max_, 255, onTrackbar);
        cv::createTrackbar("V min", "Trackbar - SL Yellow", &yellow_v_min_, 255, onTrackbar);
        cv::createTrackbar("V max", "Trackbar - SL Yellow", &yellow_v_max_, 255, onTrackbar);
    }

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            frame_ = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        } catch (const cv::Exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                 "[stopline_node] Image decode error: %s", e.what());
        }
    }

    void missionCallback(const std_msgs::msg::Int32::SharedPtr msg) {
        mission_state_ = msg->data;
    }

    void publishStoplineFlag(bool detected) {
        std_msgs::msg::Bool msg;
        msg.data = detected;
        stopline_flag_pub_->publish(msg);
        RCLCPP_WARN(get_logger(), "[stopline_node] /stopline_flag -> %s", detected ? "true" : "false");
    }

    void publishPendingPulse() {
        if (pulse_publish_remaining_ > 0) {
            publishStoplineFlag(true);
            --pulse_publish_remaining_;
            if (pulse_publish_remaining_ == 0) {
                pulse_reset_pending_ = true;
            }
            return;
        }

        if (pulse_reset_pending_) {
            publishStoplineFlag(false);
            pulse_reset_pending_ = false;
        }
    }

    cv::Mat buildStopMask(const cv::Mat& hsv) {
        cv::Mat mask_white, mask_yellow, result;
        cv::inRange(hsv,
            cv::Scalar(white_h_min_, white_s_min_, white_v_min_),
            cv::Scalar(white_h_max_, white_s_max_, white_v_max_), mask_white);
        cv::inRange(hsv,
            cv::Scalar(yellow_h_min_, yellow_s_min_, yellow_v_min_),
            cv::Scalar(yellow_h_max_, yellow_s_max_, yellow_v_max_), mask_yellow);
        cv::bitwise_or(mask_white, mask_yellow, result);
        return result;
    }

    cv::Mat maskToBgr(const cv::Mat& mask) {
        cv::Mat bgr;
        cv::cvtColor(mask, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    cv::Mat buildRoiStatusImage(const cv::Mat& binary, const std::string& status) {
        cv::Mat roi_mask = cv::Mat::zeros(binary.size(), binary.type());
        int h = binary.rows, w = binary.cols;
        int roi_y = std::max(0, h - roi_height_);
        cv::Rect roi_rect(0, roi_y, w, h - roi_y);
        binary(roi_rect).copyTo(roi_mask(roi_rect));

        cv::Mat roi_vis = maskToBgr(roi_mask);
        cv::rectangle(roi_vis, roi_rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(roi_vis, status, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        return roi_vis;
    }

    void showDebugImages(const cv::Mat& original,
                         const cv::Mat& hsv_mask,
                         const cv::Mat& roi_debug) {
        cv::imshow("SL 01 Original", original);
        cv::imshow("SL 02 HSV Mask", hsv_mask);
        cv::imshow("SL 03 ROI Debug", roi_debug);
        cv::waitKey(1);
    }

    bool detectStopLine(const cv::Mat& binary, int& out_count, int& out_threshold,
                        cv::Mat& debug_img) {
        int h = binary.rows, w = binary.cols;
        int roi_y = std::max(0, h - roi_height_);
        cv::Rect roi_rect(0, roi_y, w, h - roi_y);
        cv::Mat roi = binary(roi_rect);
        
        if (debug_mode_) {
            cv::Mat roi_mask = cv::Mat::zeros(binary.size(), binary.type());
            roi.copyTo(roi_mask(roi_rect));
            debug_img = maskToBgr(roi_mask);
        }

        cv::Mat hist;
        cv::reduce(roi, hist, 1, cv::REDUCE_SUM, CV_32S);

        out_threshold = static_cast<int>(w * 255 * row_pixel_ratio_);
        out_count = 0;

        for (int i = 0; i < hist.rows; ++i) {
            if (hist.at<int>(i, 0) > out_threshold) {
                out_count++;
                if (debug_mode_) {
                    cv::line(debug_img, cv::Point(0, roi_y + i), cv::Point(w, roi_y + i),
                             cv::Scalar(0, 0, 255), 1);
                }
            }
        }
        
        bool detected = (out_count >= detect_row_count_);
        
        if (debug_mode_) {
            cv::rectangle(debug_img, roi_rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(debug_img,
                        "rows=" + std::to_string(out_count) +
                        " / need>=" + std::to_string(detect_row_count_),
                        cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(255, 255, 0), 2);
            cv::putText(debug_img,
                        "row_threshold=" + std::to_string(out_threshold) +
                        " ratio=" + std::to_string(row_pixel_ratio_).substr(0, 4),
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                        0.55, cv::Scalar(255, 255, 0), 2);
            cv::putText(debug_img,
                        detected ? "DETECTED" : "not detected",
                        cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, detected ? cv::Scalar(0, 0, 255) : cv::Scalar(200, 200, 200), 2);
        }
        return detected;
    }

    void run() {
        if (frame_.empty()) return;

        publishPendingPulse();

        // state 2,4,6 (정지선 감지 구간)일 때만 동작
        if (mission_state_ != 2 && mission_state_ != 4 && mission_state_ != 6) {
            if (debug_mode_) {
                cv::Mat hsv;
                cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);
                cv::Mat stop_mask = buildStopMask(hsv);
                cv::Mat blurred, binary;
                cv::GaussianBlur(stop_mask, blurred, cv::Size(5, 5), 0);
                cv::threshold(blurred, binary, 127, 255, cv::THRESH_BINARY);
                cv::Mat roi_debug = buildRoiStatusImage(
                    binary,
                    "inactive state=" + std::to_string(mission_state_));
                showDebugImages(frame_, maskToBgr(stop_mask), roi_debug);
            }
            return;
        }

        // ── 쿨다운 해제 ──────────────────────────
        if (in_cooldown_) {
            double elapsed = (now() - cooldown_start_).seconds();
            if (elapsed >= cooldown_sec_) {
                in_cooldown_    = false;
                flag_published_ = false;
                RCLCPP_INFO(get_logger(), "[stopline_node] Cooldown ended. count=%d", stopline_count_);
            } else {
                if (debug_mode_) {
                    cv::Mat hsv;
                    cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);
                    cv::Mat stop_mask = buildStopMask(hsv);
                    cv::Mat blurred, binary;
                    cv::GaussianBlur(stop_mask, blurred, cv::Size(5, 5), 0);
                    cv::threshold(blurred, binary, 127, 255, cv::THRESH_BINARY);
                    cv::Mat roi_debug = buildRoiStatusImage(binary, "cooldown");
                    showDebugImages(frame_, maskToBgr(stop_mask), roi_debug);
                }
                return;
            }
        }

        cv::Mat hsv;
        cv::cvtColor(frame_, hsv, cv::COLOR_BGR2HSV);
        cv::Mat stop_mask = buildStopMask(hsv);
        cv::Mat blurred, binary;
        cv::GaussianBlur(stop_mask, blurred, cv::Size(5, 5), 0);
        cv::threshold(blurred, binary, 127, 255, cv::THRESH_BINARY);

        int cur_count = 0, threshold = 0;
        cv::Mat debug_img;
        bool detected = detectStopLine(binary, cur_count, threshold, debug_img);

        if (debug_mode_) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
                "[stopline_node] stopline_count=%d  pixel_count=%d / threshold=%d",
                stopline_count_, cur_count, threshold);
        }

        if (detected && !flag_published_) {
            stopline_count_++;
            RCLCPP_WARN(get_logger(), "[stopline_node] STOPLINE DETECTED! total count=%d", stopline_count_);

            // true는 짧은 펄스로만 보내고, 쿨다운 동안에는 false 상태로 둔다.
            pulse_publish_remaining_ = flag_pulse_count_;
            pulse_reset_pending_ = false;

            flag_published_ = true;
            in_cooldown_    = true;
            cooldown_start_ = now();

            if (debug_mode_) {
                cv::putText(debug_img,
                            "STOPLINE #" + std::to_string(stopline_count_),
                            cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
                            1.2, cv::Scalar(0, 0, 255), 3);
            }
        } else if (!detected) {
            if (debug_mode_) {
                std::string status =
                    "count=" + std::to_string(cur_count) +
                    " / need>=" + std::to_string(detect_row_count_) +
                    "  total=" + std::to_string(stopline_count_);
                cv::putText(debug_img, status, cv::Point(10, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(255, 255, 0), 2);
            }
        }

        if (debug_mode_) {
            showDebugImages(frame_, maskToBgr(stop_mask), debug_img);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StoplineNode>());
    rclcpp::shutdown();
    return 0;
}