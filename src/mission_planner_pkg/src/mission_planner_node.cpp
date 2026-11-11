// Mission planner: turns detection flags into two simple mode topics.
// The controller (control_pkg) and the lane detector (ufld_line_pkg) stay
// dumb executors; all mission sequencing lives here.
//
//   inputs (std_msgs/Bool):
//     /traffic_light_green, /traffic_light_red, /aruco_flag   <- yolo_pkg
//     /white_lane_detected, /yellow_lane_detected             <- ufld_line_node
//     /warning_zone_detected                                  <- ufld_line_node (red road in BEV)
//     /yellow_cross_detected                                  <- (future detector; absent = false)
//   outputs (std_msgs/Int32):
//     /mission_lane_mode   0 WHITE  1 YELLOW  2 YELLOW_RIGHT_ONLY  3 YELLOW_LEFT_ONLY
//     /mission_speed_mode  0 NORMAL 1 SLOW    2 STOP
//
// Speed priority: traffic-light stop (red -> hold until green) > aruco timed stop
// > warning-zone SLOW (control scales speed by slow_speed_scale, default 0.70) > normal.
// Lane missions: white lane by default; switch to yellow when yellow is stable
// and white is gone; while in yellow, switch back as soon as white is stable
// again (yellow may still be visible). Yellow X-crossings force one-sided lane
// tracking for a short window (1st: right only, 2nd: left only).
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

class MissionPlannerNode : public rclcpp::Node {
public:
    MissionPlannerNode() : Node("mission_planner_node") {
        green_topic_ = declare_parameter<std::string>("traffic_light_green_topic", "/traffic_light_green");
        red_topic_ = declare_parameter<std::string>("traffic_light_red_topic", "/traffic_light_red");
        aruco_topic_ = declare_parameter<std::string>("aruco_flag_topic", "/aruco_flag");
        white_lane_topic_ = declare_parameter<std::string>("white_lane_topic", "/white_lane_detected");
        yellow_lane_topic_ = declare_parameter<std::string>("yellow_lane_topic", "/yellow_lane_detected");
        yellow_cross_topic_ = declare_parameter<std::string>("yellow_cross_topic", "/yellow_cross_detected");
        warning_zone_topic_ = declare_parameter<std::string>("warning_zone_topic", "/warning_zone_detected");
        lane_mode_topic_ = declare_parameter<std::string>("lane_mode_topic", "/mission_lane_mode");
        speed_mode_topic_ = declare_parameter<std::string>("speed_mode_topic", "/mission_speed_mode");

        planner_hz_ = declare_parameter<double>("planner_hz", 20.0);
        white_stable_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("white_stable_ticks", 3)));
        yellow_stable_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("yellow_stable_ticks", 3)));
        force_lane_duration_sec_ = std::max(0.1, declare_parameter<double>("force_lane_duration_sec", 2.0));
        aruco_stop_duration_sec_ = std::max(0.1, declare_parameter<double>("aruco_stop_duration_sec", 3.0));
        // Both mode transitions trigger on "the other color is stable" alone, so
        // in an overlap zone (both colors visible) the mode would chatter every
        // tick without a cooldown after each switch.
        mode_switch_cooldown_sec_ = std::max(0.0, declare_parameter<double>("mode_switch_cooldown_sec", 3.0));
        // One physical X crossing can still produce multiple rising edges on
        // /yellow_cross_detected (detector flicker); edges within this window
        // of the last counted cross are ignored so the RIGHT/LEFT alternation
        // stays one count per crossing.
        cross_event_cooldown_sec_ = std::max(0.0, declare_parameter<double>("cross_event_cooldown_sec", 5.0));
        // Warning zone (red road section): asymmetric debounce like the lane
        // flags — N consecutive true ticks to enter SLOW, M consecutive false
        // ticks to release, so single-frame detector flicker can't chatter the
        // speed mode while driving across the zone.
        warning_on_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("warning_on_ticks", 3)));
        warning_off_ticks_ = std::max(1, static_cast<int>(declare_parameter<int>("warning_off_ticks", 5)));
        // Hold STOP from boot until the first green light. Set false to drive immediately.
        start_stopped_ = declare_parameter<bool>("start_stopped", true);
        traffic_stop_active_ = start_stopped_;

        auto sub = [this](const std::string& topic, bool& flag) {
            return create_subscription<std_msgs::msg::Bool>(
                topic, 10, [&flag](const std_msgs::msg::Bool::SharedPtr msg) { flag = msg->data; });
        };
        green_sub_ = sub(green_topic_, green_detected_);
        red_sub_ = sub(red_topic_, red_detected_);
        aruco_sub_ = sub(aruco_topic_, aruco_detected_);
        white_lane_sub_ = sub(white_lane_topic_, white_lane_detected_);
        yellow_lane_sub_ = sub(yellow_lane_topic_, yellow_lane_detected_);
        yellow_cross_sub_ = sub(yellow_cross_topic_, yellow_cross_detected_);
        warning_zone_sub_ = sub(warning_zone_topic_, warning_zone_detected_);

        lane_mode_pub_ = create_publisher<std_msgs::msg::Int32>(lane_mode_topic_, 10);
        speed_mode_pub_ = create_publisher<std_msgs::msg::Int32>(speed_mode_topic_, 10);

        const double hz = std::max(1.0, planner_hz_);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(1.0 / hz)),
            [this]() { tick(); });

        RCLCPP_INFO(get_logger(),
                    "[mission_planner] started. hz=%.1f start_stopped=%d "
                    "white_ticks=%d yellow_ticks=%d force=%.1fs aruco_stop=%.1fs cross_cd=%.1fs",
                    hz, start_stopped_ ? 1 : 0, white_stable_ticks_, yellow_stable_ticks_,
                    force_lane_duration_sec_, aruco_stop_duration_sec_, cross_event_cooldown_sec_);
    }

private:
    enum MissionMode { WHITE_MODE = 0, YELLOW_MODE = 1 };
    enum LaneMode { LANE_WHITE = 0, LANE_YELLOW = 1, LANE_YELLOW_RIGHT_ONLY = 2, LANE_YELLOW_LEFT_ONLY = 3 };
    enum SpeedMode { SPEED_NORMAL = 0, SPEED_SLOW = 1, SPEED_STOP = 2 };

    std::string green_topic_, red_topic_, aruco_topic_;
    std::string white_lane_topic_, yellow_lane_topic_, yellow_cross_topic_;
    std::string warning_zone_topic_;
    std::string lane_mode_topic_, speed_mode_topic_;
    double planner_hz_;
    int white_stable_ticks_, yellow_stable_ticks_;
    double force_lane_duration_sec_, aruco_stop_duration_sec_;
    double mode_switch_cooldown_sec_;
    double cross_event_cooldown_sec_;
    bool has_cross_event_ = false;
    rclcpp::Time last_cross_event_time_;
    bool start_stopped_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr green_sub_, red_sub_, aruco_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr white_lane_sub_, yellow_lane_sub_, yellow_cross_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr warning_zone_sub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr lane_mode_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr speed_mode_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    bool green_detected_ = false;
    bool red_detected_ = false;
    bool aruco_detected_ = false;
    bool white_lane_detected_ = false;
    bool yellow_lane_detected_ = false;
    bool yellow_cross_detected_ = false;
    bool warning_zone_detected_ = false;

    int warning_on_ticks_ = 3;
    int warning_off_ticks_ = 5;
    int warning_on_count_ = 0;
    int warning_off_count_ = 0;
    bool warning_zone_active_ = false;

    bool prev_green_ = false;
    bool prev_red_ = false;
    bool prev_aruco_ = false;
    bool prev_yellow_cross_ = false;

    MissionMode mission_mode_ = WHITE_MODE;
    LaneMode lane_mode_ = LANE_WHITE;
    SpeedMode speed_mode_ = SPEED_STOP;

    int white_detect_count_ = 0;
    int yellow_detect_count_ = 0;
    int yellow_cross_count_ = 0;

    bool force_lane_active_ = false;
    LaneMode force_lane_mode_ = LANE_YELLOW;
    rclcpp::Time force_lane_start_time_;

    bool has_mode_switched_ = false;
    rclcpp::Time last_mode_switch_time_;

    bool traffic_stop_active_ = false;
    bool aruco_stop_active_ = false;
    rclcpp::Time aruco_stop_start_time_;

    void tick() {
        updateLaneDebounce();
        updateMissionMode();
        updateSpeedMode();

        std_msgs::msg::Int32 lane_msg;
        lane_msg.data = static_cast<int>(lane_mode_);
        lane_mode_pub_->publish(lane_msg);

        std_msgs::msg::Int32 speed_msg;
        speed_msg.data = static_cast<int>(speed_mode_);
        speed_mode_pub_->publish(speed_msg);

        prev_green_ = green_detected_;
        prev_red_ = red_detected_;
        prev_aruco_ = aruco_detected_;
        prev_yellow_cross_ = yellow_cross_detected_;
    }

    void updateLaneDebounce() {
        white_detect_count_ = white_lane_detected_ ? std::min(100, white_detect_count_ + 1) : 0;
        yellow_detect_count_ = yellow_lane_detected_ ? std::min(100, yellow_detect_count_ + 1) : 0;
    }

    bool whiteStable() const { return white_detect_count_ >= white_stable_ticks_; }
    bool yellowStable() const { return yellow_detect_count_ >= yellow_stable_ticks_; }

    bool modeSwitchAllowed() {
        return !has_mode_switched_ ||
               (now() - last_mode_switch_time_).seconds() >= mode_switch_cooldown_sec_;
    }

    void markModeSwitch() {
        has_mode_switched_ = true;
        last_mode_switch_time_ = now();
    }

    void updateMissionMode() {
        switch (mission_mode_) {
        case WHITE_MODE:
            lane_mode_ = LANE_WHITE;
            // Enter yellow as soon as yellow is stable — white may still be
            // visible at the boundary; the switch cooldown prevents chattering
            // while both colors overlap.
            if (yellowStable() && modeSwitchAllowed()) {
                mission_mode_ = YELLOW_MODE;
                yellow_cross_count_ = 0;
                force_lane_active_ = false;
                lane_mode_ = LANE_YELLOW;
                markModeSwitch();
                RCLCPP_INFO(get_logger(), "[mission_planner] WHITE_MODE -> YELLOW_MODE");
            }
            break;

        case YELLOW_MODE:
            updateYellowCrossMission();

            // Return to white as soon as white is stable again — yellow may
            // still be visible at the transition, that must not block it.
            if (!force_lane_active_ && !yellowStable() && modeSwitchAllowed()) {
                mission_mode_ = WHITE_MODE;
                yellow_cross_count_ = 0;
                lane_mode_ = LANE_WHITE;
                markModeSwitch();
                RCLCPP_INFO(get_logger(), "[mission_planner] YELLOW_MODE -> WHITE_MODE");
                break;
            }

            if (force_lane_active_) {
                const double elapsed = (now() - force_lane_start_time_).seconds();
                if (elapsed < force_lane_duration_sec_) {
                    lane_mode_ = force_lane_mode_;
                } else {
                    force_lane_active_ = false;
                    lane_mode_ = LANE_YELLOW;
                    RCLCPP_INFO(get_logger(), "[mission_planner] force lane finished");
                }
            } else {
                lane_mode_ = LANE_YELLOW;
            }
            break;
        }
    }

    void updateYellowCrossMission() {
        if (yellow_cross_detected_ && !prev_yellow_cross_) {
            if (has_cross_event_ &&
                (now() - last_cross_event_time_).seconds() < cross_event_cooldown_sec_) {
                RCLCPP_INFO(get_logger(),
                            "[mission_planner] yellow X edge ignored (cooldown %.1fs)",
                            cross_event_cooldown_sec_);
                return;
            }
            has_cross_event_ = true;
            last_cross_event_time_ = now();
            yellow_cross_count_++;
            if (yellow_cross_count_ > 2) yellow_cross_count_ = 1;

            force_lane_active_ = true;
            force_lane_start_time_ = now();
            if (yellow_cross_count_ == 1) {
                force_lane_mode_ = LANE_YELLOW_RIGHT_ONLY;
                RCLCPP_INFO(get_logger(), "[mission_planner] yellow X #1: RIGHT only %.1fs",
                            force_lane_duration_sec_);
            } else {
                force_lane_mode_ = LANE_YELLOW_LEFT_ONLY;
                RCLCPP_INFO(get_logger(), "[mission_planner] yellow X #2: LEFT only %.1fs",
                            force_lane_duration_sec_);
            }
        }
    }

    void updateSpeedMode() {
        // Traffic light: red latches STOP, only green releases it.
        if (red_detected_ && !prev_red_ && !traffic_stop_active_) {
            traffic_stop_active_ = true;
            RCLCPP_INFO(get_logger(), "[mission_planner] RED light: STOP until green");
        }
        if (green_detected_ && !prev_green_ && traffic_stop_active_) {
            traffic_stop_active_ = false;
            RCLCPP_INFO(get_logger(), "[mission_planner] GREEN light: GO");
        }

        // Aruco: timed stop.
        if (aruco_detected_ && !prev_aruco_ && !aruco_stop_active_) {
            aruco_stop_active_ = true;
            aruco_stop_start_time_ = now();
            RCLCPP_INFO(get_logger(), "[mission_planner] aruco: STOP for %.1fs", aruco_stop_duration_sec_);
        }
        if (aruco_stop_active_ &&
            (now() - aruco_stop_start_time_).seconds() >= aruco_stop_duration_sec_) {
            aruco_stop_active_ = false;
            RCLCPP_INFO(get_logger(), "[mission_planner] aruco stop finished");
        }

        // Warning zone: red road section -> SLOW while inside (asymmetric debounce).
        if (warning_zone_detected_) {
            warning_on_count_ = std::min(1000, warning_on_count_ + 1);
            warning_off_count_ = 0;
        } else {
            warning_off_count_ = std::min(1000, warning_off_count_ + 1);
            warning_on_count_ = 0;
        }
        if (!warning_zone_active_ && warning_on_count_ >= warning_on_ticks_) {
            warning_zone_active_ = true;
            RCLCPP_INFO(get_logger(), "[mission_planner] warning zone: SLOW (70%% speed)");
        }
        if (warning_zone_active_ && warning_off_count_ >= warning_off_ticks_) {
            warning_zone_active_ = false;
            RCLCPP_INFO(get_logger(), "[mission_planner] warning zone cleared: NORMAL");
        }

        if (traffic_stop_active_ || aruco_stop_active_) {
            speed_mode_ = SPEED_STOP;
        } else if (warning_zone_active_) {
            speed_mode_ = SPEED_SLOW;
        } else {
            speed_mode_ = SPEED_NORMAL;
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MissionPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
