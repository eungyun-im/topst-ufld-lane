// Lane-following controller (no mission FSM).
//
// Control law:
//   error    = offset + clamp(heading_gain * heading, +-heading_term_limit_px)   [px]
//   kp_eff   = p_gain * straightGainScale(curvature)   (straight: straight_p_scale, curve: 1.0)
//   P        = kp_eff * error
//   I        = (i_gain / nominal_dt) * integral_seconds
//   D        = d_gain * (d_filtered_per_sec * nominal_dt)
//   steering = steering_sign * (P+I+D)/image_center * max_steering_rad
//              -> clamp -> slew limit
//   throttle = offset-risk based slowdown (unchanged from previous design)
//
// The user-facing i_gain / d_gain keep the same meaning they had at the
// nominal control period (1 / control_hz). Internally the controller
// normalizes by dt so changes in timer rate or camera rate do not change the
// closed-loop behavior as dramatically.
//
// The previous per-state gain sets, mission FSM, and error-magnitude dynamic
// P/D gain schedule were removed on purpose: the dynamic gain made loop gain
// depend on error amplitude (a limit-cycle generator) and cut damping exactly
// at zero-crossings. Curvature-based scheduling is amplitude-independent.
#include <rclcpp/rclcpp.hpp>
#include <control_msgs/msg/control.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

class ControlNode : public rclcpp::Node {
public:
    ControlNode()
        : Node("auto_control_node"),
          line_offset_(0.0f),
          line_heading_(0.0f),
          line_curvature_(0.0f),
          line_valid_(false),
          have_line_(false),
          offset_is_fresh_(false),
          prev_error_(0.0),
          has_prev_error_(false),
          integral_(0.0),
          filtered_derivative_(0.0),
          last_valid_steering_(0.0),
          last_published_steering_(0.0),
          has_last_published_steering_(false),
          filtered_throttle_(0.0f),
          has_filtered_throttle_(false)
    {
        debug_mode_ = declare_parameter<bool>("debug_mode", false);
        metrics_log_enable_ = declare_parameter<bool>("metrics_log_enable", true);
        metrics_log_period_ms_ = declare_parameter<int>("metrics_log_period_ms", 500);
        metrics_log_period_ms_ = std::max(100, metrics_log_period_ms_);

        line_offset_topic_ = declare_parameter<std::string>("line_offset_topic", "/line_offset");
        line_heading_topic_ = declare_parameter<std::string>("line_heading_topic", "/line_heading");
        line_curvature_topic_ = declare_parameter<std::string>("line_curvature_topic", "/line_curvature");
        line_valid_topic_ = declare_parameter<std::string>("line_valid_topic", "/line_valid");
        control_tuning_topic_ = declare_parameter<std::string>("control_tuning_topic", "/control_tuning");
        mission_speed_mode_topic_ = declare_parameter<std::string>("mission_speed_mode_topic", "/mission_speed_mode");
        debug_metrics_topic_ = declare_parameter<std::string>("debug_metrics_topic", "/control_debug");
        control_topic_ = declare_parameter<std::string>("control_topic", "/control");

        control_hz_ = declare_parameter<double>("control_hz", 20.0);
        line_timeout_sec_ = declare_parameter<double>("line_timeout_sec", 0.30);
        line_timeout_sec_ = std::max(0.05, std::min(2.0, line_timeout_sec_));
        line_invalid_streak_limit_ = declare_parameter<int>("line_invalid_streak_limit", 3);
        line_invalid_streak_limit_ = std::max(1, line_invalid_streak_limit_);

        // Steering geometry
        max_steering_rad_ = declare_parameter<double>("max_steering_rad", 0.45);
        image_center_ = declare_parameter<double>("image_center", 160.0);
        max_steer_step_per_cycle_ = declare_parameter<double>("max_steer_step_per_cycle", 0.05);
        max_steer_step_per_cycle_ = std::max(0.0, std::min(1.0, max_steer_step_per_cycle_));
        steering_sign_ = declare_parameter<double>("steering_sign", -1.0);
        steering_sign_ = (steering_sign_ < 0.0) ? -1.0 : 1.0;

        // PID gains (single global set)
        p_gain_ = declare_parameter<double>("p_gain", 1.20);
        i_gain_ = declare_parameter<double>("i_gain", 0.02);
        d_gain_ = declare_parameter<double>("d_gain", 0.60);
        heading_gain_ = declare_parameter<double>("heading_gain", 20.0);
        d_filter_alpha_ = declare_parameter<double>("d_filter_alpha", 0.35);
        d_filter_alpha_ = std::max(0.0, std::min(1.0, d_filter_alpha_));
        heading_term_limit_px_ = declare_parameter<double>("heading_term_limit_px", 80.0);
        heading_term_limit_px_ = std::max(0.0, std::min(320.0, heading_term_limit_px_));

        // Integral guards: integrate only near center while steering is not
        // saturated, leak so stale bias decays, clamp the I contribution in px.
        integral_error_limit_px_ = declare_parameter<double>("integral_error_limit_px", 40.0);
        integral_error_limit_px_ = std::max(0.0, integral_error_limit_px_);
        integral_term_limit_px_ = declare_parameter<double>("integral_term_limit_px", 30.0);
        integral_term_limit_px_ = std::max(0.0, integral_term_limit_px_);
        integral_leak_ = declare_parameter<double>("integral_leak", 0.995);
        integral_leak_ = std::max(0.0, std::min(1.0, integral_leak_));

        // Curvature-based straight-section P reduction. Amplitude-independent:
        // the lane's curvature stays ~0 on a straight even while the car
        // oscillates, so this cannot re-create an error-driven limit cycle.
        straight_p_scale_ = declare_parameter<double>("straight_p_scale", 0.70);
        straight_p_scale_ = std::max(0.2, std::min(1.0, straight_p_scale_));
        straight_curv_deadband_ = declare_parameter<double>("straight_curv_deadband", 0.002);
        straight_curv_deadband_ = std::max(0.0, straight_curv_deadband_);
        straight_curv_full_ = declare_parameter<double>("straight_curv_full", 0.008);
        straight_curv_full_ = std::max(straight_curv_deadband_ + 1e-5, straight_curv_full_);

        // Speed / throttle
        max_speed_ = declare_parameter<double>("max_speed", 0.35);
        min_speed_ = declare_parameter<double>("min_speed", 0.10);
        slow_speed_scale_ = declare_parameter<double>("slow_speed_scale", 0.70);
        slow_speed_scale_ = std::max(0.1, std::min(1.0, slow_speed_scale_));
        throttle_risk_scale_ = declare_parameter<double>("throttle_risk_scale", 1.0);
        throttle_offset_deadband_px_ = declare_parameter<double>("throttle_offset_deadband_px", 20.0);
        throttle_offset_full_px_ = declare_parameter<double>("throttle_offset_full_px", 120.0);
        throttle_heading_deadband_rad_ = declare_parameter<double>("throttle_heading_deadband_rad", 0.08);
        throttle_heading_full_rad_ = declare_parameter<double>("throttle_heading_full_rad", 0.50);
        throttle_derivative_deadband_px_ = declare_parameter<double>("throttle_derivative_deadband_px", 4.0);
        throttle_derivative_full_px_ = declare_parameter<double>("throttle_derivative_full_px", 35.0);
        throttle_curvature_deadband_ = declare_parameter<double>("throttle_curvature_deadband", 0.0015);
        throttle_curvature_full_ = declare_parameter<double>("throttle_curvature_full", 0.012);
        throttle_decel_alpha_ = declare_parameter<double>("throttle_decel_alpha", 0.45);
        throttle_accel_alpha_ = declare_parameter<double>("throttle_accel_alpha", 0.08);
        throttle_offset_deadband_px_ = std::max(0.0, throttle_offset_deadband_px_);
        throttle_offset_full_px_ = std::max(throttle_offset_deadband_px_ + 1.0, throttle_offset_full_px_);
        throttle_heading_deadband_rad_ = std::max(0.0, throttle_heading_deadband_rad_);
        throttle_heading_full_rad_ = std::max(throttle_heading_deadband_rad_ + 0.01, throttle_heading_full_rad_);
        throttle_derivative_deadband_px_ = std::max(0.0, throttle_derivative_deadband_px_);
        throttle_derivative_full_px_ = std::max(throttle_derivative_deadband_px_ + 1.0, throttle_derivative_full_px_);
        throttle_curvature_deadband_ = std::max(0.0, throttle_curvature_deadband_);
        throttle_curvature_full_ = std::max(throttle_curvature_deadband_ + 1e-4, throttle_curvature_full_);
        throttle_decel_alpha_ = std::max(0.0, std::min(1.0, throttle_decel_alpha_));
        throttle_accel_alpha_ = std::max(0.0, std::min(1.0, throttle_accel_alpha_));

        offset_sub_ = create_subscription<std_msgs::msg::Float32>(
            line_offset_topic_, 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) { offsetCb(msg); });
        heading_sub_ = create_subscription<std_msgs::msg::Float32>(
            line_heading_topic_, 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) { headingCb(msg); });
        curvature_sub_ = create_subscription<std_msgs::msg::Float32>(
            line_curvature_topic_, 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) { curvatureCb(msg); });
        line_valid_sub_ = create_subscription<std_msgs::msg::Bool>(
            line_valid_topic_, 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg) { lineValidCb(msg); });
        control_tuning_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
            control_tuning_topic_, 10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) { controlTuningCb(msg); });
        // Optional mission planner input. Nothing received (planner not running)
        // keeps NORMAL, so the controller drives standalone exactly as before.
        speed_mode_sub_ = create_subscription<std_msgs::msg::Int32>(
            mission_speed_mode_topic_, 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) { speed_mode_ = msg->data; });

        control_pub_ = create_publisher<control_msgs::msg::Control>(control_topic_, 10);
        debug_metrics_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(debug_metrics_topic_, 10);

        const double timer_hz = std::max(1.0, control_hz_);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(1.0 / timer_hz)),
            [this]() { run(); });

        RCLCPP_INFO(get_logger(),
                    "[control_node] Lane-follow controller started. hz=%.1f p=%.2f i=%.3f d=%.2f "
                    "h_gain=%.1f straight_scale=%.2f curv=[%.4f, %.4f] sign=%.0f",
                    timer_hz, p_gain_, i_gain_, d_gain_, heading_gain_,
                    straight_p_scale_, straight_curv_deadband_, straight_curv_full_, steering_sign_);
    }

private:
    struct Command {
        double steering = 0.0;
        float throttle = 0.0f;
    };

    struct PendingLaneSample {
        float offset = 0.0f;
        float heading = 0.0f;
        float curvature = 0.0f;
        bool have_offset = false;
        bool have_heading = false;
        bool have_curvature = false;

        void clear() {
            have_offset = false;
            have_heading = false;
            have_curvature = false;
        }
    };

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr offset_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr heading_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr curvature_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr line_valid_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr control_tuning_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr speed_mode_sub_;
    rclcpp::Publisher<control_msgs::msg::Control>::SharedPtr control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_metrics_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string line_offset_topic_, line_heading_topic_, line_curvature_topic_, line_valid_topic_;
    std::string control_tuning_topic_, debug_metrics_topic_, control_topic_;
    std::string mission_speed_mode_topic_;

    // /mission_speed_mode: 0 NORMAL, 1 SLOW, 2 STOP (default NORMAL when no planner runs)
    int speed_mode_ = 0;
    double slow_speed_scale_;

    // Lane measurement state
    float line_offset_;
    float line_heading_;
    float line_curvature_;
    bool line_valid_;
    bool have_line_;
    rclcpp::Time last_line_time_;
    int line_invalid_streak_ = 0;
    int line_invalid_streak_limit_;
    bool offset_is_fresh_;
    PendingLaneSample pending_lane_;

    // PID state
    double prev_error_;
    bool has_prev_error_;
    double integral_;
    double filtered_derivative_;
    rclcpp::Time last_pid_update_time_;
    double steering_target_ = 0.0;

    // Gains / tuning
    double p_gain_, i_gain_, d_gain_, heading_gain_;
    double d_filter_alpha_;
    double heading_term_limit_px_;
    double integral_error_limit_px_, integral_term_limit_px_, integral_leak_;
    double straight_p_scale_, straight_curv_deadband_, straight_curv_full_;
    double max_steering_rad_;
    double image_center_;
    double max_steer_step_per_cycle_;
    double steering_sign_;
    double control_hz_;
    double line_timeout_sec_;

    // Speed / throttle
    double max_speed_, min_speed_, throttle_risk_scale_;
    double throttle_offset_deadband_px_, throttle_offset_full_px_;
    double throttle_heading_deadband_rad_, throttle_heading_full_rad_;
    double throttle_derivative_deadband_px_, throttle_derivative_full_px_;
    double throttle_curvature_deadband_, throttle_curvature_full_;
    double throttle_decel_alpha_, throttle_accel_alpha_;
    float filtered_throttle_;
    bool has_filtered_throttle_;

    // Steering output state
    double last_valid_steering_;
    double last_published_steering_;
    bool has_last_published_steering_;
    double last_steering_delta_ = 0.0;

    // Debug / logging
    bool debug_mode_;
    bool metrics_log_enable_;
    int metrics_log_period_ms_;
    double last_pid_error_ = 0.0;
    double last_pid_derivative_ = 0.0;
    double last_pid_filtered_derivative_ = 0.0;
    double last_pid_p_term_ = 0.0;
    double last_pid_i_term_ = 0.0;
    double last_pid_d_term_ = 0.0;
    double last_pid_output_ = 0.0;
    double last_p_gain_ = 0.0;
    double last_d_gain_ = 0.0;
    double last_pid_dt_ = 0.0;
    double last_raw_heading_term_ = 0.0;
    double last_heading_term_ = 0.0;
    double last_effective_error_ = 0.0;
    double last_straight_scale_ = 1.0;
    bool last_derivative_updated_ = false;
    bool last_integral_updated_ = false;
    bool last_output_limited_ = false;
    bool last_steering_saturated_ = false;
    bool last_steering_slew_limited_ = false;
    bool last_sample_consumed_ = false;
    double last_raw_steering_ = 0.0;
    double last_clamped_steering_ = 0.0;
    double last_throttle_offset_risk_ = 0.0;
    double last_throttle_heading_risk_ = 0.0;
    double last_throttle_derivative_risk_ = 0.0;
    double last_throttle_curvature_risk_ = 0.0;
    double last_throttle_risk_ = 0.0;
    double last_throttle_target_ = 0.0;

    void offsetCb(const std_msgs::msg::Float32::SharedPtr msg) {
        pending_lane_.offset = msg->data;
        pending_lane_.have_offset = true;
    }

    void headingCb(const std_msgs::msg::Float32::SharedPtr msg) {
        pending_lane_.heading = msg->data;
        pending_lane_.have_heading = true;
    }

    void curvatureCb(const std_msgs::msg::Float32::SharedPtr msg) {
        pending_lane_.curvature = msg->data;
        pending_lane_.have_curvature = true;
    }

    bool hasPendingLaneSample() const {
        return pending_lane_.have_offset &&
               pending_lane_.have_heading &&
               pending_lane_.have_curvature;
    }

    void commitPendingLaneSample() {
        if (!hasPendingLaneSample()) {
            return;
        }

        line_offset_ = pending_lane_.offset;
        line_heading_ = pending_lane_.heading;
        line_curvature_ = pending_lane_.curvature;
        offset_is_fresh_ = true;
        have_line_ = true;
        last_line_time_ = now();
        pending_lane_.clear();
    }

    void lineValidCb(const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data) {
            line_invalid_streak_ = 0;
            line_valid_ = true;
            commitPendingLaneSample();
        } else if (++line_invalid_streak_ >= line_invalid_streak_limit_) {
            // Only drop to invalid after several consecutive bad frames, so a single
            // dropped/noisy detection doesn't hard-zero the throttle on an otherwise clean straight.
            line_valid_ = false;
        }
    }

    // /control_tuning protocol v2 (ufld_view.py):
    //   [0]  reserved (was FSM state selector; ignored)
    //   [1]  p_gain            [2]  d_gain           [3]  heading_gain
    //   [4]  d_filter_alpha    [5]  max_steering_rad
    //   [6]  max_speed         [7]  min_speed        [8]  throttle_risk_scale
    //   [9]  heading_term_limit_px                   [10] max_steer_step_per_cycle
    //   -- fields below are only parsed when the array has >= 21 entries, so a
    //      legacy 19-entry publisher (old dynamic-gain sliders) is ignored safely --
    //   [11] i_gain            [12] straight_p_scale [13] straight_curv_deadband
    //   [14] throttle_decel_alpha                    [15] throttle_accel_alpha
    //   [16] straight_curv_full                      [17] integral_error_limit_px
    //   [18] integral_term_limit_px                  [19] integral_leak
    //   [20] protocol version marker (2.0)
    void controlTuningCb(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (msg->data.size() < 11) return;

        p_gain_ = std::max(0.0, std::min(10.0, static_cast<double>(msg->data[1])));
        d_gain_ = std::max(0.0, std::min(10.0, static_cast<double>(msg->data[2])));
        heading_gain_ = std::max(-300.0, std::min(300.0, static_cast<double>(msg->data[3])));
        d_filter_alpha_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[4])));
        max_steering_rad_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[5])));
        max_speed_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[6])));
        min_speed_ = std::max(0.0, std::min(max_speed_, static_cast<double>(msg->data[7])));
        throttle_risk_scale_ = std::max(0.0, std::min(2.0, static_cast<double>(msg->data[8])));
        heading_term_limit_px_ = std::max(0.0, std::min(320.0, static_cast<double>(msg->data[9])));
        max_steer_step_per_cycle_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[10])));

        if (msg->data.size() >= 21) {
            i_gain_ = std::max(0.0, std::min(0.2, static_cast<double>(msg->data[11])));
            straight_p_scale_ = std::max(0.2, std::min(1.0, static_cast<double>(msg->data[12])));
            straight_curv_deadband_ = std::max(0.0, static_cast<double>(msg->data[13]));
            throttle_decel_alpha_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[14])));
            throttle_accel_alpha_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[15])));
            straight_curv_full_ = std::max(straight_curv_deadband_ + 1e-5, static_cast<double>(msg->data[16]));
            integral_error_limit_px_ = std::max(0.0, static_cast<double>(msg->data[17]));
            integral_term_limit_px_ = std::max(0.0, static_cast<double>(msg->data[18]));
            integral_leak_ = std::max(0.0, std::min(1.0, static_cast<double>(msg->data[19])));
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "[ctrl_tuning] p=%.3f i=%.4f d=%.3f h=%.1f d_alpha=%.2f max_steer=%.3f "
            "spd=[%.2f, %.2f] risk=%.2f h_lim=%.1f step=%.3f "
            "straight=%.2f curv=[%.4f, %.4f] i_lim=[%.0fpx, %.0fpx] leak=%.3f",
            p_gain_, i_gain_, d_gain_, heading_gain_, d_filter_alpha_, max_steering_rad_,
            max_speed_, min_speed_, throttle_risk_scale_, heading_term_limit_px_,
            max_steer_step_per_cycle_, straight_p_scale_,
            straight_curv_deadband_, straight_curv_full_,
            integral_error_limit_px_, integral_term_limit_px_, integral_leak_);
    }

    double normalizedOverDeadband(double value, double deadband, double full_scale) const {
        const double mag = std::fabs(value);
        if (mag <= deadband) return 0.0;
        const double span = std::max(1e-6, full_scale - deadband);
        return std::max(0.0, std::min(1.0, (mag - deadband) / span));
    }

    double nominalControlDt() const {
        return 1.0 / std::max(1.0, control_hz_);
    }

    double scaledFilterAlpha(double base_alpha, double dt) const {
        if (base_alpha <= 0.0) return 0.0;
        if (base_alpha >= 1.0) return 1.0;
        const double nominal_dt = nominalControlDt();
        return 1.0 - std::pow(1.0 - base_alpha, dt / nominal_dt);
    }

    double scaledIntegralLeak(double dt) const {
        if (integral_leak_ <= 0.0) return 0.0;
        if (integral_leak_ >= 1.0) return 1.0;
        const double nominal_dt = nominalControlDt();
        return std::pow(integral_leak_, dt / nominal_dt);
    }

    double clampSteering(double steering) const {
        return std::max(-max_steering_rad_, std::min(max_steering_rad_, steering));
    }

    double applySteeringSlewLimitValue(double steering, bool* limited = nullptr) const {
        if (limited) *limited = false;
        if (max_steer_step_per_cycle_ <= 0.0 || !has_last_published_steering_) {
            return steering;
        }

        const double delta = steering - last_published_steering_;
        const double limited_delta = std::max(-max_steer_step_per_cycle_,
                                              std::min(max_steer_step_per_cycle_, delta));
        if (limited) {
            *limited = std::fabs(limited_delta - delta) > 1e-9;
        }
        return last_published_steering_ + limited_delta;
    }

    // Straight lane (|curvature| <= deadband) -> straight_p_scale.
    // Curved lane (|curvature| >= full)       -> 1.0 (curve tuning untouched).
    double straightGainScale() const {
        const double t = normalizedOverDeadband(
            static_cast<double>(line_curvature_), straight_curv_deadband_, straight_curv_full_);
        return straight_p_scale_ + t * (1.0 - straight_p_scale_);
    }

    double calcSteering() {
        const double raw_heading_term = heading_gain_ * static_cast<double>(line_heading_);
        const double heading_term = std::max(-heading_term_limit_px_,
                                             std::min(heading_term_limit_px_, raw_heading_term));
        const double effective_error = static_cast<double>(line_offset_) + heading_term;

        const double scale = straightGainScale();
        const double kp = p_gain_ * scale;
        const double nominal_dt = nominalControlDt();
        double dt = nominal_dt;
        if (last_pid_update_time_.nanoseconds() != 0) {
            dt = (last_line_time_ - last_pid_update_time_).seconds();
        }
        if (!std::isfinite(dt) || dt <= 1e-4) {
            dt = nominal_dt;
        }
        dt = std::max(1e-4, std::min(1.0, dt));

        const double ki = (nominal_dt > 0.0) ? (i_gain_ / nominal_dt) : 0.0;

        double derivative_rate = 0.0;
        bool derivative_updated = false;
        if (has_prev_error_) {
            derivative_rate = (effective_error - prev_error_) / dt;
            derivative_updated = true;
        }
        if (derivative_updated) {
            const double alpha = scaledFilterAlpha(d_filter_alpha_, dt);
            filtered_derivative_ += alpha * (derivative_rate - filtered_derivative_);
        }

        const double legacy_filtered_derivative = filtered_derivative_ * nominal_dt;

        double leaked_integral = 0.0;
        if (ki > 0.0) {
            leaked_integral = integral_ * scaledIntegralLeak(dt);
        } else {
            integral_ = 0.0;
        }

        const double provisional_i_term = ki * leaked_integral;
        const double provisional_output =
            kp * effective_error + provisional_i_term + d_gain_ * legacy_filtered_derivative;
        const double provisional_raw_steering =
            steering_sign_ * (provisional_output / image_center_) * max_steering_rad_;
        const double provisional_clamped_steering = clampSteering(provisional_raw_steering);
        bool provisional_slew_limited = false;
        applySteeringSlewLimitValue(provisional_clamped_steering, &provisional_slew_limited);
        const bool provisional_saturated =
            std::fabs(provisional_clamped_steering - provisional_raw_steering) > 1e-9;
        const bool output_limited = provisional_saturated || provisional_slew_limited;

        bool integral_updated = false;
        if (ki > 0.0) {
            const bool within_error_band = std::fabs(effective_error) <= integral_error_limit_px_;
            const bool pushing_into_limit = output_limited && (effective_error * provisional_output > 0.0);
            if (within_error_band && !pushing_into_limit) {
                leaked_integral += effective_error * dt;
                integral_updated = true;
            }

            const double i_max = integral_term_limit_px_ / ki;
            leaked_integral = std::max(-i_max, std::min(i_max, leaked_integral));
            integral_ = leaked_integral;
        }

        const double i_term = ki * integral_;
        const double output = kp * effective_error + i_term + d_gain_ * legacy_filtered_derivative;
        const double raw_steering = steering_sign_ * (output / image_center_) * max_steering_rad_;
        const double clamped_steering = clampSteering(raw_steering);
        bool slew_limited = false;
        applySteeringSlewLimitValue(clamped_steering, &slew_limited);
        const bool saturated = std::fabs(clamped_steering - raw_steering) > 1e-9;

        prev_error_ = effective_error;
        has_prev_error_ = true;
        offset_is_fresh_ = false;
        last_pid_update_time_ = last_line_time_;

        last_straight_scale_ = scale;
        last_raw_heading_term_ = raw_heading_term;
        last_heading_term_ = heading_term;
        last_effective_error_ = effective_error;
        last_pid_error_ = effective_error;
        last_pid_derivative_ = derivative_rate * nominal_dt;
        last_pid_filtered_derivative_ = legacy_filtered_derivative;
        last_pid_p_term_ = kp * effective_error;
        last_pid_i_term_ = i_term;
        last_pid_d_term_ = d_gain_ * legacy_filtered_derivative;
        last_pid_output_ = output;
        last_p_gain_ = kp;
        last_d_gain_ = d_gain_;
        last_pid_dt_ = dt;
        last_derivative_updated_ = derivative_updated;
        last_integral_updated_ = integral_updated;
        last_output_limited_ = saturated || slew_limited;
        last_steering_saturated_ = saturated;
        last_steering_slew_limited_ = slew_limited;
        last_raw_steering_ = raw_steering;
        last_clamped_steering_ = clamped_steering;

        return clamped_steering;
    }

    float calcThrottle() {
        const double mission_scale = (speed_mode_ == 1) ? slow_speed_scale_ : 1.0;
        const float max_spd = static_cast<float>(max_speed_ * mission_scale);
        const float min_spd = static_cast<float>(min_speed_ * mission_scale);

        const double offset_risk = normalizedOverDeadband(
            static_cast<double>(line_offset_), throttle_offset_deadband_px_, throttle_offset_full_px_);
        const double heading_risk = normalizedOverDeadband(
            static_cast<double>(line_heading_), throttle_heading_deadband_rad_, throttle_heading_full_rad_);
        const double derivative_risk = normalizedOverDeadband(
            last_pid_filtered_derivative_, throttle_derivative_deadband_px_, throttle_derivative_full_px_);
        const double curvature_risk = normalizedOverDeadband(
            static_cast<double>(line_curvature_), throttle_curvature_deadband_, throttle_curvature_full_);

        last_throttle_offset_risk_ = offset_risk;
        last_throttle_heading_risk_ = heading_risk;
        last_throttle_derivative_risk_ = derivative_risk;
        last_throttle_curvature_risk_ = curvature_risk;

        if (max_spd <= min_spd) {
            filtered_throttle_ = min_spd;
            has_filtered_throttle_ = true;
            last_throttle_risk_ = 0.0;
            last_throttle_target_ = min_spd;
            return min_spd;
        }

        // Speed reduction is driven purely by how far off-center the car is (offset_risk, 0~1).
        // heading/derivative/curvature risk are still computed above for tuner visibility only.
        const double scaled_risk = std::max(0.0, std::min(1.0, offset_risk * throttle_risk_scale_));
        const float target = std::max(
            min_spd,
            std::min(max_spd, max_spd - static_cast<float>(scaled_risk) * (max_spd - min_spd)));

        if (!has_filtered_throttle_) {
            filtered_throttle_ = target;
            has_filtered_throttle_ = true;
        } else {
            const double alpha = target < filtered_throttle_ ? throttle_decel_alpha_ : throttle_accel_alpha_;
            filtered_throttle_ += static_cast<float>(alpha * (target - filtered_throttle_));
        }
        filtered_throttle_ = std::max(min_spd, std::min(max_spd, filtered_throttle_));

        last_throttle_risk_ = scaled_risk;
        last_throttle_target_ = target;
        return filtered_throttle_;
    }

    bool lineIsUsable() {
        if (!have_line_ || !line_valid_) return false;
        if (last_line_time_.nanoseconds() == 0) return false;
        return (now() - last_line_time_).seconds() <= line_timeout_sec_;
    }

    // Called every cycle while the line is unusable: keep the integral decaying
    // and drop stale derivative state so regaining the line doesn't produce a
    // derivative kick from a pre-loss error sample.
    void onLineLost() {
        integral_ *= scaledIntegralLeak(nominalControlDt());
        filtered_derivative_ = 0.0;
        has_prev_error_ = false;
        offset_is_fresh_ = false;
        steering_target_ = 0.0;
        last_output_limited_ = false;
        last_steering_saturated_ = false;
        last_steering_slew_limited_ = false;
        last_sample_consumed_ = false;
    }

    Command sanitizeCommand(Command cmd) {
        const bool steering_ok = std::isfinite(cmd.steering);
        const bool throttle_ok = std::isfinite(static_cast<double>(cmd.throttle));

        if (!steering_ok || !throttle_ok) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "[control_safety] invalid command steer=%.4f throttle=%.3f -> steer=%.4f throttle=0",
                cmd.steering, cmd.throttle, last_valid_steering_);
            cmd.steering = last_valid_steering_;
            cmd.throttle = 0.0f;
            return cmd;
        }

        cmd.steering = clampSteering(cmd.steering);
        cmd.throttle = std::max(0.0f, std::min(1.0f, cmd.throttle));
        if (cmd.throttle <= 0.0f) {
            filtered_throttle_ = 0.0f;
            has_filtered_throttle_ = true;
        }
        last_valid_steering_ = cmd.steering;
        return cmd;
    }

    Command applySteeringSlewLimit(Command cmd) {
        cmd.steering = applySteeringSlewLimitValue(cmd.steering);
        last_valid_steering_ = cmd.steering;
        return cmd;
    }

    void publishCommand(const Command& cmd) {
        control_msgs::msg::Control control_msg;
        control_msg.header.stamp = now();
        control_msg.steering = static_cast<float>(cmd.steering);
        control_msg.throttle = cmd.throttle;
        control_pub_->publish(control_msg);

        if (has_last_published_steering_) {
            last_steering_delta_ = cmd.steering - last_published_steering_;
        } else {
            last_steering_delta_ = 0.0;
            has_last_published_steering_ = true;
        }
        last_published_steering_ = cmd.steering;
    }

    void logControlMetrics(const Command& cmd) {
        publishDebugMetrics(cmd);
        if (!metrics_log_enable_) return;

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), metrics_log_period_ms_,
            "[ctrl_dbg] offset=%.2f heading=%.3f curv=%.4f scale=%.2f h_term=%.2f eff_err=%.2f "
            "kp=%.3f kd=%.3f p=%.2f i=%.2f d_filt=%.2f d_term=%.2f "
            "pid=%.2f target=%.4f steer=%.4f steer_delta=%.4f throttle=%.3f thr_risk=%.2f "
            "dt=%.3f new=%d fresh_d=%d i_up=%d lim=%d sat=%d slew=%d line_ok=%d",
            line_offset_, line_heading_, line_curvature_, last_straight_scale_,
            last_heading_term_, last_effective_error_,
            last_p_gain_, last_d_gain_,
            last_pid_p_term_, last_pid_i_term_, last_pid_filtered_derivative_, last_pid_d_term_,
            last_pid_output_, steering_target_, cmd.steering, last_steering_delta_,
            cmd.throttle, last_throttle_risk_,
            last_pid_dt_, last_sample_consumed_ ? 1 : 0,
            last_derivative_updated_ ? 1 : 0, last_integral_updated_ ? 1 : 0,
            last_output_limited_ ? 1 : 0, last_steering_saturated_ ? 1 : 0,
            last_steering_slew_limited_ ? 1 : 0, lineIsUsable() ? 1 : 0);
    }

    // Array layout keeps the legacy 31 entries (index 0 was the FSM state and is
    // now a constant 1.0) so existing monitor/ufld_view parsers keep working;
    // new fields are appended at the end.
    void publishDebugMetrics(const Command& cmd) {
        if (!debug_metrics_pub_) return;

        const float legacy_integral = static_cast<float>(
            nominalControlDt() > 0.0 ? integral_ / nominalControlDt() : 0.0);
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {
            1.0f,  // legacy mission state slot
            line_offset_,
            line_heading_,
            static_cast<float>(heading_gain_),
            static_cast<float>(last_heading_term_),
            static_cast<float>(last_effective_error_),
            static_cast<float>(last_pid_error_),
            static_cast<float>(last_p_gain_),
            static_cast<float>(last_d_gain_),
            static_cast<float>(last_pid_p_term_),
            static_cast<float>(last_pid_derivative_),
            static_cast<float>(last_pid_filtered_derivative_),
            static_cast<float>(last_pid_d_term_),
            static_cast<float>(last_pid_output_),
            static_cast<float>(cmd.steering),
            static_cast<float>(last_steering_delta_),
            cmd.throttle,
            last_derivative_updated_ ? 1.0f : 0.0f,
            static_cast<float>(image_center_),
            static_cast<float>(max_steering_rad_),
            static_cast<float>(last_raw_heading_term_),
            static_cast<float>(heading_term_limit_px_),
            static_cast<float>(max_steer_step_per_cycle_),
            lineIsUsable() ? 1.0f : 0.0f,
            static_cast<float>(last_throttle_target_),
            static_cast<float>(last_throttle_risk_),
            static_cast<float>(last_throttle_offset_risk_),
            static_cast<float>(last_throttle_heading_risk_),
            static_cast<float>(last_throttle_derivative_risk_),
            static_cast<float>(last_throttle_curvature_risk_),
            line_curvature_,
            // v2 additions
            static_cast<float>(last_pid_i_term_),
            legacy_integral,
            static_cast<float>(last_straight_scale_),
            static_cast<float>(speed_mode_),
            // v3 additions
            static_cast<float>(last_pid_dt_),
            last_sample_consumed_ ? 1.0f : 0.0f,
            last_integral_updated_ ? 1.0f : 0.0f,
            last_output_limited_ ? 1.0f : 0.0f,
            last_steering_saturated_ ? 1.0f : 0.0f,
            last_steering_slew_limited_ ? 1.0f : 0.0f,
            static_cast<float>(last_raw_steering_),
            static_cast<float>(last_clamped_steering_),
            static_cast<float>(steering_target_),
        };
        debug_metrics_pub_->publish(msg);
    }

    void run() {
        Command cmd;
        const bool line_ok = lineIsUsable();
        last_sample_consumed_ = false;

        if (line_ok) {
            if (offset_is_fresh_) {
                steering_target_ = calcSteering();
                last_sample_consumed_ = true;
            }
            cmd.steering = steering_target_;
            cmd.throttle = calcThrottle();
            if (speed_mode_ == 2) {
                // Mission stop (traffic light red / aruco): cut throttle but keep
                // steering on the lane so the car stays aligned for restart.
                cmd.throttle = 0.0f;
            }
        } else {
            onLineLost();
        }

        cmd = sanitizeCommand(cmd);
        cmd = applySteeringSlewLimit(cmd);
        publishCommand(cmd);
        logControlMetrics(cmd);

        if (debug_mode_) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
                "[control_node] steer=%.4f throttle=%.3f offset=%.2f line_ok=%d",
                cmd.steering, cmd.throttle, line_offset_, line_ok);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControlNode>());
    rclcpp::shutdown();
    return 0;
}
