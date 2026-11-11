#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace ufld_line_pkg {

struct UfldConfig {
    int input_width = 320;
    int input_height = 160;
    int num_lanes = 4;
    int griding_num = 100;
    int cls_num_per_lane = 36;
    float near_band = 0.70f;       // target y-coordinate fraction of image height for control offset
    float offset_band_ratio = 0.16f;
    float offset_max_step_px = 35.0f;
    float half_lane_width = 70.0f; // fallback offset in pixels if only one lane is detected
    int min_lane_points = 4;
    float min_vertical_span_ratio = 0.08f;
    float heading_min_vertical_span_ratio = 0.18f;
    float heading_max_abs = 0.50f;
    float dedupe_max_dx_px = 28.0f;
    float dedupe_min_overlap = 0.45f;
    int curvature_min_points = 6;    // quadratic fit needs more samples than the 2-point heading slope
    float curvature_max_abs = 0.02f; // clamp on fitted x = a*y^2 + b*y + c coefficient 'a'
};

struct UfldTensor {
    std::vector<int> dims;
    std::vector<float> data;
};

struct UfldOutput {
    UfldTensor output; // UFLD v1 single output tensor [1, griding_num + 1, cls_num_per_lane, num_lanes]
};

struct LaneDetection {
    std::vector<std::vector<cv::Point2f>> lanes; // lanes[0]: white_left, lanes[1]: yellow_left, etc.
    bool valid = false;
    float lane_center_x = 0.0f;
    float offset = 0.0f;
    float heading_error = 0.0f; // dx/dy of the visible lane centerline; positive means lane moves right toward the bottom.
    float curvature = 0.0f;     // quadratic coefficient 'a' of x=a*y^2+b*y+c fit to the lane; captures bend rate, unlike heading.
    int detected_lane_count = 0;
};

}  // namespace ufld_line_pkg
