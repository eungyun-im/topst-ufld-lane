#include "ufld_line_pkg/ufld_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ufld_line_pkg {

const int ROW_ANCHORS_36[36] = {
    64, 72, 76, 84, 88, 96, 100, 108, 116, 120, 128, 132,
    140, 144, 152, 160, 164, 172, 176, 184, 188, 196, 204,
    208, 216, 220, 228, 232, 240, 248, 252, 260, 264, 272,
    276, 284
};

UfldDecoder::UfldDecoder(UfldConfig config) : config_(config) {
    row_anchors_.assign(ROW_ANCHORS_36, ROW_ANCHORS_36 + 36);
}

namespace {
float verticalSpan(const std::vector<cv::Point2f>& lane) {
    if (lane.empty()) return 0.0f;
    float min_y = lane.front().y;
    float max_y = lane.front().y;
    for (const auto& p : lane) {
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    return max_y - min_y;
}

float laneHeadingDxDy(std::vector<cv::Point2f> lane) {
    if (lane.size() < 2) return 0.0f;

    std::sort(lane.begin(), lane.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    const size_t top_idx = lane.size() / 4;
    const size_t bottom_idx = lane.size() - 1 - lane.size() / 4;
    const float dy = lane[bottom_idx].y - lane[top_idx].y;
    if (std::abs(dy) < 1.0f) return 0.0f;

    return (lane[bottom_idx].x - lane[top_idx].x) / dy;
}

// Least-squares fit of x = a*y^2 + b*y + c; returns 'a', which captures how fast the
// lane's slope changes with distance (unlike the single secant slope used for heading).
float laneCurvatureQuad(const std::vector<cv::Point2f>& lane, int min_points) {
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

    // Normal equations for [a, b, c]^T: M * p = v
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

    // Cramer's rule, solving only for 'a' (first column replaced by v).
    const double det_a = v[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                         m[0][1] * (v[1] * m[2][2] - m[1][2] * v[2]) +
                         m[0][2] * (v[1] * m[2][1] - m[1][1] * v[2]);

    return static_cast<float>(det_a / det);
}

bool lanesAreDuplicate(const std::vector<cv::Point2f>& a,
                       const std::vector<cv::Point2f>& b,
                       float max_dx,
                       float min_overlap) {
    if (a.empty() || b.empty()) return false;

    int overlap = 0;
    float dx_sum = 0.0f;
    for (const auto& pa : a) {
        bool found = false;
        float best_dy = 0.0f;
        float best_dx = 0.0f;
        for (const auto& pb : b) {
            const float dy = std::abs(pa.y - pb.y);
            const float dx = std::abs(pa.x - pb.x);
            if (!found || dy < best_dy) {
                found = true;
                best_dy = dy;
                best_dx = dx;
            }
        }
        if (found && best_dy <= 8.0f) {
            ++overlap;
            dx_sum += best_dx;
        }
    }

    if (overlap == 0) return false;
    const float overlap_ratio = static_cast<float>(overlap) /
                                static_cast<float>(std::max<size_t>(1, std::min(a.size(), b.size())));
    const float mean_dx = dx_sum / static_cast<float>(overlap);
    return overlap_ratio >= min_overlap && mean_dx <= max_dx;
}
}  // namespace

LaneDetection UfldDecoder::decode(const UfldOutput& output, int image_width, int image_height) const {
    LaneDetection result;
    if (image_width <= 0 || image_height <= 0) return result;

    const auto& tensor = output.output;
    // Expected output shape is [1, 101, 36, 4]
    if (tensor.data.size() < static_cast<size_t>((config_.griding_num + 1) * config_.cls_num_per_lane * config_.num_lanes)) {
        return result;
    }

    result.lanes.resize(config_.num_lanes);

    const int GRIDING_NUM = config_.griding_num;
    const int CLS_NUM_PER_LANE = config_.cls_num_per_lane;
    const int NUM_LANES = config_.num_lanes;
    const float* out_data = tensor.data.data();

    float col_sample_w = 800.0f / (GRIDING_NUM - 1);

    for (int l = 0; l < NUM_LANES; ++l) {
        std::vector<cv::Point2f> points;

        for (int r = 0; r < CLS_NUM_PER_LANE; ++r) {
            // Flip the row index mapping as defined in model configuration:
            // near/bottom of screen maps to row 0, far/top maps to row 35.
            int tensor_r = CLS_NUM_PER_LANE - 1 - r;

            // Argmax over classes (grid index + background class)
            int argmax_idx = 0;
            float max_val = out_data[(0 * CLS_NUM_PER_LANE + tensor_r) * NUM_LANES + l];
            for (int g = 1; g <= GRIDING_NUM; ++g) {
                float val = out_data[(g * CLS_NUM_PER_LANE + tensor_r) * NUM_LANES + l];
                if (val > max_val) {
                    max_val = val;
                    argmax_idx = g;
                }
            }

            if (argmax_idx == GRIDING_NUM) {
                continue; // Background class, no lane point detected here
            }

            // Softmax over valid grid cells (0 to GRIDING_NUM - 1)
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(GRIDING_NUM);
            for (int g = 0; g < GRIDING_NUM; ++g) {
                exp_vals[g] = std::exp(out_data[(g * CLS_NUM_PER_LANE + tensor_r) * NUM_LANES + l] - max_val);
                sum_exp += exp_vals[g];
            }

            float loc = 0.0f;
            for (int g = 0; g < GRIDING_NUM; ++g) {
                float prob = exp_vals[g] / sum_exp;
                loc += prob * (g + 1);
            }

            // Map grid location to actual pixel coordinates
            float x = (loc * col_sample_w * image_width / 800.0f) - 1.0f;
            float y = (image_height * (row_anchors_[tensor_r] / 288.0f)) - 1.0f;

            if (x >= 0.0f && x < image_width && y >= 0.0f && y < image_height) {
                points.emplace_back(x, y);
            }
        }

        result.lanes[l] = points;
    }

    const float min_span = static_cast<float>(image_height) * config_.min_vertical_span_ratio;
    for (auto& lane : result.lanes) {
        if (static_cast<int>(lane.size()) < config_.min_lane_points || verticalSpan(lane) < min_span) {
            lane.clear();
        }
    }

    for (size_t i = 0; i < result.lanes.size(); ++i) {
        if (result.lanes[i].empty()) continue;
        for (size_t j = i + 1; j < result.lanes.size(); ++j) {
            if (result.lanes[j].empty()) continue;
            if (!lanesAreDuplicate(result.lanes[i], result.lanes[j],
                                   config_.dedupe_max_dx_px, config_.dedupe_min_overlap)) {
                continue;
            }
            if (result.lanes[i].size() >= result.lanes[j].size()) {
                result.lanes[j].clear();
            } else {
                result.lanes[i].clear();
                break;
            }
        }
    }

    std::vector<float> headings;
    const float heading_min_span = static_cast<float>(image_height) *
                                   std::max(config_.min_vertical_span_ratio,
                                            config_.heading_min_vertical_span_ratio);
    for (const auto& lane : result.lanes) {
        if (lane.empty()) continue;
        ++result.detected_lane_count;
        if (verticalSpan(lane) < heading_min_span) continue;
        headings.push_back(laneHeadingDxDy(lane));
    }
    if (!headings.empty()) {
        std::sort(headings.begin(), headings.end());
        result.heading_error = headings[headings.size() / 2];
        const float heading_max_abs = std::max(0.0f, config_.heading_max_abs);
        result.heading_error = std::max(-heading_max_abs, std::min(heading_max_abs, result.heading_error));
    }

    std::vector<float> curvatures;
    for (const auto& lane : result.lanes) {
        if (lane.empty() || verticalSpan(lane) < heading_min_span) continue;
        curvatures.push_back(laneCurvatureQuad(lane, config_.curvature_min_points));
    }
    if (!curvatures.empty()) {
        std::sort(curvatures.begin(), curvatures.end());
        result.curvature = curvatures[curvatures.size() / 2];
        const float curvature_max_abs = std::max(0.0f, config_.curvature_max_abs);
        result.curvature = std::max(-curvature_max_abs, std::min(curvature_max_abs, result.curvature));
    }

    // --- Class ID-based lane grouping around a target Y band for robust control offset ---
    // Class 0: white_left, Class 1: yellow_left
    // Class 2: yellow_right, Class 3: white_right
    std::vector<float> lx, rx;
    const float target_y = image_height * config_.near_band;
    const float band_half = image_height * std::max(0.0f, config_.offset_band_ratio) * 0.5f;
    const float y_min = std::max(0.0f, target_y - band_half);
    const float y_max = std::min(static_cast<float>(image_height - 1), target_y + band_half);

    for (int lane_idx : {0, 1}) {
        for (const auto& pt : result.lanes[lane_idx]) {
            if (pt.y >= y_min && pt.y <= y_max) {
                lx.push_back(pt.x);
            }
        }
    }

    for (int lane_idx : {2, 3}) {
        for (const auto& pt : result.lanes[lane_idx]) {
            if (pt.y >= y_min && pt.y <= y_max) {
                rx.push_back(pt.x);
            }
        }
    }

    float left_x = 0.0f, right_x = 0.0f;
    bool left_found = !lx.empty();
    bool right_found = !rx.empty();

    if (left_found) {
        std::sort(lx.begin(), lx.end());
        left_x = lx[lx.size() / 2];
    }
    if (right_found) {
        std::sort(rx.begin(), rx.end());
        right_x = rx[rx.size() / 2];
    }

    if (left_found && right_found) {
        result.lane_center_x = 0.5f * (left_x + right_x);
        result.valid = true;
    } else if (left_found) {
        result.lane_center_x = left_x + config_.half_lane_width;
        result.valid = true;
    } else if (right_found) {
        result.lane_center_x = right_x - config_.half_lane_width;
        result.valid = true;
    } else {
        result.lane_center_x = image_width * 0.5f;
        result.valid = false;
    }

    result.offset = result.lane_center_x - static_cast<float>(image_width) * 0.5f;

    return result;
}

}  // namespace ufld_line_pkg
