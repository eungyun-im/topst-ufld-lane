#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "ufld_line_pkg/ufld_types.hpp"

namespace ufld_line_pkg {

class UfldDecoder {
public:
    explicit UfldDecoder(UfldConfig config);

    LaneDetection decode(const UfldOutput& output, int image_width, int image_height) const;

private:
    UfldConfig config_;
    std::vector<float> row_anchors_;
};

}  // namespace ufld_line_pkg
