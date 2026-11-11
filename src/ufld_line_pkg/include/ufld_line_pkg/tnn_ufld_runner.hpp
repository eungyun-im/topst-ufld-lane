#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include "ufld_line_pkg/ufld_types.hpp"

namespace ufld_line_pkg {

class TnnUfldRunner {
public:
    TnnUfldRunner();
    ~TnnUfldRunner();

    bool init(const std::string& proto_path,
              const std::string& model_path,
              const std::string& input_name,
              const std::string& output_name,
              const UfldConfig& config,
              bool use_opencl,
              int cpu_threads,
              std::string* error);

    bool infer(const cv::Mat& bgr_image, UfldOutput* output, double* infer_ms, std::string* error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ufld_line_pkg
