#include "ufld_line_pkg/tnn_ufld_runner.hpp"

namespace ufld_line_pkg {

struct TnnUfldRunner::Impl {};

TnnUfldRunner::TnnUfldRunner() : impl_(new Impl()) {}
TnnUfldRunner::~TnnUfldRunner() = default;

bool TnnUfldRunner::init(const std::string&, const std::string&, const std::string&,
                         const std::string&, const UfldConfig&, bool, int, std::string* error) {
    if (error) {
        *error = "ufld_line_pkg was built without TNN. Rebuild with "
                 "--cmake-args -DUFLD_USE_TNN=ON -DTNN_ROOT=/path/to/TNN";
    }
    return false;
}

bool TnnUfldRunner::infer(const cv::Mat&, UfldOutput*, double*, std::string* error) {
    if (error) {
        *error = "TNN backend is not compiled in";
    }
    return false;
}

}  // namespace ufld_line_pkg
