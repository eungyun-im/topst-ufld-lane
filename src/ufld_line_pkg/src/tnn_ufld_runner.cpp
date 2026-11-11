#include "ufld_line_pkg/tnn_ufld_runner.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>

#include <opencv2/imgproc.hpp>

#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/mat.h"
#include "tnn/core/tnn.h"

// Set device type depending on host vs target CPU architecture
#if defined(__arm__) || defined(__aarch64__)
#define TNN_DEVICE_CPU TNN_NS::DEVICE_ARM
#else
#define TNN_DEVICE_CPU TNN_NS::DEVICE_X86
#endif

namespace ufld_line_pkg {

namespace {
std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

size_t elementCount(const std::vector<int>& dims) {
    if (dims.empty()) return 0;
    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                           [](size_t a, int b) { return a * static_cast<size_t>(std::max(0, b)); });
}

TNN_NS::DimsVector toDims(const std::vector<int>& dims) {
    TNN_NS::DimsVector out;
    out.reserve(dims.size());
    for (int value : dims) out.push_back(value);
    return out;
}
}  // namespace

struct TnnUfldRunner::Impl {
    UfldConfig config;
    std::string input_name;
    std::string output_name;
    std::shared_ptr<TNN_NS::TNN> net;
    std::shared_ptr<TNN_NS::Instance> instance;
    std::vector<float> input_buffer;

    bool copyOutput(const std::string& name, UfldTensor* tensor, std::string* error) {
        std::shared_ptr<TNN_NS::Mat> mat;
        auto status = instance->GetOutputMat(
            mat, TNN_NS::MatConvertParam(), name, TNN_DEVICE_CPU, TNN_NS::NCHW_FLOAT);
        if (status != TNN_NS::TNN_OK || !mat) {
            if (error) *error = "GetOutputMat failed for " + name + ": " + status.description();
            return false;
        }

        tensor->dims = mat->GetDims();
        const size_t count = elementCount(tensor->dims);
        const float* data = static_cast<const float*>(mat->GetData());
        if (!data || count == 0) {
            if (error) *error = "empty output tensor: " + name;
            return false;
        }
        tensor->data.assign(data, data + count);
        return true;
    }
};

TnnUfldRunner::TnnUfldRunner() : impl_(new Impl()) {}
TnnUfldRunner::~TnnUfldRunner() = default;

bool TnnUfldRunner::init(const std::string& proto_path,
                         const std::string& model_path,
                         const std::string& input_name,
                         const std::string& output_name,
                         const UfldConfig& config,
                         bool use_opencl,
                         int cpu_threads,
                         std::string* error) {
    const std::string proto = readFile(proto_path);
    const std::string model = readFile(model_path);
    if (proto.empty() || model.empty()) {
        if (error) *error = "failed to read TNN model files: " + proto_path + ", " + model_path;
        return false;
    }

    impl_->config = config;
    impl_->input_name = input_name;
    impl_->output_name = output_name;
    impl_->input_buffer.resize(static_cast<size_t>(config.input_width * config.input_height * 3));

    TNN_NS::ModelConfig model_config;
    model_config.model_type = TNN_NS::MODEL_TYPE_TNN;
    model_config.params = {proto, model};

    impl_->net = std::make_shared<TNN_NS::TNN>();
    auto status = impl_->net->Init(model_config);
    if (status != TNN_NS::TNN_OK) {
        if (error) *error = "TNN Init failed: " + status.description();
        return false;
    }

    TNN_NS::NetworkConfig network_config;
    network_config.device_type = use_opencl ? TNN_NS::DEVICE_OPENCL : TNN_DEVICE_CPU;

    TNN_NS::InputShapesMap input_shapes;
    input_shapes[input_name] = toDims({1, 3, config.input_height, config.input_width});

    impl_->instance = impl_->net->CreateInst(network_config, status, input_shapes);
    if (status != TNN_NS::TNN_OK || !impl_->instance) {
        if (error) *error = "CreateInst failed: " + status.description();
        return false;
    }
    if (!use_opencl && cpu_threads > 0) {
        impl_->instance->SetCpuNumThreads(cpu_threads);
    }

    return true;
}

bool TnnUfldRunner::infer(const cv::Mat& bgr_image, UfldOutput* output, double* infer_ms, std::string* error) {
    if (!impl_->instance || bgr_image.empty() || !output) {
        if (error) *error = "TNN runner is not initialized or input is empty";
        return false;
    }

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(impl_->config.input_width, impl_->config.input_height), 0, 0, cv::INTER_LINEAR);
    auto input_mat = std::make_shared<TNN_NS::Mat>(
        TNN_DEVICE_CPU,
        TNN_NS::N8UC3,
        toDims({1, 3, impl_->config.input_height, impl_->config.input_width}),
        resized.data);

    TNN_NS::MatConvertParam input_param;
    input_param.scale = {0.01712475f, 0.01750700f, 0.01742919f, 0.0f};
    input_param.bias = {-2.11790393f, -2.03571429f, -1.80444444f, 0.0f};
    input_param.reverse_channel = true;

    auto start = std::chrono::steady_clock::now();
    auto status = impl_->instance->SetInputMat(input_mat, input_param, impl_->input_name);
    if (status != TNN_NS::TNN_OK) {
        if (error) *error = "SetInputMat failed: " + status.description();
        return false;
    }

    status = impl_->instance->Forward();
    if (status != TNN_NS::TNN_OK) {
        if (error) *error = "Forward failed: " + status.description();
        return false;
    }
    auto end = std::chrono::steady_clock::now();
    if (infer_ms) {
        *infer_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    if (!impl_->copyOutput(impl_->output_name, &output->output, error)) return false;

    return true;
}

}  // namespace ufld_line_pkg
