#include <windows.h>

#if __has_include(<onnxruntime_c_api.h>)
#include <onnxruntime_c_api.h>
#elif __has_include("../third_party/onnxruntime/include/onnxruntime_c_api.h")
#include "../third_party/onnxruntime/include/onnxruntime_c_api.h"
#else
#error "onnxruntime_c_api.h not found. Run bootstrap.ps1 first."
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr const char* kInputName =
    "/embedder/base/htsat/patch_embed/norm/LayerNormalization_output_0";
constexpr const char* kOutputName =
    "/embedder/base/htsat/layers.0/blocks.0/Reshape_3_output_0";

struct Bytes {
    std::vector<uint8_t> data;

    void byte(uint8_t v) { data.push_back(v); }

    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        data.insert(data.end(), b, b + n);
    }

    void varint(uint64_t v) {
        while (v >= 0x80) {
            byte(static_cast<uint8_t>(v | 0x80));
            v >>= 7;
        }
        byte(static_cast<uint8_t>(v));
    }

    void key(int field, int wire) { varint(static_cast<uint64_t>((field << 3) | wire)); }

    void int64_field(int field, int64_t v) {
        key(field, 0);
        varint(static_cast<uint64_t>(v));
    }

    void int32_field(int field, int32_t v) {
        key(field, 0);
        varint(static_cast<uint32_t>(v));
    }

    void float_field(int field, float v) {
        key(field, 5);
        raw(&v, sizeof(v));
    }

    void string_field(int field, const std::string& s) {
        key(field, 2);
        varint(s.size());
        raw(s.data(), s.size());
    }

    void bytes_field(int field, const std::vector<uint8_t>& b) {
        key(field, 2);
        varint(b.size());
        raw(b.data(), b.size());
    }

    void message_field(int field, const Bytes& m) {
        key(field, 2);
        varint(m.data.size());
        raw(m.data.data(), m.data.size());
    }
};

std::vector<uint8_t> raw_i64(std::initializer_list<int64_t> values) {
    std::vector<uint8_t> out;
    out.reserve(values.size() * sizeof(int64_t));
    for (int64_t v : values) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        out.insert(out.end(), p, p + sizeof(v));
    }
    return out;
}

std::vector<uint8_t> raw_f32(const std::vector<float>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(float));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

Bytes tensor(const std::string& name, int data_type, const std::vector<int64_t>& dims,
             const std::vector<uint8_t>& raw) {
    Bytes t;
    for (int64_t d : dims) t.int64_field(1, d);
    t.int32_field(2, data_type);
    t.string_field(8, name);
    t.bytes_field(9, raw);
    return t;
}

Bytes attr_tensor(const std::string& name, const Bytes& value) {
    Bytes a;
    a.string_field(1, name);
    a.message_field(5, value);
    a.int32_field(20, 4);  // AttributeProto::TENSOR
    return a;
}

Bytes attr_int(const std::string& name, int64_t value) {
    Bytes a;
    a.string_field(1, name);
    a.int64_field(3, value);
    a.int32_field(20, 2);  // AttributeProto::INT
    return a;
}

Bytes attr_float(const std::string& name, float value) {
    Bytes a;
    a.string_field(1, name);
    a.float_field(2, value);
    a.int32_field(20, 1);  // AttributeProto::FLOAT
    return a;
}

Bytes attr_ints(const std::string& name, std::initializer_list<int64_t> values) {
    Bytes a;
    a.string_field(1, name);
    for (int64_t v : values) a.int64_field(8, v);
    a.int32_field(20, 7);  // AttributeProto::INTS
    return a;
}

Bytes node(const std::string& op, const std::string& name, std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs, std::initializer_list<Bytes> attrs = {}) {
    Bytes n;
    for (const auto& i : inputs) n.string_field(1, i);
    for (const auto& o : outputs) n.string_field(2, o);
    n.string_field(3, name);
    n.string_field(4, op);
    for (const auto& a : attrs) n.message_field(5, a);
    return n;
}

Bytes constant_i64_node(const std::string& name, const std::string& output,
                        std::initializer_list<int64_t> values) {
    const std::vector<int64_t> dims = values.size() == 1 ? std::vector<int64_t>{1} : std::vector<int64_t>{};
    return node("Constant", name, {}, {output},
                {attr_tensor("value", tensor("", 7, dims, raw_i64(values)))});
}

Bytes scalar_i64_node(const std::string& name, const std::string& output, int64_t value) {
    return node("Constant", name, {}, {output},
                {attr_tensor("value", tensor("", 7, {}, raw_i64({value})))});
}

Bytes dim_value(int64_t v) {
    Bytes d;
    d.int64_field(1, v);
    return d;
}

Bytes dim_param(const std::string& s) {
    Bytes d;
    d.string_field(2, s);
    return d;
}

Bytes value_info(const std::string& name, std::initializer_list<Bytes> dims) {
    Bytes shape;
    for (const auto& d : dims) shape.message_field(1, d);
    Bytes tensor_type;
    tensor_type.int32_field(1, 1);  // FLOAT
    tensor_type.message_field(2, shape);
    Bytes type;
    type.message_field(1, tensor_type);
    Bytes vi;
    vi.string_field(1, name);
    vi.message_field(2, type);
    return vi;
}

std::string gather_dim(std::vector<Bytes>& nodes, const std::string& source, int dim,
                       const std::string& suffix) {
    const std::string shape = "/embedder/base/htsat/layers.0/blocks.0/Shape" + suffix + "_output_0";
    const std::string c = "/embedder/base/htsat/layers.0/blocks.0/Constant" + suffix + "_output_0";
    const std::string g = "/embedder/base/htsat/layers.0/blocks.0/Gather" + suffix + "_output_0";
    nodes.push_back(node("Shape", "/embedder/base/htsat/layers.0/blocks.0/Shape" + suffix,
                         {source}, {shape}));
    nodes.push_back(scalar_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant" + suffix, c, dim));
    nodes.push_back(node("Gather", "/embedder/base/htsat/layers.0/blocks.0/Gather" + suffix,
                         {shape, c}, {g}, {attr_int("axis", 0)}));
    return g;
}

std::string unsqueeze(std::vector<Bytes>& nodes, const std::string& input, const std::string& output,
                      const std::string& name, const std::string& axes_name) {
    const std::string axes = "onnx::Unsqueeze_" + axes_name;
    nodes.push_back(constant_i64_node("Constant_" + axes_name, axes, {0}));
    nodes.push_back(node("Unsqueeze", name, {input, axes}, {output}));
    return output;
}

std::vector<uint8_t> build_model_proto() {
    std::vector<Bytes> nodes;
    const std::string batch = gather_dim(nodes, kInputName, 0, "");
    const std::string channels = gather_dim(nodes, kInputName, 2, "_1");

    std::vector<float> ones(96, 1.0f), zeros(96, 0.0f);
    const Bytes weight = tensor("embedder.base.htsat.layers.0.blocks.0.norm1.weight", 1, {96}, raw_f32(ones));
    const Bytes bias = tensor("embedder.base.htsat.layers.0.blocks.0.norm1.bias", 1, {96}, raw_f32(zeros));

    const std::string norm = "/embedder/base/htsat/layers.0/blocks.0/norm1/LayerNormalization_output_0";
    nodes.push_back(node("LayerNormalization",
                         "/embedder/base/htsat/layers.0/blocks.0/norm1/LayerNormalization",
                         {kInputName, "embedder.base.htsat.layers.0.blocks.0.norm1.weight",
                          "embedder.base.htsat.layers.0.blocks.0.norm1.bias"},
                         {norm}, {attr_int("axis", -1), attr_float("epsilon", 1e-5f)}));

    const auto batch_u = unsqueeze(nodes, batch, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_output_0",
                                   "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze", "369");
    const std::string c64a = "/embedder/base/htsat/layers.0/blocks.0/Constant_2_output_0";
    const std::string c64b = "/embedder/base/htsat/layers.0/blocks.0/Constant_3_output_0";
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_2", c64a, {64}));
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_3", c64b, {64}));
    const auto channels_u =
        unsqueeze(nodes, channels, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_1_output_0",
                  "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_1", "375");
    const std::string shape1 = "/embedder/base/htsat/layers.0/blocks.0/Concat_output_0";
    nodes.push_back(node("Concat", "/embedder/base/htsat/layers.0/blocks.0/Concat",
                         {batch_u, c64a, c64b, channels_u}, {shape1}, {attr_int("axis", 0)}));
    const std::string r0 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_output_0";
    nodes.push_back(node("Reshape", "/embedder/base/htsat/layers.0/blocks.0/Reshape",
                         {norm, shape1}, {r0}, {attr_int("allowzero", 0)}));

    std::vector<std::string> r0_dims;
    for (int i = 0; i < 4; ++i) r0_dims.push_back(gather_dim(nodes, r0, i, "_r0_" + std::to_string(i)));

    const std::string c8a = "/embedder/base/htsat/layers.0/blocks.0/Constant_8_output_0";
    const std::string c8b = "/embedder/base/htsat/layers.0/blocks.0/Constant_9_output_0";
    nodes.push_back(scalar_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_8", c8a, 8));
    nodes.push_back(scalar_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_9", c8b, 8));
    const std::string hdiv = "/embedder/base/htsat/layers.0/blocks.0/Div_output_0";
    const std::string wdiv = "/embedder/base/htsat/layers.0/blocks.0/Div_1_output_0";
    nodes.push_back(node("Div", "/embedder/base/htsat/layers.0/blocks.0/Div", {r0_dims[1], c8a}, {hdiv}));
    nodes.push_back(node("Div", "/embedder/base/htsat/layers.0/blocks.0/Div_1", {r0_dims[2], c8b}, {wdiv}));
    const std::string hcast0 = "/embedder/base/htsat/layers.0/blocks.0/Cast_output_0";
    const std::string hcast1 = "/embedder/base/htsat/layers.0/blocks.0/Cast_1_output_0";
    const std::string wcast0 = "/embedder/base/htsat/layers.0/blocks.0/Cast_2_output_0";
    const std::string wcast1 = "/embedder/base/htsat/layers.0/blocks.0/Cast_3_output_0";
    nodes.push_back(node("Cast", "/embedder/base/htsat/layers.0/blocks.0/Cast", {hdiv}, {hcast0}, {attr_int("to", 7)}));
    nodes.push_back(node("Cast", "/embedder/base/htsat/layers.0/blocks.0/Cast_1", {hcast0}, {hcast1}, {attr_int("to", 7)}));
    nodes.push_back(node("Cast", "/embedder/base/htsat/layers.0/blocks.0/Cast_2", {wdiv}, {wcast0}, {attr_int("to", 7)}));
    nodes.push_back(node("Cast", "/embedder/base/htsat/layers.0/blocks.0/Cast_3", {wcast0}, {wcast1}, {attr_int("to", 7)}));

    const std::string cc8a = "/embedder/base/htsat/layers.0/blocks.0/Constant_10_output_0";
    const std::string cc8b = "/embedder/base/htsat/layers.0/blocks.0/Constant_11_output_0";
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_10", cc8a, {8}));
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_11", cc8b, {8}));
    const std::string shape2 = "/embedder/base/htsat/layers.0/blocks.0/Concat_1_output_0";
    nodes.push_back(node("Concat", "/embedder/base/htsat/layers.0/blocks.0/Concat_1",
                         {unsqueeze(nodes, r0_dims[0], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_2_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_2", "400"),
                          unsqueeze(nodes, hcast1, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_3_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_3", "402"),
                          cc8a,
                          unsqueeze(nodes, wcast1, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_4_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_4", "406"),
                          cc8b,
                          unsqueeze(nodes, r0_dims[3], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_5_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_5", "410")},
                         {shape2}, {attr_int("axis", 0)}));
    const std::string r1 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_1_output_0";
    nodes.push_back(node("Reshape", "/embedder/base/htsat/layers.0/blocks.0/Reshape_1",
                         {r0, shape2}, {r1}, {attr_int("allowzero", 0)}));
    const std::string t0 = "/embedder/base/htsat/layers.0/blocks.0/Transpose_output_0";
    nodes.push_back(node("Transpose", "/embedder/base/htsat/layers.0/blocks.0/Transpose",
                         {r1}, {t0}, {attr_ints("perm", {0, 1, 3, 2, 4, 5})}));

    const std::string cm1a = "/embedder/base/htsat/layers.0/blocks.0/Constant_12_output_0";
    const std::string c8c = "/embedder/base/htsat/layers.0/blocks.0/Constant_13_output_0";
    const std::string c8d = "/embedder/base/htsat/layers.0/blocks.0/Constant_14_output_0";
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_12", cm1a, {-1}));
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_13", c8c, {8}));
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_14", c8d, {8}));
    const std::string shape3 = "/embedder/base/htsat/layers.0/blocks.0/Concat_2_output_0";
    nodes.push_back(node("Concat", "/embedder/base/htsat/layers.0/blocks.0/Concat_2",
                         {cm1a, c8c, c8d,
                          unsqueeze(nodes, r0_dims[3], "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_6_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_6", "421")},
                         {shape3}, {attr_int("axis", 0)}));
    const std::string r2 = "/embedder/base/htsat/layers.0/blocks.0/Reshape_2_output_0";
    nodes.push_back(node("Reshape", "/embedder/base/htsat/layers.0/blocks.0/Reshape_2",
                         {t0, shape3}, {r2}, {attr_int("allowzero", 0)}));

    const std::string cm1b = "/embedder/base/htsat/layers.0/blocks.0/Constant_15_output_0";
    const std::string c64 = "/embedder/base/htsat/layers.0/blocks.0/Constant_16_output_0";
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_15", cm1b, {-1}));
    nodes.push_back(constant_i64_node("/embedder/base/htsat/layers.0/blocks.0/Constant_16", c64, {64}));
    const std::string shape4 = "/embedder/base/htsat/layers.0/blocks.0/Concat_3_output_0";
    nodes.push_back(node("Concat", "/embedder/base/htsat/layers.0/blocks.0/Concat_3",
                         {cm1b, c64,
                          unsqueeze(nodes, channels, "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_7_output_0",
                                    "/embedder/base/htsat/layers.0/blocks.0/Unsqueeze_7", "429")},
                         {shape4}, {attr_int("axis", 0)}));
    nodes.push_back(node("Reshape", "/embedder/base/htsat/layers.0/blocks.0/Reshape_3",
                         {r2, shape4}, {kOutputName}, {attr_int("allowzero", 0)}));

    Bytes graph;
    for (const auto& n : nodes) graph.message_field(1, n);
    graph.string_field(2, "amd-vaiml-window-partition-repro");
    graph.message_field(5, weight);
    graph.message_field(5, bias);
    graph.message_field(11, value_info(kInputName, {dim_value(1), dim_value(4096), dim_value(96)}));
    graph.message_field(12, value_info(kOutputName, {dim_param("unk__6"), dim_value(64), dim_value(96)}));

    Bytes opset;
    opset.int64_field(2, 17);

    Bytes model;
    model.int64_field(1, 8);
    model.message_field(7, graph);
    model.message_field(8, opset);
    return model.data;
}

std::string sha256_file(const fs::path&) {
    // Avoid pulling crypto dependencies into the native repro. The model is source-generated and
    // ORT validates it while creating the session; the README records the known Python-generator hash.
    return "(not computed by C++ runner)";
}

void write_model(const fs::path& path, bool force) {
    if (!force && fs::exists(path)) return;
    const auto bytes = build_model_proto();
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to write " + path.string());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    std::cout << "Generated " << path << " (" << bytes.size() << " bytes)\n";
}

struct OrtDyn {
    HMODULE lib = nullptr;
    const OrtApi* api = nullptr;

    explicit OrtDyn(const fs::path& dll) {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        AddDllDirectory(dll.parent_path().c_str());
        lib = LoadLibraryW(dll.c_str());
        if (!lib) throw std::runtime_error("LoadLibrary failed for " + dll.string());
        using GetBase = const OrtApiBase*(ORT_API_CALL*)();
        auto get_base = reinterpret_cast<GetBase>(GetProcAddress(lib, "OrtGetApiBase"));
        if (!get_base) throw std::runtime_error("OrtGetApiBase not exported by " + dll.string());
        api = get_base()->GetApi(ORT_API_VERSION);
        if (!api) throw std::runtime_error("OrtGetApi returned null");
        std::cout << "ORT version: " << get_base()->GetVersionString() << "\n";
        std::cout << "ORT DLL    : " << dll << "\n";
    }
};

void check(const OrtApi* api, OrtStatus* st, const char* where) {
    if (!st) return;
    std::string msg = where;
    msg += ": ";
    msg += api->GetErrorMessage(st);
    api->ReleaseStatus(st);
    throw std::runtime_error(msg);
}

template <typename T>
struct OrtPtr {
    const OrtApi* api = nullptr;
    T* p = nullptr;
};

void run_once(const fs::path& ort_dll, const fs::path& model, bool vitis, const fs::path& cache_dir) {
    OrtDyn ort(ort_dll);
    const OrtApi* api = ort.api;
    OrtEnv* env = nullptr;
    OrtSessionOptions* so = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* mem = nullptr;
    OrtValue* input = nullptr;
    OrtValue* output = nullptr;

    try {
        check(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "amd-vaiml-repro", &env), "CreateEnv");
        check(api, api->CreateSessionOptions(&so), "CreateSessionOptions");
        check(api, api->SetSessionGraphOptimizationLevel(so, ORT_DISABLE_ALL),
              "SetSessionGraphOptimizationLevel");
        if (vitis) {
            fs::create_directories(cache_dir);
            const std::string cache = cache_dir.string();
            const std::string key = "cpp-window-partition";
            const char* keys[] = {"cache_dir", "cache_key"};
            const char* vals[] = {cache.c_str(), key.c_str()};
            check(api, api->SessionOptionsAppendExecutionProvider_VitisAI(so, keys, vals, 2),
                  "SessionOptionsAppendExecutionProvider_VitisAI");
        }

        std::cout << "Creating " << (vitis ? "VitisAI" : "CPU") << " session...\n";
        check(api, api->CreateSession(env, model.c_str(), so, &session), "CreateSession");

        std::vector<float> data(1 * 4096 * 96);
        std::mt19937 rng(20260711);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& v : data) v = normal(rng);
        int64_t dims[] = {1, 4096, 96};
        check(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem), "CreateCpuMemoryInfo");
        check(api, api->CreateTensorWithDataAsOrtValue(mem, data.data(), data.size() * sizeof(float),
                                                       dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input),
              "CreateTensorWithDataAsOrtValue");
        int is_tensor = 0;
        check(api, api->IsTensor(input, &is_tensor), "IsTensor");
        if (!is_tensor) throw std::runtime_error("input is not a tensor");

        const char* input_names[] = {kInputName};
        const char* output_names[] = {kOutputName};
        std::cout << "Running first inference...\n";
        check(api, api->Run(session, nullptr, input_names, const_cast<const OrtValue* const*>(&input), 1,
                            output_names, 1, &output),
              "Run");

        OrtTensorTypeAndShapeInfo* info = nullptr;
        check(api, api->GetTensorTypeAndShape(output, &info), "GetTensorTypeAndShape");
        size_t rank = 0;
        check(api, api->GetDimensionsCount(info, &rank), "GetDimensionsCount");
        std::vector<int64_t> out_dims(rank);
        check(api, api->GetDimensions(info, out_dims.data(), rank), "GetDimensions");
        api->ReleaseTensorTypeAndShapeInfo(info);
        float* out = nullptr;
        check(api, api->GetTensorMutableData(output, reinterpret_cast<void**>(&out)), "GetTensorMutableData");
        std::cout << "Output shape:";
        for (int64_t d : out_dims) std::cout << " " << d;
        std::cout << "\nFirst value: " << out[0] << "\n";
    } catch (...) {
        if (output) api->ReleaseValue(output);
        if (input) api->ReleaseValue(input);
        if (mem) api->ReleaseMemoryInfo(mem);
        if (session) api->ReleaseSession(session);
        if (so) api->ReleaseSessionOptions(so);
        if (env) api->ReleaseEnv(env);
        throw;
    }
    if (output) api->ReleaseValue(output);
    if (input) api->ReleaseValue(input);
    if (mem) api->ReleaseMemoryInfo(mem);
    if (session) api->ReleaseSession(session);
    if (so) api->ReleaseSessionOptions(so);
    if (env) api->ReleaseEnv(env);
}

fs::path default_ort_dll() {
    const wchar_t* env = _wgetenv(L"ORT_DLL");
    if (env && *env) return fs::path(env);
    fs::path p = LR"(C:\ProgramData\miniforge3\envs\ryzen-ai-1.8.0-beta\Lib\site-packages\onnxruntime\capi\onnxruntime.dll)";
    return p;
}

int wmain(int argc, wchar_t** argv) {
    fs::path model = "model.onnx";
    fs::path ort_dll = default_ort_dll();
    fs::path cache = fs::temp_directory_path() / "amd-vaiml-window-partition-cpp-cache";
    bool regen = false;
    bool cpu_only = false;
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--model" && i + 1 < argc) model = argv[++i];
        else if (a == L"--ort-dll" && i + 1 < argc) ort_dll = argv[++i];
        else if (a == L"--cache-dir" && i + 1 < argc) cache = argv[++i];
        else if (a == L"--regenerate-model") regen = true;
        else if (a == L"--cpu-only") cpu_only = true;
        else {
            std::wcerr << L"usage: amd_vaiml_repro.exe [--ort-dll path] [--model path] "
                          L"[--cache-dir path] [--regenerate-model] [--cpu-only]\n";
            return 2;
        }
    }

    try {
        write_model(model, regen);
        std::cout << "Model      : " << model << "\n";
        std::cout << "Model hash : " << sha256_file(model) << "\n";
        run_once(ort_dll, model, false, cache);
        if (!cpu_only) run_once(ort_dll, model, true, cache);
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
