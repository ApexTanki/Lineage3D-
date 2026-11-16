#pragma once

#include "core/Result.h"
#include "rhi/Device.h"

#include <shaderformat/ShaderPackage.h>

#include <memory>
#include <string>
#include <vector>

namespace renderer {

// Loads .fsp packages and creates RHI shaders for the running backend's target (SPIR-V or GLSL 330).
// Package memory stays alive for the library's lifetime, so reflection views remain valid.
class ShaderLibrary {
public:
    explicit ShaderLibrary(rhi::Device& device);
    ~ShaderLibrary();
    ShaderLibrary(const ShaderLibrary&) = delete;
    ShaderLibrary& operator=(const ShaderLibrary&) = delete;

    core::Result<void> load(const std::string& name, const std::string& path);

    struct Program {
        rhi::ShaderHandle vertex;
        rhi::ShaderHandle fragment;
        const shaderformat::Reflection* reflection = nullptr; // valid for the library's lifetime
    };
    // Creates the variant's shaders on first use; later calls return the cached pair.
    core::Result<Program> program(const std::string& name, uint64_t key);

    // A permutation axis of a loaded package, by name, so key bits never have to be hard-coded (M2 plan D1).
    core::Result<shaderformat::Axis> axis(const std::string& name, const char* axisName) const;

    // Hot reload (dev; M9 plan D9): reads the package again and gives every shader made from it its variant's new code
    // (rhi::Device::replaceShaders makes their pipelines again in place, so no handle changes). A variant the new
    // package lacks, whose interface changed (shaderformat::sameInterface), or whose pipelines can't be made from the new
    // code keeps the old code (logged). The old bytes and reflection stay alive for the library's lifetime, because
    // callers may hold pointers into them: each reload of a package keeps one more copy of it (dev builds only).
    struct ReloadResult {
        uint32_t variants = 0; // in use (made by program())
        uint32_t replaced = 0;
    };
    core::Result<ReloadResult> reload(const std::string& name, const std::string& path);

private:
    struct Variant {
        uint64_t key;
        std::unique_ptr<shaderformat::VariantView> view;
        Program program;
    };
    struct Package {
        std::string name;
        std::vector<std::byte> data;
        shaderformat::PackageReader reader;
        std::vector<Variant> variants;
    };

    rhi::Device& m_device;
    std::vector<std::unique_ptr<Package>> m_packages;
    // Replaced by reloads, kept for the library's lifetime (reflection pointers handed out point into them).
    std::vector<std::vector<std::byte>> m_retiredData;
    std::vector<std::unique_ptr<shaderformat::VariantView>> m_retiredViews;
};

} // namespace renderer
