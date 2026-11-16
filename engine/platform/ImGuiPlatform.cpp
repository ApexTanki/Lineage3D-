#include "platform/ImGuiPlatform.h"

#include "core/AllocationCounter.h"
#include "platform/Window.h"

#include <SDL3/SDL_events.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>

namespace platform {
namespace {

void* imguiAlloc(size_t size, void*) {
    return core::countedDebugUiMalloc(size);
}
void imguiFree(void* memory, void*) {
    core::countedFree(memory);
}

} // namespace

bool ImGuiPlatform::init(Window& window) {
    IMGUI_CHECKVERSION();
    // ImGui's allocations are counted too, but reported as debug-UI allocations (ARCHITECTURE.md §4.12).
    if constexpr (core::kAllocationCounterEnabled) ImGui::SetAllocatorFunctions(imguiAlloc, imguiFree);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // don't write imgui.ini into the working directory
    ImGui::StyleColorsDark();
    if (!ImGui_ImplSDL3_InitForOther(window.sdlWindow())) {
        ImGui::DestroyContext();
        return false;
    }
    m_window = &window;
    window.setEventHook(&ImGuiPlatform::onEvent, this);
    return true;
}

void ImGuiPlatform::shutdown() {
    if (!m_window) return;
    m_window->setEventHook(nullptr, nullptr);
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    m_window = nullptr;
}

void ImGuiPlatform::newFrame() {
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void ImGuiPlatform::onEvent(const void* sdlEvent, void*) {
    ImGui_ImplSDL3_ProcessEvent(static_cast<const SDL_Event*>(sdlEvent));
}

} // namespace platform
