#pragma once

namespace platform {

class Window;

// Owns the Dear ImGui context and its SDL3 input backend. Rendering lives in
// renderer::ImGuiRenderer, so SDL stays inside platform/.
class ImGuiPlatform {
public:
    bool init(Window& window);
    void shutdown();
    void newFrame(); // ImGui_ImplSDL3_NewFrame + ImGui::NewFrame

private:
    static void onEvent(const void* sdlEvent, void* user);
    Window* m_window = nullptr;
};

} // namespace platform
