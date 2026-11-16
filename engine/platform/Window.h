#pragma once

#include "core/Result.h"

#include <cstdint>
#include <memory>
#include <span>

struct SDL_Window;

namespace platform {

enum class GraphicsApi : uint8_t { Vulkan, OpenGL33 };

struct WindowDesc {
    const char* title = "FlashLineage3D";
    int width = 1280;
    int height = 720;
    bool vsync = true;
    bool debugContext = false; // GL: request a KHR_debug-capable debug context
    GraphicsApi api = GraphicsApi::Vulkan;
};

enum class Key : uint8_t { W, A, S, D, Q, E, Shift, Ctrl, Space, F1, Count };

struct InputState {
    bool quitRequested = false;
    bool resized = false;
    bool escapePressed = false;
    bool keyDown[uint32_t(Key::Count)] = {};    // held
    bool keyPressed[uint32_t(Key::Count)] = {}; // went down during this pumpEvents()
    bool rightMouseDown = false;                // held; the window is in relative mouse mode meanwhile
    float mouseDeltaX = 0.0f;                   // relative motion in pixels since the last pumpEvents()
    float mouseDeltaY = 0.0f;
    float wheel = 0.0f;
    bool down(Key k) const { return keyDown[uint32_t(k)]; }
    bool pressed(Key k) const { return keyPressed[uint32_t(k)]; }
};

// Receives every raw SDL event (as const void*), so tools such as the ImGui backend can consume input
// without SDL leaking outside platform/.
using EventHook = void (*)(const void* sdlEvent, void* user);

// Owns the SDL window and, for GL, the context. All SDL calls stay inside platform/.
class Window {
public:
    static core::Result<std::unique_ptr<Window>> create(const WindowDesc& desc);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Returns the input of this frame; false once the user asked to quit.
    bool pumpEvents(InputState& out);
    void setEventHook(EventHook hook, void* user);

    uint32_t drawableWidth() const;
    uint32_t drawableHeight() const;
    GraphicsApi api() const { return m_api; }
    bool vsync() const { return m_vsync; }
    SDL_Window* sdlWindow() const { return m_window; }

    // Vulkan hooks. Opaque types keep Vulkan headers out of platform/.
    std::span<const char* const> vulkanInstanceExtensions() const;
    bool createVulkanSurface(void* vkInstance, uint64_t* outSurface) const;

    // GL hooks.
    void* glGetProcAddress(const char* name) const;
    void glSwapBuffers();
    void* glContext() const { return m_glContext; }

private:
    Window() = default;

    SDL_Window* m_window = nullptr;
    void* m_glContext = nullptr;
    GraphicsApi m_api = GraphicsApi::Vulkan;
    bool m_vsync = true;
    bool m_keys[uint32_t(Key::Count)] = {};
    bool m_rightMouse = false;
    EventHook m_hook = nullptr;
    void* m_hookUser = nullptr;
};

} // namespace platform
