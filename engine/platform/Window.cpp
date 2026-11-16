#include "platform/Window.h"

#include "core/AllocationCounter.h"
#include "core/Log.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace platform {
namespace {

int keyIndex(SDL_Scancode sc) {
    switch (sc) {
    case SDL_SCANCODE_W: return int(Key::W);
    case SDL_SCANCODE_A: return int(Key::A);
    case SDL_SCANCODE_S: return int(Key::S);
    case SDL_SCANCODE_D: return int(Key::D);
    case SDL_SCANCODE_Q: return int(Key::Q);
    case SDL_SCANCODE_E: return int(Key::E);
    case SDL_SCANCODE_LSHIFT:
    case SDL_SCANCODE_RSHIFT: return int(Key::Shift);
    case SDL_SCANCODE_LCTRL:
    case SDL_SCANCODE_RCTRL: return int(Key::Ctrl);
    case SDL_SCANCODE_SPACE: return int(Key::Space);
    case SDL_SCANCODE_F1: return int(Key::F1);
    default: return -1;
    }
}

bool setGlAttributes(const WindowDesc& desc) {
    bool ok = true;
    ok &= SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    ok &= SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    ok &= SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    ok &= SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, desc.debugContext ? SDL_GL_CONTEXT_DEBUG_FLAG : 0);
    // M9 plan D7: a context that is lost on a driver reset, so glGetGraphicsResetStatus can report it (the GL backend
    // checks what it got). Robust buffer access isn't needed for that, and isn't asked for.
    ok &= SDL_GL_SetAttribute(SDL_GL_CONTEXT_RESET_NOTIFICATION, SDL_GL_CONTEXT_RESET_LOSE_CONTEXT);
    ok &= SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    // The backbuffer only receives final composites, so it needs no depth or stencil.
    ok &= SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    ok &= SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    // A plain RGBA8 backbuffer: the GL backend encodes sRGB in shaders (see GlDevice::queryCaps).
    ok &= SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 0);
    return ok;
}

} // namespace

core::Result<std::unique_ptr<Window>> Window::create(const WindowDesc& desc) {
    // SDL allocations count toward the zero-allocation rule like ours do (ARCHITECTURE.md §4.12). This must
    // happen before SDL allocates anything.
    if constexpr (core::kAllocationCounterEnabled)
        SDL_SetMemoryFunctions(core::countedMalloc, core::countedCalloc, core::countedRealloc, core::countedFree);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        FL_LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
        return core::Error::InitFailed;
    }

    SDL_WindowFlags flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (desc.api == GraphicsApi::Vulkan) {
        flags |= SDL_WINDOW_VULKAN;
    } else {
        flags |= SDL_WINDOW_OPENGL;
        if (!setGlAttributes(desc)) {
            FL_LOG_ERROR("SDL_GL_SetAttribute failed: %s", SDL_GetError());
            return core::Error::InitFailed;
        }
    }

    std::unique_ptr<Window> window(new Window());
    window->m_api = desc.api;
    window->m_vsync = desc.vsync;
    window->m_window = SDL_CreateWindow(desc.title, desc.width, desc.height, flags);
    if (!window->m_window) {
        FL_LOG_ERROR("SDL_CreateWindow failed: %s", SDL_GetError());
        return core::Error::InitFailed;
    }

    if (desc.api != GraphicsApi::Vulkan) {
        SDL_GLContext context = SDL_GL_CreateContext(window->m_window);
        if (!context) { // a driver that refuses reset notification still gets a context, without it
            FL_LOG_WARN("No OpenGL context with reset notification (%s): trying without", SDL_GetError());
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_RESET_NOTIFICATION, SDL_GL_CONTEXT_RESET_NO_NOTIFICATION);
            context = SDL_GL_CreateContext(window->m_window);
        }
        if (!context) {
            FL_LOG_ERROR("Creating the OpenGL 3.3 core context failed: %s", SDL_GetError());
            return core::Error::InitFailed;
        }
        window->m_glContext = context;
        SDL_GL_MakeCurrent(window->m_window, context);
        if (!SDL_GL_SetSwapInterval(desc.vsync ? 1 : 0))
            FL_LOG_WARN("SDL_GL_SetSwapInterval failed: %s", SDL_GetError());
    }
    return window;
}

Window::~Window() {
    if (m_glContext) SDL_GL_DestroyContext(static_cast<SDL_GLContext>(m_glContext));
    if (m_window) SDL_DestroyWindow(m_window);
    SDL_Quit();
}

bool Window::pumpEvents(InputState& out) {
    out = {};
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (m_hook) m_hook(&event, m_hookUser);
        switch (event.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED: out.quitRequested = true; break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: out.resized = true; break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            const bool down = event.type == SDL_EVENT_KEY_DOWN;
            if (down && event.key.key == SDLK_ESCAPE) out.escapePressed = true;
            const int k = keyIndex(event.key.scancode);
            if (k >= 0) {
                if (down && !event.key.repeat) out.keyPressed[k] = true;
                m_keys[k] = down;
            }
            break;
        }
        case SDL_EVENT_MOUSE_MOTION:
            out.mouseDeltaX += event.motion.xrel;
            out.mouseDeltaY += event.motion.yrel;
            break;
        case SDL_EVENT_MOUSE_WHEEL: out.wheel += event.wheel.y; break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.button == SDL_BUTTON_RIGHT) {
                m_rightMouse = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
                SDL_SetWindowRelativeMouseMode(m_window, m_rightMouse); // mouse-look without hitting the screen edge
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            for (bool& k : m_keys) k = false; // key-up events are lost while unfocused
            m_rightMouse = false;
            SDL_SetWindowRelativeMouseMode(m_window, false);
            break;
        default: break;
        }
    }
    std::copy(std::begin(m_keys), std::end(m_keys), out.keyDown);
    out.rightMouseDown = m_rightMouse;
    return !out.quitRequested;
}

void Window::setEventHook(EventHook hook, void* user) {
    m_hook = hook;
    m_hookUser = user;
}

uint32_t Window::drawableWidth() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(m_window, &w, &h);
    return uint32_t(w > 0 ? w : 0);
}

uint32_t Window::drawableHeight() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(m_window, &w, &h);
    return uint32_t(h > 0 ? h : 0);
}

std::span<const char* const> Window::vulkanInstanceExtensions() const {
    Uint32 count = 0;
    const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!names) return {};
    return {names, count};
}

bool Window::createVulkanSurface(void* vkInstance, uint64_t* outSurface) const {
    VkSurfaceKHR surface{};
    if (!SDL_Vulkan_CreateSurface(m_window, static_cast<VkInstance>(vkInstance), nullptr, &surface)) {
        FL_LOG_ERROR("SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
        return false;
    }
    static_assert(sizeof(surface) == sizeof(uint64_t), "VkSurfaceKHR is a 64-bit handle");
    *outSurface = reinterpret_cast<uint64_t>(surface);
    return true;
}

void* Window::glGetProcAddress(const char* name) const {
    return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name));
}

void Window::glSwapBuffers() {
    SDL_GL_SwapWindow(m_window);
}

} // namespace platform
