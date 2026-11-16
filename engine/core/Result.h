#pragma once

#include "core/Assert.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace core {

enum class Error : uint16_t {
    None,
    OutOfMemory,
    NotFound,
    IoError,
    InvalidData,
    Unsupported,
    CompileFailed,
    InitFailed,
    DeviceLost,
};

constexpr const char* errorName(Error error) {
    switch (error) {
    case Error::None: return "None";
    case Error::OutOfMemory: return "OutOfMemory";
    case Error::NotFound: return "NotFound";
    case Error::IoError: return "IoError";
    case Error::InvalidData: return "InvalidData";
    case Error::Unsupported: return "Unsupported";
    case Error::CompileFailed: return "CompileFailed";
    case Error::InitFailed: return "InitFailed";
    case Error::DeviceLost: return "DeviceLost";
    }
    return "Unknown";
}

// C++20 has no std::expected, so this is the minimal equivalent: a value or an error code.
template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : m_value(std::move(value)) {}
    Result(Error error) : m_error(error) { FL_ASSERT(error != Error::None, "an error Result needs an error code"); }

    bool ok() const { return m_error == Error::None; }
    explicit operator bool() const { return ok(); }
    Error error() const { return m_error; }

    T& value() {
        FL_ASSERT(ok(), "value() on an error Result");
        return *m_value;
    }
    const T& value() const {
        FL_ASSERT(ok(), "value() on an error Result");
        return *m_value;
    }

private:
    std::optional<T> m_value;
    Error m_error = Error::None;
};

template <>
class [[nodiscard]] Result<void> {
public:
    Result() = default;
    Result(Error error) : m_error(error) {}

    bool ok() const { return m_error == Error::None; }
    explicit operator bool() const { return ok(); }
    Error error() const { return m_error; }

private:
    Error m_error = Error::None;
};

} // namespace core
