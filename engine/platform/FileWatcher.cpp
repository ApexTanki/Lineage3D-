#include "platform/FileWatcher.h"

#include "core/Log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <system_error>

namespace platform {
namespace fs = std::filesystem;

FileWatcher::~FileWatcher() { stop(); }

bool FileWatcher::start(const std::string& root, std::vector<std::string> extensions, uint32_t intervalMs) {
    stop();
    std::error_code ec;
    const fs::path rootPath(std::u8string(reinterpret_cast<const char8_t*>(root.data()), root.size()));
    if (!fs::is_directory(rootPath, ec)) {
        FL_LOG_WARN("File watcher: %s is not a directory", root.c_str());
        return false;
    }
    m_root = root;
    m_extensions = std::move(extensions);
    m_intervalMs = std::max(50u, intervalMs);
    m_stop = false;
    m_stamps.clear();
    scan(true); // what exists now isn't a change
    m_thread = std::thread([this] { run(); });
    return true;
}

void FileWatcher::stop() {
    if (!m_thread.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    m_thread.join();
}

void FileWatcher::drain(std::vector<std::string>& out) {
    out.clear();
    std::lock_guard<std::mutex> lock(m_mutex);
    out.swap(m_changed);
    m_pending.store(false, std::memory_order_release);
}

void FileWatcher::run() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (!m_stop) {
        m_wake.wait_for(lock, std::chrono::milliseconds(m_intervalMs), [this] { return m_stop; });
        if (m_stop) break;
        lock.unlock();
        scan(false);
        lock.lock();
    }
}

void FileWatcher::scan(bool initial) {
    std::error_code ec;
    const fs::path rootPath(std::u8string(reinterpret_cast<const char8_t*>(m_root.data()), m_root.size()));
    std::vector<std::string> ready;
    for (auto it = fs::recursive_directory_iterator(rootPath, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const std::u8string ext = it->path().extension().u8string();
        const std::string extension(ext.begin(), ext.end());
        if (std::find(m_extensions.begin(), m_extensions.end(), extension) == m_extensions.end()) continue;
        const uint64_t size = it->file_size(ec);
        const int64_t time = int64_t(it->last_write_time(ec).time_since_epoch().count());
        if (ec) continue; // vanished or locked mid-scan: the next scan sees it
        const std::u8string relative = fs::relative(it->path(), rootPath, ec).generic_u8string();
        const std::string key(relative.begin(), relative.end());
        auto found = m_stamps.find(key);
        if (found == m_stamps.end()) {
            m_stamps.emplace(key, Stamp{size, time, !initial});
            continue;
        }
        Stamp& s = found->second;
        if (s.size != size || s.time != time) { // changed: wait until it holds still for a scan
            s.size = size;
            s.time = time;
            s.changing = true;
        } else if (s.changing) {
            s.changing = false;
            ready.push_back(key);
        }
    }
    if (ready.empty()) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    for (std::string& path : ready)
        if (std::find(m_changed.begin(), m_changed.end(), path) == m_changed.end()) m_changed.push_back(std::move(path));
    m_pending.store(true, std::memory_order_release);
}

} // namespace platform
