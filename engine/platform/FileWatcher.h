#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace platform {

// Watches a directory tree for changed files (hot reload in dev builds; M9 plan D9). A background thread compares the
// sizes and modification times of files with the given extensions every interval and reports a path once its change has
// held still for one interval, so a file that is still being written (the cooker) isn't read half-done. The first scan
// only records what exists; files that appear later count as changed. Paths are relative to the root, with '/'.
class FileWatcher {
public:
    FileWatcher() = default;
    ~FileWatcher();
    FileWatcher(const FileWatcher&) = delete;
    FileWatcher& operator=(const FileWatcher&) = delete;

    bool start(const std::string& root, std::vector<std::string> extensions, uint32_t intervalMs = 250);
    void stop();

    // Cheap enough for every frame: one atomic load, no allocation.
    bool pending() const { return m_pending.load(std::memory_order_acquire); }
    // Moves the reported paths into `out` (cleared first).
    void drain(std::vector<std::string>& out);

private:
    struct Stamp {
        uint64_t size = 0;
        int64_t time = 0;
        bool changing = false; // changed in the last scan; reported when the next one finds it unchanged
    };
    void run();
    void scan(bool initial);

    std::string m_root;
    std::vector<std::string> m_extensions;
    uint32_t m_intervalMs = 250;
    std::unordered_map<std::string, Stamp> m_stamps; // the watcher thread's own
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    bool m_stop = false;
    std::vector<std::string> m_changed; // guarded by m_mutex
    std::atomic<bool> m_pending{false};
};

} // namespace platform
