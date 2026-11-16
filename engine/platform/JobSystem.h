#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace platform {

// Fixed worker pool for coarse fork-join work (ARCHITECTURE.md §8). parallelFor() splits [0, count) into
// batches, runs them on the workers and on the calling thread, and returns when every batch is done. Jobs are
// coarse (≥ ~50 µs per batch), so one mutex + condition variable is enough. A plain function pointer and
// context keep calls allocation-free. Called from one thread at a time (the main thread); not reentrant.
class JobSystem {
public:
    // workerIndex is 0..workerCount()-1 on workers and workerCount() on the calling thread, so per-participant
    // scratch arrays need participantCount() entries.
    using RangeFn = void (*)(void* context, uint32_t begin, uint32_t end, uint32_t workerIndex);

    explicit JobSystem(uint32_t workerCount = 0); // 0 = hardware threads − 1, at least 1
    ~JobSystem();
    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    // Batches hold at least minBatch items; small counts run inline on the caller.
    void parallelFor(uint32_t count, uint32_t minBatch, RangeFn fn, void* context);

    uint32_t workerCount() const { return uint32_t(m_threads.size()); }
    uint32_t participantCount() const { return workerCount() + 1; }

private:
    void workerLoop(uint32_t index);
    bool runOneBatch(uint32_t workerIndex);

    std::vector<std::thread> m_threads;
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::condition_variable m_done;
    bool m_stop = false;
    uint64_t m_generation = 0; // bumped per parallelFor; wakes the workers
    uint32_t m_active = 0;     // workers currently draining a job (guarded by m_mutex)

    // The current job. Written under m_mutex before m_generation changes and never while a worker is active.
    RangeFn m_fn = nullptr;
    void* m_context = nullptr;
    uint32_t m_count = 0;
    uint32_t m_batch = 1;
    std::atomic<uint32_t> m_next{0};      // first item of the next unclaimed batch
    std::atomic<uint32_t> m_remaining{0}; // batches not yet finished
};

} // namespace platform
