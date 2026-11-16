#include "platform/JobSystem.h"

#include <algorithm>

namespace platform {

JobSystem::JobSystem(uint32_t workerCount) {
    if (workerCount == 0) {
        const uint32_t hw = std::thread::hardware_concurrency();
        workerCount = hw > 1 ? hw - 1 : 1;
    }
    m_threads.reserve(workerCount);
    for (uint32_t i = 0; i < workerCount; ++i) m_threads.emplace_back([this, i] { workerLoop(i); });
}

JobSystem::~JobSystem() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    for (std::thread& t : m_threads) t.join();
}

void JobSystem::parallelFor(uint32_t count, uint32_t minBatch, RangeFn fn, void* context) {
    if (count == 0) return;
    const uint32_t participants = participantCount();
    // About four batches per participant balances uneven batches without much queue traffic.
    const uint32_t batch = std::max<uint32_t>(std::max<uint32_t>(minBatch, 1u), (count + participants * 4 - 1) / (participants * 4));
    if (batch >= count) {
        fn(context, 0, count, workerCount());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_fn = fn;
        m_context = context;
        m_count = count;
        m_batch = batch;
        m_remaining.store((count + batch - 1) / batch);
        m_next.store(0);
        ++m_generation;
    }
    m_wake.notify_all();
    while (runOneBatch(workerCount())) {}

    // Wait until every batch is done and no worker is still inside this job, so the next job's setup can
    // never race with a worker that is about to claim a batch.
    std::unique_lock<std::mutex> lock(m_mutex);
    m_done.wait(lock, [this] { return m_remaining.load() == 0 && m_active == 0; });
    m_fn = nullptr;
    m_context = nullptr;
}

bool JobSystem::runOneBatch(uint32_t workerIndex) {
    const uint32_t begin = m_next.fetch_add(m_batch);
    if (begin >= m_count) return false;
    const uint32_t end = std::min(begin + m_batch, m_count);
    m_fn(m_context, begin, end, workerIndex);
    if (m_remaining.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_done.notify_all();
    }
    return true;
}

void JobSystem::workerLoop(uint32_t index) {
    uint64_t seen = 0;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_wake.wait(lock, [&] { return m_stop || (m_generation != seen && m_fn != nullptr); });
            if (m_stop) return;
            seen = m_generation;
            ++m_active;
        }
        while (runOneBatch(index)) {}
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_active;
        }
        m_done.notify_all();
    }
}

} // namespace platform
