

#ifndef _THREAD_POOL_HPP_
#define _THREAD_POOL_HPP_


#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <memory>
#include <glog/logging.h>




class ThreadPool {
public:
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    ThreadPool(uint32_t capacity=std::thread::hardware_concurrency(), 
            uint32_t n_threads=std::thread::hardware_concurrency()
            ): capacity(capacity), n_threads(n_threads) {
        init();
    }

    ~ThreadPool() noexcept {
        shutdown();
    }

    void init() {
        CHECK_GT(capacity, 0) << "task queue capacity should be greater than 0";
        CHECK_GT(n_threads, 0) << "thread pool capacity should be greater than 0";
        for (int i{0}; i < n_threads; ++i) {
            pool.emplace_back(std::thread([this] {
                std::function<void(void)> task;
                while (!this->stop) {
                    {
                        std::unique_lock<std::mutex> lock(this->q_mutex);
                        task_q_empty.wait(lock, [&] {return this->stop | !task_q.empty();});
                        if (this->stop) break;
                        task = this->task_q.front();
                        this->task_q.pop();
                        task_q_full.notify_one();
                    }
                    // auto id = std::this_thread::get_id();
                    task();
                }
            }));
        }
    }

    void shutdown() {
        stop = true;
        task_q_empty.notify_all();
        task_q_full.notify_all();
        for (auto& thread : pool) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    template<typename F, typename...Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using res_type = decltype(f(args...));
        std::function<res_type(void)> func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<res_type()>>(func);
        {
            std::unique_lock<std::mutex> lock(q_mutex);
            task_q_full.wait(lock, [&] {return this->stop | task_q.size() <= capacity;});
            CHECK (this->stop == false) << "should not add task to stopped queue\n";
            task_q.emplace([task_ptr]{(*task_ptr)();});
        }
        task_q_empty.notify_one();
        return task_ptr->get_future();
    }

private:
    std::vector<std::thread> pool;
    std::queue<std::function<void(void)>> task_q;
    std::condition_variable task_q_full;
    std::condition_variable task_q_empty;
    std::atomic<bool> stop{false};
    std::mutex q_mutex;
    uint32_t capacity;
    uint32_t n_threads;
};


#endif
