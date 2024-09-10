#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>


// @https://stackoverflow.com/questions/36762248/why-is-stdqueue-not-thread-safe
// @https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
template <typename T>
class SharedQueue
{
public:
    SharedQueue() {}
    ~SharedQueue() {}

    // Add an element to the queue.
    void enqueue(const T& t)
    {
        std::lock_guard<std::mutex> lock(m);
        q.push_back(t);
        c.notify_one();
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue() {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty())
        {
            // release lock as long as the wait and reaquire it afterwards.
            c.wait(lock);
        }
        T val = q.front();
        q.pop_front();
        return val;
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(m);
        q.clear();
    }

    int size()
    {
        std::unique_lock<std::mutex> mlock(m);
        return q.size();
    }
    bool empty()
    {
        std::unique_lock<std::mutex> mlock(m);
        return q.empty();
    }

private:
    std::deque<T> q;
    std::mutex m;
    std::condition_variable c;
};
