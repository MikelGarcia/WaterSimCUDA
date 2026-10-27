#pragma once

#include <deque>

namespace water {

class TimingAverage {
public:
    explicit TimingAverage(double started) : refreshed_(started) {}

    void record(double now, float milliseconds) {
        samples_.push_back({now, milliseconds});
    }

    void update(double now) {
        // Completed measurements belong to the trailing interval (now - 1, now].
        while (!samples_.empty() && samples_.front().time <= now - 1.0)
            samples_.pop_front();
        if (now - refreshed_ < 1.0) return;
        double total = 0;
        for (const auto& sample : samples_) total += sample.milliseconds;
        milliseconds_ = samples_.empty() ? 0.0f : static_cast<float>(total / samples_.size());
        refreshed_ = now;
    }

    float milliseconds() const { return milliseconds_; }

private:
    struct Sample { double time; float milliseconds; };
    std::deque<Sample> samples_;
    double refreshed_;
    float milliseconds_ = 0;
};

} // namespace water
