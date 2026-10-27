#include "timing_average.hpp"
#include "test_support.hpp"

TEST_CASE("timing", holds_display_until_one_second_refresh) {
    water::TimingAverage average(10.0);
    average.record(10.25, 2.0f);
    average.record(10.75, 6.0f);
    average.update(10.999);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.update(11.0);
    CHECK_NEAR(average.milliseconds(), 4.0f, 1e-6f);
    average.record(11.25, 18.0f);
    average.update(11.999);
    CHECK_NEAR(average.milliseconds(), 4.0f, 1e-6f);
    average.update(12.0);
    CHECK_NEAR(average.milliseconds(), 18.0f, 1e-6f);
}

TEST_CASE("timing", excludes_samples_at_the_trailing_window_boundary) {
    water::TimingAverage average(0.0);
    average.record(0.0, 100.0f);
    average.record(0.25, 2.0f);
    average.record(1.0, 8.0f);
    average.update(1.0);
    CHECK_NEAR(average.milliseconds(), 5.0f, 1e-6f);
    average.record(1.75, 12.0f);
    average.record(2.0, 16.0f);
    average.update(2.0);
    CHECK_NEAR(average.milliseconds(), 14.0f, 1e-6f);
}

TEST_CASE("timing", averages_each_completed_measurement_once) {
    water::TimingAverage average(0.0);
    average.record(0.125, 3.0f);
    average.record(0.875, 9.0f);
    average.record(0.875, 12.0f);
    average.update(0.875);
    average.update(0.9375);
    average.update(1.0);
    CHECK_NEAR(average.milliseconds(), 8.0f, 1e-6f);
}

TEST_CASE("timing", refreshes_on_irregular_frames_without_updating_too_soon) {
    water::TimingAverage average(0.0);
    average.record(0.25, 100.0f);
    average.record(0.75, 4.0f);
    average.record(1.25, 10.0f);
    average.update(1.5);
    CHECK_NEAR(average.milliseconds(), 7.0f, 1e-6f);
    average.record(1.75, 20.0f);
    average.update(2.0);
    CHECK_NEAR(average.milliseconds(), 7.0f, 1e-6f);
    average.update(2.5);
    CHECK_NEAR(average.milliseconds(), 20.0f, 1e-6f);
}

TEST_CASE("timing", expires_paused_samples_and_recovers_after_long_gaps) {
    water::TimingAverage average(0.0);
    average.record(0.75, 6.0f);
    average.update(1.0);
    CHECK_NEAR(average.milliseconds(), 6.0f, 1e-6f);
    average.update(1.75);
    CHECK_NEAR(average.milliseconds(), 6.0f, 1e-6f);
    average.update(2.0);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.record(2.25, 50.0f);
    average.update(10.0);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.record(10.25, 8.0f);
    average.record(10.75, 12.0f);
    average.update(11.0);
    CHECK_NEAR(average.milliseconds(), 10.0f, 1e-6f);
}

TEST_CASE("timing", empty_startup_and_long_gap_stay_zero) {
    water::TimingAverage average(2.0);
    average.update(3.0);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.update(30.0);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.record(30.25, 16.0f);
    average.update(30.5);
    CHECK_NEAR(average.milliseconds(), 0.0f, 1e-6f);
    average.update(31.0);
    CHECK_NEAR(average.milliseconds(), 16.0f, 1e-6f);
}
