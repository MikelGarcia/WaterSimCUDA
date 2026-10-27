#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* group;
    const char* name;
    void (*run)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> registered;
    return registered;
}

struct Register {
    Register(const char* group, const char* name, void (*run)()) {
        cases().push_back({group, name, run});
    }
};

inline void fail(const char* expression, const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression);
}

} // namespace test

#define TEST_CASE(group, name) \
    static void name(); \
    static test::Register register_##name(group, #name, name); \
    static void name()

#define CHECK(expression) \
    do { if (!(expression)) test::fail(#expression, __FILE__, __LINE__); } while (false)

#define CHECK_NEAR(actual, expected, tolerance) \
    CHECK(std::abs((actual) - (expected)) <= (tolerance))

#define CHECK_THROWS(expression, exception_type) \
    do { \
        bool caught = false; \
        try { (void)(expression); } catch (const exception_type&) { caught = true; } \
        CHECK(caught); \
    } while (false)
