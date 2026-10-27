#include "test_support.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: water_tests [group]\n";
        return 1;
    }
    int selected = 0;
    int failed = 0;
    for (const auto& test : test::cases()) {
        if (argc == 2 && std::string_view(argv[1]) != test.group) continue;
        ++selected;
        try {
            test.run();
            std::cout << "PASS " << test.group << "/" << test.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << test.group << "/" << test.name << ": "
                      << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "FAIL " << test.group << "/" << test.name
                      << ": unknown exception\n";
        }
    }
    if (selected == 0) {
        std::cerr << "No tests matched the requested group\n";
        return 1;
    }
    std::cout << selected - failed << '/' << selected << " tests passed\n";
    return failed == 0 ? 0 : 1;
}
