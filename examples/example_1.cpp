#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <print>
#include "ui.hpp"
#include <cxxabi.h>

template <typename T>
std::string to_name() {
    int status = -4; // some arbitrary value to eliminate the compiler warning

    auto name = typeid(T).name();
    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}

using namespace ui;

int main() {

    using type = int;
    std::array<type, 200> source;
    static constexpr auto N = 4ul;
    std::iota(source.begin(), source.end(), 1);
    auto a = Vec<N, type>::load(source.data(), N);
    auto b = Vec<N, type>::load(2);
    std::println("A: {}\nB: {}\n", a, b);
    std::println("ARM: {}", (b >> 1));
    return 0; 
}
