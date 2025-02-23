#include <cxxabi.h>
#include <array>
#include <cfloat>
#include <numeric>
#include <print>
#include <memory>

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

#include "ui.hpp"

using namespace ui;

template <std::size_t N, typename T>
void random(std::array<T, N>& arr) {
    for (auto i = 0ul; i < N; ++i) {
        arr[i] = static_cast<T>(rand() % 1000);
    }
}

int main() {

    using type = int64_t;
    std::array<type, 200> source;
    static constexpr auto N = 4u;
    std::iota(source.begin(), source.end(), 1);
    /*random(source);*/
    auto a = Vec<N, type>::load(source.data(), N);
    auto b = Vec<N, type>::load(source.data() + N, N);
    /*b[0] = a[0] = NAN;*/
    /*b[1] = NAN;*/
    /*a[2] = NAN;*/
    a[0] = -1;
    b[0] = 2;
    a[1] = INT64_MIN;
    a[2] = b[2]; 
    std::println("A: {}\nB: {}\n=> {}", a, b, abs_acc_diff(Vec<N, type>::load(1), a, b));

    /*std::println("{} | {}", t, to_name<decltype(t)>());*/
    return 0; 
}
