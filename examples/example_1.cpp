#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <print>
#include "ui/arch/arm/arm.hpp"
#include "ui/arch/arm/reciprocal.hpp"
#include "ui/arch/arm/sqrt.hpp"
#include "ui/base.hpp"
#include "ui/base_vec.hpp"
#include "ui/maths.hpp"
#include "ui/vec_op.hpp"
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

int main() {

    using type = int;
    std::array<type, 100> source;
    std::iota(source.begin(), source.end(), 1);

    static constexpr auto N = 4zu;

    auto a = ui::Vec<N, type>::load(source.data(), source.size());
    using ot = int;
    auto b = ui::Vec<N, ot>::load(source.data() + N, source.size() - N);

    std::println("A: {}\nB: {}", a.to_span(), b.to_span());

    /*auto r = ui::Vec<N / 2, int64_t>::load(2);*/
    auto t = ui::arm::fold(a, ui::op::max_t{});

    /*std::println("Vec: {} | {}", t.to_span(), to_name<decltype(t)::element_t>());*/
    std::println("Vec: {} | {}", t, to_name<decltype(t)>()); 
    return 0; 
}
