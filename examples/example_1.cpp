#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
#include <numeric>
#include <print>
#include "ui/arch/arm/arm.hpp"
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

    static constexpr auto N = 1zu;

    auto a = ui::load<N>(source.data(), source.size());
    using ot = int;
    auto b = ui::load<N, ot>(source.data() + N, source.size() - N);

    /*a[0] = INFINITY;*/
    /*b[0] = -INFINITY;*/

    std::println("A: {}\nB: {}", a.to_span(), b.to_span());

    auto r = ui::load<N, double>(10);
    auto e = ui::arm::reciprocal_estimate(r);
    std::println("Estimate: {}", e.to_span());

    auto t = ui::arm::reciprocal_refine(r, e);

    std::println("Vec: {} | {}", t.to_span(), to_name<decltype(t)::element_t>());
    return 0; 
}
