#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <print>
#include "ui/arch/arch.hpp"
#include "ui/arch/emul/manip.hpp"
#include "ui/arch/emul/minmax.hpp"
#include "ui/arch/emul/shift.hpp"
#include "ui/base.hpp"
#include "ui/float.hpp"
#include "ui/format.hpp"
#include "ui/base_vec.hpp"
#include "ui/maths.hpp"
#include "ui/vec_headers.hpp"
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

using namespace ui::arm::neon;
using namespace ui;

int main() {

    using type = std::int32_t;
    std::array<type, 200> source;
    static constexpr auto N = 4ul;
    std::iota(source.begin(), source.end(), 1);
    auto a = Vec<N, type>::load(source.data(), N);
    auto b = Vec<N, type>::load(2);
    std::println("A: {}\nB: {}\n", a, b);
    std::println("Emu: {}", emul::sub(a, b));
    std::println("ARM: {}", sub(a, b));
    return 0; 
}
