#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <print>
#include "ui/arch/arm/arm.hpp"
#include "ui/arch/arm/bit.hpp"
#include "ui/arch/arm/manip.hpp"
#include "ui/arch/arm/reciprocal.hpp"
#include "ui/arch/arm/shift.hpp"
#include "ui/arch/arm/sqrt.hpp"
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

    using type = int32_t;
    std::array<type, 100> source;
    std::iota(source.begin(), source.end(), 1);
    auto t = Vec<4, bfloat16>::load(bfloat16(12.4f));
    /*auto f = ui::bfloat16(t);*/
    /*std::println("{}", f + f);*/
    std::println("Vec: {} | {}", mul(t, t), to_name<decltype(t)>()); 
    return 0; 
}
