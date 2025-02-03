#ifndef AMT_UI_VEC_OP_HPP
#define AMT_UI_VEC_OP_HPP

#include "base_vec.hpp"
#include "features.hpp"
#include "ui/arch/arm/arm.hpp"
#include <initializer_list>
#include <type_traits>

namespace ui {
    template <std::size_t N, typename T>
    inline constexpr auto Vec<N, T>::load(T val) noexcept -> Vec<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::load<N>(val);
        #else
            #error "not implemented"
        #endif
    }
} // namespace ui

#endif // AMT_UI_VEC_OP_HPP
