#ifndef AMT_UI_VEC_OP_HPP
#define AMT_UI_VEC_OP_HPP

#include "base_vec.hpp"
#include "features.hpp"
#include <type_traits>
#include "float.hpp"

#ifdef UI_ARM_HAS_NEON
    #include "arch/arm/arm.hpp"
#endif

namespace ui {
    template <std::size_t N, typename T>
        requires (std::is_arithmetic_v<T>)
    static inline constexpr auto load(T val) noexcept -> Vec<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::neon::load<N>(val);
        #else
            #error "not implemented"
        #endif
    }

    template <std::size_t N, typename T>
    inline constexpr auto Vec<N, T>::load(T val) noexcept -> Vec<N, T> {
        return ui::load<N, T>(val);
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
    inline constexpr auto load(Vec<M, T> const& v) noexcept -> Vec<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::neon::load<N, Lane>(v);
        #else
            #error "not implemented"
        #endif
    }

    template <std::size_t N, typename T>
    template <unsigned Lane, std::size_t M>
    inline constexpr auto Vec<N, T>::load(Vec<M, T> const& v) noexcept -> Vec<N, T> {
        return ui::load<N, Lane>(v);
    }
} // namespace ui

#endif // AMT_UI_VEC_OP_HPP
