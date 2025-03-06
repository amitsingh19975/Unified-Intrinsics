#ifndef AMT_UI_ARCH_EMUL_CAST_HPP
#define AMT_UI_ARCH_EMUL_CAST_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../basic.hpp"
#include "../../vec_headers.hpp"
#include "../../maths.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include <concepts>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ui::emul {
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        if constexpr (std::same_as<To, From>) return v;
        return map([](auto v_) { 
            if constexpr (std::floating_point<From>) {
                static constexpr auto min = std::numeric_limits<To>::min();
                static constexpr auto max = std::numeric_limits<To>::max();
                using type = std::conditional_t<
                    std::same_as<From, float16> || std::same_as<From, bfloat16>,
                    float,
                    From
                >;
                if constexpr (std::integral<To>) {
                    if (v_ == std::numeric_limits<From>::infinity()) {
                        return max;
                    } else if (v_ == -std::numeric_limits<From>::infinity()) {
                        return min;
                    }
                    return static_cast<To>(type(v_));
                } else {
                    return static_cast<To>(type(v_));
                }
            }
            return static_cast<To>(v_);
        }, v);
    }

    template <typename To, std::size_t N, std::integral From>
    UI_ALWAYS_INLINE auto sat_cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        if constexpr (std::same_as<To, From>) return v;
        return map([](auto v_) {
            return ::ui::internal::saturating_cast_helper<To, true>(v_);
        }, v);
    }

    // retinterpret cast
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE constexpr auto rcast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return std::bit_cast<Vec<N, To>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto to_vec(auto const& v) noexcept {
        return v;
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(auto const& v) noexcept {
        return v;
    }

    template <typename T>
    UI_ALWAYS_INLINE constexpr auto from_vec(auto const& v) noexcept {
        return rcast<T>(v);
    }
} // namespace ui::emul

#endif // AMT_UI_ARCH_EMUL_CAST_HPP
