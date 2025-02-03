#ifndef AMT_UI_ARCH_ARM_DIV_HPP
#define AMT_UI_ARCH_ARM_DIV_HPP

#include "cast.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "basic.hpp"

namespace ui::arm {

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
                        vdiv_f64(to_vec(num), to_vec(den))
                    );
                }
            #endif
            return {
                .val = num.val / den.val
            };
        } else {
        
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vdiv_f32(to_vec(num), to_vec(den))
                    );
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vdivq_f32(to_vec(num), to_vec(den))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vdivq_f64(to_vec(num), to_vec(den))
                    );
                }
            }

            #endif

            return join(
                div(num.lo, den.lo),
                div(num.hi, den.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = num.val / num.den
            };
        } else {
            if constexpr (sizeof(T) == 8) {
                auto tn = cast<double>(num);
                auto td = cast<double>(den);
                return cast<T>(div(tn, td));
            } else {
                auto tn = cast<float>(num);
                auto td = cast<float>(den);
                return cast<T>(div(tn, td));
            }
        }
    }
} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_DIV_HPP
