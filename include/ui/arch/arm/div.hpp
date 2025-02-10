#ifndef AMT_UI_ARCH_ARM_DIV_HPP
#define AMT_UI_ARCH_ARM_DIV_HPP

#include "cast.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>

namespace ui::arm::neon {

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(
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
                    return from_vec<T>(
                        vdiv_f32(to_vec(num), to_vec(den))
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vdivq_f32(to_vec(num), to_vec(den))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vdivq_f64(to_vec(num), to_vec(den))
                    );
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec(vdiv_f16(to_vec(num), to_vec(den)));
                } else if constexpr (N == 8) {
                    return from_vec(vdivq_f16(to_vec(num), to_vec(den)));
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
} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_DIV_HPP
