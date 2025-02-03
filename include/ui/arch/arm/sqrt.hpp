#ifndef AMT_UI_ARCH_ARM_SQRT_HPP
#define AMT_UI_ARCH_ARM_SQRT_HPP

#include "cast.hpp"
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>

namespace ui::arm {

    template<std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(std::sqrt(v.val))
            };
        } else {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vsqrt_f32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vsqrtq_f32(to_vec(v)));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vsqrtq_f64(to_vec(v)));
                    }
                }
            #endif

            return join(
                sqrt(v.lo),
                sqrt(v.hi)
            );
        }
    }

}

#endif // AMT_UI_ARCH_ARM_SQRT_HPP
