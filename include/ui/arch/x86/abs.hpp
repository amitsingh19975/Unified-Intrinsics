#ifndef AMT_UI_ARCH_X86_JOIN_HPP
#define AMT_UI_ARCH_X86_JOIN_HPP

#include "cast.hpp"
#include "../emul/abs.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::abs_diff(lhs, rhs);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (size == sizeof(__m128)) {
                    auto l = to_vec(lhs);
                    auto r = to_vec(rhs);
                    auto cmp = _mm_cmpgt_epi8(a,b)
                } else if constexpr (sizeo * 2 == sizeof(__m128)) {
                    return abs_diff(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)));
                }
            }

            return join(
                abs_diff(lhs.lo, rhs.lo),
                abs_diff(lhs.hi, rhs.hi)
            );
        }
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_JOIN_HPP
