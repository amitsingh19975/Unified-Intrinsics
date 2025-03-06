#ifndef AMT_UI_ARCH_X86_DIV_HPP
#define AMT_UI_ARCH_X86_DIV_HPP

#include "cast.hpp"
#include "../emul/div.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(num);
        if constexpr (N == 1) {
            return emul::div(num, den);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_div_ps(to_vec(num), to_vec(den)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_div_pd(to_vec(num), to_vec(den)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(div(cast<float>(num), cast<float>(den)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                return div(from_vec<T>(fit_to_vec(num)), from_vec<T>(fit_to_vec(den))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_div_ps(to_vec(num), to_vec(den)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_div_pd(to_vec(num), to_vec(den)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(div(cast<float>(num), cast<float>(den)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_div_ps(to_vec(num), to_vec(den)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_div_pd(to_vec(num), to_vec(den)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(div(cast<float>(num), cast<float>(den)));
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
        static constexpr auto bits = sizeof(num);
        if constexpr (N == 1) {
            return emul::div(num, den);
        } else {
            if constexpr (sizeof(T) == 8) {
                auto tn = cast<double>(num);
                auto td = cast<double>(den);
                auto res = round<std::float_round_style::round_toward_zero>(div(tn, td));
                return cast<T>(res);
            } else {
                auto tn = cast<float>(num);
                auto td = cast<float>(den);
                auto res = round<std::float_round_style::round_toward_zero>(div(tn, td));
                return cast<T>(res);
            }
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto rem(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        auto q = round<std::float_round_style::round_toward_zero>(div(num, den));
        auto temp = fused_mul_acc(num, q, den, op::sub_t{});
        return temp;
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rem(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        return mul_acc(num, div(num, den), den, op::sub_t{});
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_DIV_HPP
