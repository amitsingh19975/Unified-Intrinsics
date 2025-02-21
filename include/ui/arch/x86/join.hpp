#ifndef AMT_UI_ARCH_X86_JOIN_HPP
#define AMT_UI_ARCH_X86_JOIN_HPP

#include "../../vec_headers.hpp"
#include "../../forward.hpp"
#include "../../float.hpp"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto join_impl(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<2 * N, T> {
        using ret_t = Vec<2 * N, T>;
        constexpr auto cast = [](auto v) -> ret_t {
            return std::bit_cast<ret_t>(v);
        };
        static constexpr auto size = sizeof(x);
        if constexpr (N == 1) {
            return { x, y };
        } else {
            if constexpr (size * 2 == sizeof(__m128)) {
                auto x0 = _mm_cvtsi64_si128(std::bit_cast<std::int64_t>(x));
                auto y0 = _mm_cvtsi64_si128(std::bit_cast<std::int64_t>(y));
                auto temp = _mm_unpacklo_epi64(x0, y0);

                if constexpr (std::same_as<T, float>) {
                    return cast(_mm_castsi128_ps(temp));
                } else if constexpr (std::same_as<T, double>) {
                    return cast(_mm_castsi128_pd(temp));
                } else {
                    return cast(temp);
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size * 2 == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    auto x0 = std::bit_cast<__m128>(x);
                    auto y0 = std::bit_cast<__m128>(y);
                    return cast(_mm256_set_m128(y0, x0));
                } else if constexpr (std::same_as<T, double>) {
                    auto x0 = std::bit_cast<__m128d>(x);
                    auto y0 = std::bit_cast<__m128d>(y);
                    return cast(_mm256_set_m128d(y0, x0));
                } else {
                    auto x0 = std::bit_cast<__m128i>(x);
                    auto y0 = std::bit_cast<__m128i>(y);
                    return cast(_mm256_set_m128i(y0, x0));
                }
            }

            #endif
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size * 2 == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    auto x0 = std::bit_cast<__m256>(x);
                    auto y0 = std::bit_cast<__m256>(y);
                    return cast(_mm512_insertf32x8(_mm512_castps256_ps512(x0), y0));
                } else if constexpr (std::same_as<T, double>) {
                    auto x0 = std::bit_cast<__m256d>(x);
                    auto y0 = std::bit_cast<__m256d>(y);
                    return cast(_mm512_insertf64x4(_mm512_castpd256_pd512(x0), y0));
                } else {
                    auto x0 = std::bit_cast<__m256i>(x);
                    auto y0 = std::bit_cast<__m256i>(y);
                    return cast(_mm512_inserti64x4(_mm512_castsi256_si512(x0), y0));
                }
            }
            #endif

            return { x, y };
        }
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_JOIN_HPP
