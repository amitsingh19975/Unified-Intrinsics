#ifndef AMT_UI_ARCH_X86_ABS_HPP
#define AMT_UI_ARCH_X86_ABS_HPP

#include "cast.hpp"
#include "cmp.hpp"
#include "logical.hpp"
#include "../emul/abs.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "sub.hpp"
#include "add.hpp"

namespace ui::x86 {

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto mask = ( ~mask_inner_t<T>(0) ) >> 1;
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::abs(v);
        } else {
            if constexpr (!std::is_signed_v<T>) return v;
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    auto m = _mm_castsi128_ps(_mm_set1_epi32(static_cast<std::int32_t>(mask))); 
                    return from_vec<T>(_mm_and_ps(a, m)); 
                } else if constexpr (std::same_as<T, double>) {
                    auto m = _mm_castsi128_pd(_mm_set1_epi64x(static_cast<std::int64_t>(mask))); 
                    return from_vec<T>(_mm_and_pd(a, m)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(abs(cast<float>(v)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_abs_epi8(a));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_abs_epi16(a));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_abs_epi32(a));
                    } else if constexpr (sizeof(T) == 8) {
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        return from_vec<T>(_mm_abs_epi64(a));
                        #endif
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return abs(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    auto m = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<std::int32_t>(mask))); 
                    return from_vec<T>(_mm256_and_ps(a, m)); 
                } else if constexpr (std::same_as<T, double>) {
                    auto m = _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<std::int64_t>(mask))); 
                    return from_vec<T>(_mm256_and_pd(a, m)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(abs(cast<float>(v)));
                } else if constexpr (std::is_signed_v<T>) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_abs_epi8(a));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_abs_epi16(a));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_abs_epi32(a));
                    } else if constexpr (sizeof(T) == 8) {
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        return from_vec<T>(_mm256_abs_epi64(a));
                        #else
                        auto neg = _mm256_sub_epi64(_mm256_setzero_si256(), a);
                        auto ad = _mm256_castsi256_pd(a);
                        auto nd = _mm256_castsi256_pd(neg);
                        auto res = _mm256_castpd_si256(_mm256_blendv_pd(ad, nd, ad));
                        return from_vec<T>(res);
                        #endif
                    }
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return abs(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    auto m = _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<std::int32_t>(mask))); 
                    return from_vec<T>(_mm512_and_ps(a, m)); 
                } else if constexpr (std::same_as<T, double>) {
                    auto m = _mm512_castsi512_pd(_mm512_set1_epi64x(static_cast<std::int64_t>(mask))); 
                    return from_vec<T>(_mm512_and_pd(a, m)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(abs(cast<float>(v)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_abs_epi8(a));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_abs_epi16(a));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_abs_epi32(a));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_abs_epi64(a));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return abs(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif
            return join(
                abs<false>(v.lo),
                abs<false>(v.hi)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T>;

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            auto s0 = sub(lhs, rhs);
            auto s1 = sub(rhs, lhs);
            auto c = cmp(lhs, rhs, op::greater_t{});
            auto temp = bitwise_select(c, s0, s1);
            return temp;
        } else {
            auto s0 = sat_sub(lhs, rhs);
            auto s1 = sat_sub(rhs, lhs);
            return bitwise_or(s0, s1);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return abs_diff(cast<result_t>(lhs), cast<result_t>(rhs));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        auto res = abs_diff(lhs, rhs);
        return add(acc, res);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        auto res = widening_abs_diff(lhs, rhs);
        return add(acc, res);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        auto m = Vec<N, T>::load(static_cast<T>(1) << (sizeof(T) * 8 - 1));
        auto a = abs(v);
        auto c = cmp(a, m, op::equal_t{});
        return bitwise_xor(a, rcast<T>(c));
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_ABS_HPP
