#ifndef AMT_UI_ARCH_X86_CMP_HPP
#define AMT_UI_ARCH_X86_CMP_HPP

#include "cast.hpp"
#include "../emul/cmp.hpp"
#include "logical.hpp"
#include "minmax.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

// MARK: Bitwise equal and 'and' test
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        static constexpr auto size = sizeof(lhs);

        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm_castps_si128(_mm_cmpeq_ps(to_vec(lhs), to_vec(rhs))));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm_castpd_si128(_mm_cmpeq_pd(to_vec(lhs), to_vec(rhs))));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<result_t>(_mm_cmpeq_epi8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<result_t>(_mm_cmpeq_epi16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<result_t>(_mm_cmpeq_epi32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<result_t>(_mm_cmpeq_epi64(to_vec(lhs), to_vec(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(l, r, _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(l, r, _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (sizeof(T) == 1) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    return from_vec<result_t>(_mm256_cmpeq_epi8(l, r));
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    return from_vec<result_t>(_mm256_cmpeq_epi16(l, r));
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    return from_vec<result_t>(_mm256_cmpeq_epi32(l, r));
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    return from_vec<result_t>(_mm256_cmpeq_epi64(l, r));
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif

            // TODO: Add AVX 512
            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::and_test_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            auto zero = Vec<N, T>{};
            auto ones = cmp(zero, zero, op::equal_t{});
            auto res = bitwise_and(lhs, rhs);
            return bitwise_xor(
                cmp(res, zero, op::equal_t{}),
                ones
            );
        }
    }

// !MARK

// MARK:  Greater than or equal to
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                 if constexpr (std::same_as<T, float>) {
                     return from_vec<result_t>(_mm_castps_si128(_mm_cmpge_ps(l, r)));
                 } else if constexpr (std::same_as<T, double>) {
                     return from_vec<result_t>(_mm_castpd_si128(_mm_cmpge_pd(l, r)));
                 } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                     return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                 } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        auto gt = _mm_cmpgt_epi8(l, r);
                        auto eq = _mm_cmpeq_epi8(l, r);
                        auto res = _mm_or_si128(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto gt = _mm_cmpgt_epi16(l, r);
                        auto eq = _mm_cmpeq_epi16(l, r);
                        auto res = _mm_or_si128(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto gt = _mm_cmpgt_epi32(l, r);
                        auto eq = _mm_cmpeq_epi32(l, r);
                        auto res = _mm_or_si128(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto gt = _mm_cmpgt_epi64(l, r);
                        auto eq = _mm_cmpeq_epi64(l, r);
                        auto res = _mm_or_si128(gt, eq);
                        return from_vec<result_t>(res);
                    }
                } else {
                    // x86 does not support unsigned compare so 'lhs >= rhs <=> max(lhs, rhs) == lhs'
                    // lhs: [1, 3, 4], rhs: [2, 2, 5]
                    // lhs >= rhs => [0s, 1s, 0s] 
                    // mx = max(lhs, rhs) = [2, 3, 5]
                    // mx == lhs => [2, 3, 5] == [1, 3, 4] => [0s, 1s, 0s]
                    auto mx = max(lhs, rhs);
                    return cmp(mx, lhs, op::equal_t{});
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(l, r, _CMP_GE_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(l, r, _CMP_GE_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (std::is_signed_v<T>) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        auto gt = _mm256_cmpgt_epi8(l, r);
                        auto eq = _mm256_cmpeq_epi8(l, r);
                        auto res = _mm256_or_si256(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto gt = _mm256_cmpgt_epi16(l, r);
                        auto eq = _mm256_cmpeq_epi16(l, r);
                        auto res = _mm256_or_si256(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto gt = _mm256_cmpgt_epi32(l, r);
                        auto eq = _mm256_cmpeq_epi32(l, r);
                        auto res = _mm256_or_si256(gt, eq);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto gt = _mm256_cmpgt_epi64(l, r);
                        auto eq = _mm256_cmpeq_epi64(l, r);
                        auto res = _mm256_or_si256(gt, eq);
                        return from_vec<result_t>(res);
                    }
                    #endif
                } else {
                    auto mx = max(lhs, rhs);
                    return cmp(mx, lhs, op::equal_t{});
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif

            // TODO: Add AVX 512
            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    // v >= 0
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::greater_equal_t{});
    }
// !MARK

// MARK: Less than or equal to
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::less_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                 if constexpr (std::same_as<T, float>) {
                     return from_vec<result_t>(_mm_castps_si128(_mm_cmple_ps(l, r)));
                 } else if constexpr (std::same_as<T, double>) {
                     return from_vec<result_t>(_mm_castpd_si128(_mm_cmple_pd(l, r)));
                 } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                     return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                 } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        auto ones = _mm_cmpeq_epi8(l, l);
                        auto gt = _mm_cmpgt_epi8(l, r);
                        auto res = _mm_andnot_si128(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto ones = _mm_cmpeq_epi16(l, l);
                        auto gt = _mm_cmpgt_epi16(l, r);
                        auto res = _mm_andnot_si128(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto ones = _mm_cmpeq_epi32(l, l);
                        auto gt = _mm_cmpgt_epi32(l, r);
                        auto res = _mm_andnot_si128(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto ones = _mm_cmpeq_epi64(l, l);
                        auto gt = _mm_cmpgt_epi64(l, r);
                        auto res = _mm_andnot_si128(gt, ones);
                        return from_vec<result_t>(res);
                    }
                } else {
                    // x86 does not support unsigned compare so 'lhs <= rhs <=> min(lhs, rhs) == lhs'
                    // lhs: [1, 3, 4], rhs: [2, 2, 5]
                    // lhs <= rhs => [0s, 1s, 0s] 
                    // mi = min(lhs, rhs) = [1, 2, 4]
                    // mi == lhs => [1, 2, 4] == [1, 3, 4] => [1s, 0s, 1s]
                    auto mx = min(lhs, rhs);
                    return cmp(mx, lhs, op::equal_t{});
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(l, r, _CMP_GE_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(l, r, _CMP_GE_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (std::is_signed_v<T>) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        auto ones = _mm256_cmpeq_epi8(l, l);
                        auto gt = _mm256_cmpgt_epi8(l, r);
                        auto res = _mm256_andnot_si256(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto ones = _mm256_cmpeq_epi16(l, l);
                        auto gt = _mm256_cmpgt_epi16(l, r);
                        auto res = _mm256_andnot_si256(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto ones = _mm256_cmpeq_epi32(l, l);
                        auto gt = _mm256_cmpgt_epi32(l, r);
                        auto res = _mm256_andnot_si256(gt, ones);
                        return from_vec<result_t>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto ones = _mm256_cmpeq_epi64(l, l);
                        auto gt = _mm256_cmpgt_epi64(l, r);
                        auto res = _mm256_andnot_si256(gt, ones);
                        return from_vec<result_t>(res);
                    }
                    #endif
                } else {
                    auto mx = min(lhs, rhs);
                    return cmp(mx, lhs, op::equal_t{});
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif

            // TODO: Add AVX 512
            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    // v <= 0
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::less_equal_t{});
    }
// !MARK

// MARK: Greater Than
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                 if constexpr (std::same_as<T, float>) {
                     return from_vec<result_t>(_mm_castps_si128(_mm_cmpgt_ps(l, r)));
                 } else if constexpr (std::same_as<T, double>) {
                     return from_vec<result_t>(_mm_castpd_si128(_mm_cmpgt_pd(l, r)));
                 } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                     return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                 } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(_mm_cmpgt_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(_mm_cmpgt_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(_mm_cmpgt_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(_mm_cmpgt_epi64(l, r));
                    }
                 } else {
                    if constexpr (sizeof(T) == 1) {
                        auto zeros = _mm_setzero_si128();
                        auto res = _mm_subs_epu8(l, r);
                        return from_vec<result_t>(_mm_cmpgt_epi8(res, zeros));
                    } else if constexpr (sizeof(T) == 2) {
                        auto zeros = _mm_setzero_si128();
                        auto res = _mm_subs_epu16(l, r);
                        return from_vec<result_t>(_mm_cmpgt_epi16(res, zeros));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm_set1_epi32(0x8000'0000);
                        auto a = _mm_sub_epi32(l, sign);
                        auto b = _mm_sub_epi32(r, sign);
                        return from_vec<result_t>(_mm_cmpgt_epi32(a, b));
                    } else if constexpr (sizeof(T) == 8) {
                        auto sign = _mm_set1_epi64x(static_cast<std::int64_t>(0x8000'0000'0000'0000ULL));
                        auto a = _mm_sub_epi64(l, sign);
                        auto b = _mm_sub_epi64(r, sign);
                        return from_vec<result_t>(_mm_cmpgt_epi64(a, b));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(l, r, _CMP_GT_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(l, r, _CMP_GT_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (std::is_signed_v<T>) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(_mm256_cmpgt_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(_mm256_cmpgt_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(_mm256_cmpgt_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(_mm256_cmpgt_epi64(l, r));
                    }
                    #endif
                } else {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        auto zeros = _mm256_setzero_si256();
                        auto res = _mm256_subs_epu8(l, r);
                        return from_vec<result_t>(_mm256_cmpgt_epi8(res, zeros));
                    } else if constexpr (sizeof(T) == 2) {
                        auto zeros = _mm256_setzero_si256();
                        auto res = _mm256_subs_epu16(l, r);
                        return from_vec<result_t>(_mm256_cmpgt_epi16(res, zeros));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm256_set1_epi32(0x8000'0000);
                        auto a = _mm256_sub_epi32(l, sign);
                        auto b = _mm256_sub_epi32(r, sign);
                        return from_vec<result_t>(_mm256_cmpgt_epi32(a, b));
                    } else if constexpr (sizeof(T) == 8) {
                        auto sign = _mm256_set1_epi64x(static_cast<std::int64_t>(0x8000'0000'0000'0000ULL));
                        auto a = _mm256_sub_epi64(l, sign);
                        auto b = _mm256_sub_epi64(r, sign);
                        return from_vec<result_t>(_mm256_cmpgt_epi64(a, b));
                    }
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif

            // TODO: Add AVX 512
            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::greater_t{});
    }
// !MARK


// MARK: Less Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::less_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(rhs, lhs, op::greater_t{});
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::less_t{});
    }
// !MARK

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T>;

// MARK: Absolute greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(
            abs<true>(lhs),
            abs<true>(rhs),
            op::greater_equal_t{}
        );
    }
// !MARK

// MARK: Absolute less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_equal_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(
            abs<true>(lhs),
            abs<true>(rhs),
            op::less_equal_t{}
        );
    }
// !MARK

// MARK: Absolute greater than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(
            abs<true>(lhs),
            abs<true>(rhs),
            op::greater_t{}
        );
    }
// !MARK

// MARK: Absolute less than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(
            abs<true>(lhs),
            abs<true>(rhs),
            op::less_t{}
        );
    }
// !MARK

} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_CMP_HPP
