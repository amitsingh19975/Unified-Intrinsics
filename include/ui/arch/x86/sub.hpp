#ifndef AMT_UI_ARCH_X86_SUB_HPP
#define AMT_UI_ARCH_X86_SUB_HPP

#include "cast.hpp"
#include "../emul/sub.hpp"
#include "logical.hpp"
#include "shift.hpp"
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

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_sub_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_sub_epi64(l, r));
                    }
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_sub_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif
            return join(
                sub<false>(lhs.lo, rhs.lo),
                sub<false>(lhs.hi, rhs.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return sub(cast<result_t>(lhs), cast<result_t>(rhs));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, internal::widening_result_t<T>> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return sub(cast<result_t>(lhs), rhs);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, internal::widening_result_t<T>> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return sub(lhs, cast<result_t>(rhs));
    }

// MARK: Narrowing Addition
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::halving_sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm_set1_epi8(static_cast<T>(128));
                        l = _mm_add_epi8(l, mx);
                        r = _mm_add_epi8(r, mx);
                    }
                    auto avg = _mm_avg_epu8(l, r);
                    auto res = _mm_sub_epi8(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm_set1_epi16(static_cast<std::int16_t>(0x8000));
                        l = _mm_add_epi16(l, mx);
                        r = _mm_add_epi16(r, mx);
                    }
                    auto avg = _mm_avg_epu16(l, r);
                    auto res = _mm_sub_epi16(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto a = shift_right<1>(lhs); // lhs / 2
                    auto b = shift_right<1>(rhs); // rhs / 2
                    auto res = sub(a, b); // (lhs - rhs) / 2
                    auto t0 = bitwise_notand(lhs, rhs); 
                    t0 = shift_left<bits - 1>(t0);
                    t0 = rcast<T>(shift_right<bits - 1>(rcast<std::make_unsigned_t<T>>(t0)));
                    return sub(res, t0);
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return halving_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm256_set1_epi8(static_cast<T>(128));
                        l = _mm256_add_epi8(l, mx);
                        r = _mm256_add_epi8(r, mx);
                    }
                    auto avg = _mm256_avg_epu8(l, r);
                    auto res = _mm256_sub_epi8(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm256_set1_epi16(static_cast<std::int16_t>(0x8000));
                        l = _mm256_add_epi16(l, mx);
                        r = _mm256_add_epi16(r, mx);
                    }
                    auto avg = _mm256_avg_epu16(l, r);
                    auto res = _mm256_sub_epi16(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto a = shift_right<1>(lhs); // lhs / 2
                    auto b = shift_right<1>(rhs); // rhs / 2
                    auto res = sub(a, b); // (lhs - rhs) / 2
                    auto t0 = bitwise_notand(lhs, rhs); 
                    t0 = shift_left<bits - 1>(t0);
                    t0 = rcast<T>(shift_right<bits - 1>(rcast<std::make_unsigned_t<T>>(t0)));
                    return sub(res, t0);
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm512_set1_epi8(static_cast<T>(128));
                        l = _mm512_add_epi8(l, mx);
                        r = _mm512_add_epi8(r, mx);
                    }
                    auto avg = _mm512_avg_epu8(l, r);
                    auto res = _mm512_sub_epi8(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = _mm512_set1_epi16(static_cast<std::int16_t>(0x8000));
                        l = _mm512_add_epi16(l, mx);
                        r = _mm512_add_epi16(r, mx);
                    }
                    auto avg = _mm512_avg_epu16(l, r);
                    auto res = _mm512_sub_epi16(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto a = shift_right<1>(lhs); // lhs / 2
                    auto b = shift_right<1>(rhs); // rhs / 2
                    auto res = sub(a, b); // (lhs - rhs) / 2
                    auto t0 = bitwise_notand(lhs, rhs); 
                    t0 = shift_left<bits - 1>(t0);
                    t0 = rcast<T>(shift_right<bits - 1>(rcast<std::make_unsigned_t<T>>(t0)));
                    return sub(res, t0);
                }
            }
            #endif
            return join(
                halving_sub<false>(lhs.lo, rhs.lo),
                halving_sub<false>(lhs.hi, rhs.hi)
            );
        }
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <bool Merge = true, std::size_t N, std::integral T>
        requires (sizeof(T) > 1)
    UI_ALWAYS_INLINE auto high_narrowing_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::narrowing_result_t<T>> {
        static constexpr auto size = sizeof(lhs);
        using result_t = internal::narrowing_result_t<T>; 
        if constexpr (N == 1) {
            return emul::high_narrowing_sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto res = to_vec(sub(lhs, rhs));
                if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = _mm_srai_epi16(res, 8);
                        res = _mm_packs_epi16(s, s);
                    } else {
                        auto s = _mm_srli_epi16(res, 8);
                        res = _mm_packus_epi16(s, s);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = _mm_srai_epi32(res, 16);
                        res = _mm_packs_epi32(s, s);
                    } else {
                        auto s = _mm_srli_epi32(res, 16);
                        res = _mm_packus_epi32(s, s);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 8) {
                    res = _mm_shuffle_epi32(res, _MM_SHUFFLE(2, 0, 3, 1));
                    return from_vec<result_t>(res).lo;
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return high_narrowing_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                auto res = to_vec(sub(lhs, rhs));
                if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = _mm256_srai_epi16(res, 8);
                        res = _mm256_packs_epi16(s, s);
                    } else {
                        auto s = _mm256_srli_epi16(res, 8);
                        res = _mm256_packus_epi16(s, s);
                    }
                    // FIXME: We rely on compiler for extraction. Need to check if compiler
                    // produces better assembly or hand-written on then change this.
                    auto temp = from_vec<result_t>(res);
                    return join(
                        temp.lo.lo,
                        temp.hi.lo
                    );
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = _mm256_srai_epi32(res, 16);
                        res = _mm256_packs_epi32(s, s);
                    } else {
                        auto s = _mm256_srli_epi32(res, 16);
                        res = _mm256_packus_epi32(s, s);
                    }
                    // FIXME: We rely on compiler for extraction. Need to check if compiler
                    // produces better assembly or hand-written on then change this.
                    auto temp = from_vec<result_t>(res);
                    return join(
                        temp.lo.lo,
                        temp.hi.lo
                    );
                } else if constexpr (sizeof(T) == 8) {
                    auto r0 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(res, _mm256_set_epi32(
                        1, 3, 5, 7, 1, 3, 5, 7
                    )));
                    r0 = _mm_shuffle_epi32(r0, _MM_SHUFFLE(0,1,2,3));
                    return from_vec<result_t>(r0);
                }
            }
            #endif

            // TODO: Add avx512 support
            return join(
                high_narrowing_sub<false>(lhs.lo, rhs.lo),
                high_narrowing_sub<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Saturating Subtraction
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        static constexpr auto int_max_mask = static_cast<std::int32_t>(0x80000000);
        if constexpr (N == 1) {
            return emul::sat_sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(lhs);
                auto b = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm_subs_epi8(a, b));
                    } else {
                        return from_vec<T>(_mm_subs_epu8(a, b));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm_subs_epi16(a, b));
                    } else {
                        return from_vec<T>(_mm_subs_epu16(a, b));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    __m128i res;
                    if constexpr (std::is_signed_v<T>) {
                        auto mask = _mm_set1_epi32(0x7fffffff);
                        res = _mm_sub_epi32(a, b);
                        auto res_sat = _mm_srli_epi32(a, 31);
                        res_sat = _mm_add_epi32(res_sat, mask);
                        auto res_xor_a = _mm_xor_si128(res, a);
                        auto b_xor_a = _mm_xor_si128(b, a);
                        res_xor_a = _mm_and_si128(b_xor_a, res_xor_a);
                        res_xor_a = _mm_srai_epi32(res_xor_a, 31);
                        res_sat = _mm_and_si128(res_xor_a, res_sat);
                        res = _mm_andnot_si128(res_xor_a, res);
                        return from_vec<T>(_mm_or_si128(res, res_sat));
                    } else {
                        auto min = _mm_min_epu32(a, b);
                        auto mask = _mm_cmpeq_epi32(min, b);
                        res = to_vec(sub(lhs, rhs));
                        return from_vec<T>(_mm_and_si128(res, mask));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (std::is_signed_v<T>) {
                        return emul::sat_sub(lhs, rhs);
                    } else {
                        auto mask = _mm_set_epi32(int_max_mask, 0x0, int_max_mask, 0x0);
                        auto res = to_vec(sub(lhs, rhs));
                        auto suba = to_vec(sub(lhs, from_vec<T>(mask)));
                        auto subb = to_vec(sub(rhs, from_vec<T>(mask)));
                        auto c = _mm_cmpgt_epi64(suba, subb);
                        return from_vec<T>(_mm_and_si128(res, c));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return sat_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(lhs);
                auto b = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm256_subs_epi8(a, b));
                    } else {
                        return from_vec<T>(_mm256_subs_epu8(a, b));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm256_subs_epi16(a, b));
                    } else {
                        return from_vec<T>(_mm256_subs_epu16(a, b));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    __m256i res;
                    if constexpr (std::is_signed_v<T>) {
                        auto mask = _mm256_set1_epi32(0x7fffffff);
                        res = _mm256_sub_epi32(a, b);
                        auto res_sat = _mm256_srli_epi32(a, 31);
                        res_sat = _mm256_add_epi32(res_sat, mask);
                        auto res_xor_a = _mm256_xor_si256(res, a);
                        auto b_xor_a = _mm256_xor_si256(b, a);
                        res_xor_a = _mm256_and_si256(b_xor_a, res_xor_a);
                        res_xor_a = _mm256_srai_epi32(res_xor_a, 31);
                        res_sat = _mm256_and_si256(res_xor_a, res_sat);
                        res = _mm256_andnot_si256(res_xor_a, res);
                        return from_vec<T>(_mm256_or_si256(res, res_sat));
                    } else {
                        auto min = _mm256_min_epu32(a, b);
                        auto mask = _mm256_cmpeq_epi32(min, b);
                        res = to_vec(sub(lhs, rhs));
                        return from_vec<T>(_mm256_and_si256(res, mask));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    auto sign_mask = _mm256_set_epi32(
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0
                    );
                    if constexpr (std::is_signed_v<T>) {
                        auto res = sub(lhs, rhs);
                        auto diff = to_vec(res);
                        auto c0 = _mm256_and_si256(_mm256_xor_si256(diff, a), sign_mask);
                        auto c1 = _mm256_and_si256(_mm256_xor_si256(a, b), sign_mask);
                        auto ovf_temp = _mm256_and_si256(c0, c1);
                        auto ovf_mask = _mm256_cmpeq_epi64(ovf_temp, sign_mask);
                        auto logical = _mm256_srli_epi64(a, 63);
                        auto a_arith = _mm256_sub_epi64(_mm256_setzero_si256(), logical);
                        auto sat_val = _mm256_xor_si256(a_arith, _mm256_set1_epi64x(0x7FFF'FFFF'FFFF'FFFFULL));
                        return from_vec<T>(_mm256_blendv_epi8(diff, sat_val, ovf_mask));
                    } else {
                        auto res = to_vec(sub(lhs, rhs));
                        auto suba = to_vec(sub(lhs, from_vec<T>(sign_mask)));
                        auto subb = to_vec(sub(rhs, from_vec<T>(sign_mask)));
                        auto c = _mm256_cmpgt_epi64(suba, subb);
                        return from_vec<T>(_mm256_and_si256(res, c));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return sat_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(lhs);
                auto b = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm512_subs_epi8(a, b));
                    } else {
                        return from_vec<T>(_mm512_subs_epu8(a, b));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(_mm512_subs_epi16(a, b));
                    } else {
                        return from_vec<T>(_mm512_subs_epu16(a, b));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    __m512i res;
                    if constexpr (std::is_signed_v<T>) {
                        auto mask = _mm512_set1_epi32(0x7fffffff);
                        res = _mm512_sub_epi32(a, b);
                        auto res_sat = _mm512_srli_epi32(a, 31);
                        res_sat = _mm512_add_epi32(res_sat, mask);
                        auto res_xor_a = _mm512_xor_si512(res, a);
                        auto b_xor_a = _mm512_xor_si512(b, a);
                        res_xor_a = _mm512_and_si512(b_xor_a, res_xor_a);
                        res_xor_a = _mm512_srai_epi32(res_xor_a, 31);
                        res_sat = _mm512_and_si512(res_xor_a, res_sat);
                        res = _mm512_andnot_si512(res_xor_a, res);
                        return from_vec<T>(_mm512_or_si512(res, res_sat));
                    } else {
                        auto min = _mm512_min_epu32(a, b);
                        auto mask = _mm512_cmpeq_epi32(min, b);
                        res = to_vec(sub(lhs, rhs));
                        return from_vec<T>(_mm512_and_si512(res, mask));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    auto sign_mask = _mm512_set_epi32(
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0,
                        int_max_mask, 0x0
                    );
                    if constexpr (std::is_signed_v<T>) {
                        auto res = sub(lhs, rhs);
                        auto diff = to_vec(res);
                        auto c0 = _mm512_and_si512(_mm512_xor_si512(diff, a), sign_mask);
                        auto c1 = _mm512_and_si512(_mm512_xor_si512(a, b), sign_mask);
                        auto ovf_temp = _mm512_and_si512(c0, c1);
                        auto ovf_mask = _mm512_cmpeq_epi64(ovf_temp, sign_mask);
                        auto logical = _mm512_srli_epi64(a, 63);
                        auto a_arith = _mm512_sub_epi64(_mm512_setzero_si512(), logical);
                        auto sat_val = _mm512_xor_si512(a_arith, _mm512_set1_epi64x(0x7FFF'FFFF'FFFF'FFFFULL));
                        return from_vec<T>(_mm512_blendv_epi8(diff, sat_val, ovf_mask));
                    } else {
                        auto res = to_vec(sub(lhs, rhs));
                        auto suba = to_vec(sub(lhs, from_vec<T>(sign_mask)));
                        auto subb = to_vec(sub(rhs, from_vec<T>(sign_mask)));
                        auto c = _mm512_cmpgt_epi64(suba, subb);
                        return from_vec<T>(_mm512_and_si512(res, c));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return sat_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            return join(
                sat_sub<false>(lhs.lo, rhs.lo),
                sat_sub<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_SUB_HPP
