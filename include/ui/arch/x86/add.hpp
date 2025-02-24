#ifndef AMT_UI_ARCH_X86_ADD_HPP
#define AMT_UI_ARCH_X86_ADD_HPP

#include "cast.hpp"
#include "logical.hpp"
#include "shift.hpp"
#include "add.hpp"
#include "../emul/add.hpp"
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

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::add(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_add_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_add_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(add(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_add_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_add_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_add_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_add_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_add_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_add_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(add(cast<float>(l), cast<float>(r)));
                } else {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_add_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_add_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_add_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_add_epi64(l, r));
                    }
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_add_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_add_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(add(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_add_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_add_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_add_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_add_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif
            return join(
                add<false>(lhs.lo, rhs.lo),
                add<false>(lhs.hi, rhs.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        using result_t = internal::widening_result_t<T>;
        return add(cast<result_t>(lhs), cast<result_t>(rhs));
    }

// MARK: Narrowing Addition
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        auto t0 = bitwise_and(lhs, rhs);
        auto t1 = bitwise_xor(lhs, rhs);
        auto t2 = shift_right<1>(t1); 
        return add(t0, t2);
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <bool Merge = true, std::size_t N, std::integral T>
        requires (sizeof(T) > 1)
    UI_ALWAYS_INLINE auto high_narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::narrowing_result_t<T>> {
        static constexpr auto size = sizeof(lhs);
        using result_t = internal::narrowing_result_t<T>; 
        if constexpr (N == 1) {
            return emul::high_narrowing_add(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto sum = add(lhs, rhs);
                if constexpr (sizeof(T) == 2) {
                    sum = shift_right<8>(sum);
                    auto v = to_vec(sum);
                    auto res = _mm_packs_epi16(v, v);
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 4) {
                    sum = shift_right<16>(sum);
                    auto v = to_vec(sum);
                    auto res = _mm_packs_epi32(v, v);
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm_shuffle_epi32(to_vec(sum), _MM_SHUFFLE(2, 0, 3, 1));
                    return from_vec<result_t>(res).lo;
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return high_narrowing_add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                auto sum = add(lhs, rhs);
                if constexpr (sizeof(T) == 2) {
                    sum = shift_right<8>(sum);
                    auto v = to_vec(sum);
                    auto res = _mm256_packs_epi16(v, v);
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 4) {
                    sum = shift_right<16>(sum);
                    auto v = to_vec(sum);
                    auto res = _mm256_packs_epi32(v, v);
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 8) {
                    auto v = _mm256_castsi256_pd(_mm256_shuffle_epi32(to_vec(sum), _MM_SHUFFLE(2, 0, 3, 1)));
                    auto res = _mm256_castpd_si256(_mm256_permute4x64_pd(v, _MM_SHUFFLE(0, 0, 2, 0)));
                    return from_vec<result_t>(res).lo;
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return high_narrowing_add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            // TODO: Add AVX512 implementation
            return join(
                high_narrowing_add<false>(lhs.lo, rhs.lo),
                high_narrowing_add<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Saturating Add
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::sat_add(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_adds_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_adds_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm_set1_epi32(0x7fffffff);
                        auto res = _mm_add_epi32(l, r);
                        // (~(l ^ r)) & (l ^ res) => checks if carry

                        // 1. Get the sign bit
                        auto res_sat = _mm_srli_epi32(l, 31); 

                        // 2. add sign bit. If sign bit is 1, res_sat will be INT32_MIN; otherwise, INT32_MAX
                        res_sat = _mm_add_epi32(res_sat, sign);

                        // 3. l ^ res
                        auto res_xor_l = _mm_xor_si128(res, l);
                        // 4. r ^ l
                        auto r_xor_l = _mm_xor_si128(r, l);
                        // 5. ~(l^r) & (l & res)
                        res_xor_l = _mm_andnot_si128(r_xor_l, res_xor_l);
                        // 6. Gets carry bit
                        res_xor_l = _mm_srai_epi32(res_xor_l, 31);
                        // 7. (l ^ res) & (INT32_MIN or INT32_MAX)
                        res_sat = _mm_and_si128(res_xor_l, res_sat);
                        res = _mm_andnot_si128(res_xor_l, res);
                        res = _mm_or_si128(res, res_sat); 
                        return from_vec<T>(res);
                    }
                } else {
                    static constexpr auto sign_mask = static_cast<std::int32_t>(0x8000'0000);
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_adds_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_adds_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm_set1_epi32(0x8000'0000);
                        auto sum = _mm_add_epi32(l, r);
                        auto subsum = _mm_sub_epi32(sum, sign);
                        auto suba = _mm_sub_epi32(l, sign); 
                        auto c = _mm_cmpgt_epi32 (suba, subsum);
                        auto res = _mm256_or_si128(sum, cmp);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto mask = _mm_set_epi32(sign_mask, 0, sign_mask, 0);
                        auto sum = _mm_add_epi64(l, r);
                        auto subsum = _mm_sub_epi64(sum, mask);
                        auto suba = _mm_sub_epi64(l, mask);
                        auto cmp = _mm_cmpgt_epi64(suba, subsum);
                        return from_vec<T>(_mm_or_si128(sum, cmp));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return sat_add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_adds_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_adds_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm256_set1_epi32(0x7fffffff);
                        auto res = _mm256_add_epi32(l, r);
                        // (~(l ^ r)) & (l ^ res) => checks if carry

                        // 1. Get the sign bit
                        auto res_sat = _mm256_srli_epi32(l, 31); 

                        // 2. add sign bit. If sign bit is 1, res_sat will be INT32_MIN; otherwise, INT32_MAX
                        res_sat = _mm256_add_epi32(res_sat, sign);

                        // 3. l ^ res
                        auto res_xor_l = _mm256_xor_si256(res, l);
                        // 4. r ^ l
                        auto r_xor_l = _mm256_xor_si256(r, l);
                        // 5. ~(l^r) & (l & res)
                        res_xor_l = _mm256_andnot_si256(r_xor_l, res_xor_l);
                        // 6. Gets carry bit
                        res_xor_l = _mm256_srai_epi32(res_xor_l, 31);
                        // 7. (l ^ res) & (INT32_MIN or INT32_MAX)
                        res_sat = _mm256_and_si256(res_xor_l, res_sat);
                        res = _mm256_andnot_si256(res_xor_l, res);
                        res = _mm256_or_si256(res, res_sat); 
                        return from_vec<T>(res);
                    }
                } else {
                    static constexpr auto sign_mask = static_cast<std::int32_t>(0x8000'0000);
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_adds_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_adds_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = _mm256_set1_epi32(sign_mask);
                        auto sum = _mm256_add_epi32(l, r);
                        auto subsum = _mm256_sub_epi32(sum, sign);
                        auto suba = _mm256_sub_epi32(l, sign); 
                        auto c = _mm256_cmpgt_epi32 (suba, subsum);
                        auto res = _mm256_or_si256(sum, cmp);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto mask = _mm256_set_epi32(sign_mask, 0, sign_mask, 0, sign_mask, 0, sign_mask, 0);
                        auto sum = _mm256_add_epi64(l, r);
                        auto subsum = _mm256_sub_epi64(sum, mask);
                        auto suba = _mm256_sub_epi64(l, mask);
                        auto cmp = _mm256_cmpgt_epi64(suba, subsum);
                        return from_vec<T>(_mm256_or_si256(sum, cmp));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return sat_add(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            // TODO: Add AVX512
            return join(
                sat_add<false>(lhs.lo, rhs.lo),
                sat_add<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Pairwise Addition
    template <bool Merge = true, std::size_t N, typename T>
        requires (N != 1)
    UI_ALWAYS_INLINE auto padd(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 2) {
            return emul::padd(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm_hadd_ps(l, r);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(padd(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 2) {
                        auto res = _mm_hadd_epi16(l, r);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto res = _mm_hadd_epi32(l, r);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(cast<internal::widening_result_t<T>>(from_vec<T>(fit_to_vec(lhs))).lo);
                    auto r = to_vec(cast<internal::widening_result_t<T>>(from_vec<T>(fit_to_vec(rhs))).lo);
                    auto t0 = _mm_hadd_epi16(l, r);
                    auto res = _mm_shuffle_epi8(t0, *reinterpret_cast<__m128i const*>(constants::mask8_16_even_odd));
                    return from_vec<T>(res).lo;
                }
                if constexpr (Merge) {
                    return padd(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm256_hadd_ps(l, r);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm256_hadd_pd(l, r);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(padd(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 2) {
                        auto res = _mm256_hadd_epi16(l, r);
                        res = _mm256_permute4x64_epi64(res, 0xD8);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto res = _mm256_hadd_epi32(l, r);
                        res = _mm256_permute4x64_epi64(res, 0xD8);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto t0 = _mm256_unpackhi_epi64(l, r);
                        auto t1 = _mm256_unpacklo_epi64(l, r);
                        auto res = _mm256_permute4x64_epi64(_mm256_add_epi64(t0, t1), 0xD8);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && sizeof(T) != 1) {
                if constexpr (Merge) {
                    return padd(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
                }
            }
            #endif

            // TODO: Add AVX512
            return join(
                padd<false>(lhs.lo, lhs.hi),
                padd<false>(rhs.lo, rhs.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::padd_t op
    ) noexcept -> T {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 2) {
            return emul::fold(v, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
                    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
                    return _mm_cvtss_f32(a);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return static_cast<T>(fold(cast<float>(v)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        auto b = _mm_unpackhi_epi64(a, a);
                        auto sum = _mm_add_epi8(a, b);
                        auto res = _mm_sad_epu8(sum, _mm_setzero_si128());
                        return static_cast<T>(_mm_cvtsi128_si32(res));
                    } else if constexpr (sizeof(T) == 2) {
                        auto b = _mm_unpackhi_epi64(a, a);
                        auto sum = _mm_add_epi16(a, b);
                        auto shf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
                        sum = _mm_add_epi16(sum, shf);
                        auto shifted = _mm_srli_epi32(sum, 16);
                        sum = _mm_add_epi16(sum, shifted);
                        return static_cast<T>(_mm_cvtsi128_si32(sum));
                    } else if constexpr (sizeof(T) == 4) {
                        auto b = _mm_unpackhi_epi64(a, a);
                        auto sum = _mm_add_epi32(a, b);
                        auto shf = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
                        sum = _mm_add_epi32(sum, shf);
                        return static_cast<T>(_mm_cvtsi128_si32(sum));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return fold(from_vec<T>(fit_to_vec(v)), op);
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    auto lo = _mm256_castps256_ps128(a);
                    auto hi = _mm256_extractf128_ps(a, 1);
                    return fold<false>(from_vec<T>(_mm_add_ps(lo, hi)), op);
                } else if constexpr (std::same_as<T, double>) {
                    auto lo = _mm256_castpd256_pd128(a);
                    auto hi = _mm256_extractf128_pd(a, 1);
                    return fold<false>(from_vec<T>(_mm_add_pd(lo, hi)), op);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return static_cast<T>(fold(cast<float>(v)));
                } else {
                    auto lo = _mm256_castsi256_si128(a);
                    auto hi = _mm256_extracti128_si256(a, 1);
                    return fold<false>(add(from_vec<T>(lo), from_vec<T>(hi)), op);
                }
            }
            #endif

            // TODO: Add AVX512
            return static_cast<T>(fold<false>(v.lo, op) + fold<false>(v.hi, op));
        }
    }
// !MARK

// MARK: Widening Pairwise Addition
    template <std::size_t N, std::integral T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto widening_padd(
        Vec<N, T> const& v
    ) noexcept -> Vec<N / 2, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto temp = cast<result_t>(v);
        auto res = padd(temp, temp);
        return res.lo;
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_padd(
        Vec<N, internal::widening_result_t<T>> const& x,
        Vec<2 * N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return add(x, widening_padd(v));
    }
// !MARK

// MARK: Addition across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::add_t op
    ) noexcept -> T {
        return fold(v, op::padd_t{});
    }
// !MARK

// MARK: Widening Addition across vector
    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_fold(
        Vec<N, T> const& v,
        op::add_t op
    ) noexcept -> internal::widening_result_t<T> {
        using result_t = internal::widening_result_t<T>;
        auto vt = cast<result_t>(v);
        return fold(vt, op);
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_ADD_HPP
