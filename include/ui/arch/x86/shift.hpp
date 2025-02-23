#ifndef AMT_UI_ARCH_X86_SHIFT_HPP
#define AMT_UI_ARCH_X86_SHIFT_HPP

#include "cast.hpp"
#include "../emul/shift.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

// MARK: Right shift
    /*template <std::size_t N, std::integral T>*/
    /*UI_ALWAYS_INLINE auto shift_right(*/
    /*    Vec<N, T> const& v,*/
    /*    Vec<N, std::make_unsigned_t<T>> const& s*/
    /*) noexcept -> Vec<N, T> {*/
    /*    return internal::shift_left_right_helper(v, negate(rcast<std::make_signed_t<T>>(s)));*/
    /*}*/
    
    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::shift_right<Shift>(v);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(v);
                auto zero = _mm_setzero_si128();
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr int16_t mask0_16[9] = {0x0000, 0x0080, 0x00c0, 0x00e0, 0x00f0,  0x00f8, 0x00fc, 0x00fe, 0x00ff};
                    auto mask0 = _mm_set1_epi16(mask0_16[Shift]);
                    auto a_sign = _mm_cmpgt_epi8(zero, a);
                    auto r = _mm_srai_epi16(a, Shift);
                    auto a_sign_mask =  _mm_and_si128(mask0, a_sign);
                    r =  _mm_andnot_si128(mask0, r);
                    auto res = _mm_or_si128(r, a_sign_mask);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm_srai_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm_srai_epi32(a, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                auto zero = _mm256_setzero_si256();
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr int16_t mask0_16[9] = {0x0000, 0x0080, 0x00c0, 0x00e0, 0x00f0,  0x00f8, 0x00fc, 0x00fe, 0x00ff};
                    auto mask0 = _mm256_set1_epi16(mask0_16[Shift]);
                    auto a_sign = _mm256_cmpgt_epi8(zero, a);
                    auto r = _mm256_srai_epi16(a, Shift);
                    auto a_sign_mask = _mm256_and_si256(mask0, a_sign);
                    r =  _mm256_andnot_si128(mask0, r);
                    auto res = _mm256_or_si256(r, a_sign_mask);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm256_srai_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm256_srai_epi32(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    auto res = _mm256_srai_epi64(a, Shift);
                    #else
                    auto as = _mm256_srai_epi32(a, Shift); // arithmetic shift
                    auto ls = _mm256_srli_epi64(a, Shift); // logical shift
                    auto res = _mm256_blend_epi32(as, ls, 0xAA);
                    #endif
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto a = to_vec(v);
                auto zero = _mm512_setzero_si512();
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr int16_t mask0_16[9] = {0x0000, 0x0080, 0x00c0, 0x00e0, 0x00f0,  0x00f8, 0x00fc, 0x00fe, 0x00ff};
                    auto mask0 = _mm512_set1_epi16(mask0_16[Shift]);
                    auto a_sign = _mm512_cmpgt_epi8(zero, a);
                    auto r = _mm512_srai_epi16(a, Shift);
                    auto a_sign_mask = _mm512_and_si512(mask0, a_sign);
                    r =  _mm512_andnot_si128(mask0, r);
                    auto res = _mm512_or_si512(r, a_sign_mask);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm512_srai_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm512_srai_epi32(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm512_srai_epi64(a, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif
            return join(
                shift_right<Shift, false>(v.lo),
                shift_right<Shift, false>(v.hi)
            );
        }
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_SHIFT_HPP
