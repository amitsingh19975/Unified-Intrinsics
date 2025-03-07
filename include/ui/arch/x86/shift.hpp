#ifndef AMT_UI_ARCH_X86_SHIFT_HPP
#define AMT_UI_ARCH_X86_SHIFT_HPP

#include "cast.hpp"
#include "../emul/shift.hpp"
#include "logical.hpp"
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

// MARK: Left shift
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::shift_left(v, rcast<std::make_unsigned_t<T>>(s));
        } else {
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    return emul::shift_left(v, s);
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<T>(_mm_sllv_epi16(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(_mm_sllv_epi32(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(_mm_sllv_epi64(to_vec(v), to_vec(s)));
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return shift_left(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }

            if constexpr (size == sizeof(__m256)) {
                if constexpr (sizeof(T) == 1) {
                    return emul::shift_left(v, s);
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<T>(_mm256_sllv_epi16(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(_mm256_sllv_epi32(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(_mm256_sllv_epi64(to_vec(v), to_vec(s)));
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return shift_left(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }

            if constexpr (size == sizeof(__m512)) {
                if constexpr (sizeof(T) == 1) {
                    return emul::shift_left(v, s);
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<T>(_mm512_sllv_epi16(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(_mm512_sllv_epi32(to_vec(v), to_vec(s)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(_mm512_sllv_epi64(to_vec(v), to_vec(s)));
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return shift_left(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }
            return join(
                shift_left<false>(lhs.lo, s.lo),
                shift_left<false>(lhs.hi, s.lo)
            );
            #else
            return emul::shift_left(v, rcast<std::make_unsigned_t<T>>(s));
            #endif
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return shift_left(v, rcast<std::make_signed_t<T>>(s)); 
    }
 
    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        alignas(16) static constexpr std::uint16_t mask0_16[9] = {0xffff, 0xfeff, 0xfcff, 0xf8ff, 0xf0ff,  0xe0ff, 0xc0ff, 0x80ff, 0xff};
        if constexpr (N == 1) {
            return emul::shift_left<Shift>(v);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    auto mask0 = _mm_set1_epi16(static_cast<std::int16_t>(mask0_16[Shift]));
                    auto res = _mm_slli_epi16(a, Shift);
                    res = _mm_and_si128(res, mask0);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm_slli_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm_slli_epi32(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm_slli_epi64(a, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto t = to_vec(cast<::ui::internal::widening_result_t<T>>(v));
                    auto res = _mm_slli_epi16(t, Shift);
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm_shuffle_epi8(res, *reinterpret_cast<__m128i const*>(constants::mask8_16_even_odd));
                        return from_vec<T>(res).lo;
                    } else {
                        res = _mm_and_si128(res, _mm_set1_epi16(0xff));
                        return from_vec<T>(_mm_packus_epi16(res, res)).lo;
                    }
                }
                if constexpr (Merge) {
                    return shift_left<Shift>(from_vec<T>(fit_to_vec(v))).lo;
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    auto mask0 = _mm256_set1_epi16(static_cast<std::int16_t>(mask0_16[Shift]));
                    auto res = _mm256_slli_epi16(a, Shift);
                    res = _mm256_and_si256(res, mask0);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm256_slli_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm256_slli_epi32(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm256_slli_epi64(a, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return shift_left<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    auto mask0 = _mm512_set1_epi16(static_cast<std::int16_t>(mask0_16[Shift]));
                    auto res = _mm512_slli_epi16(a, Shift);
                    res = _mm512_and_si256(res, mask0);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = _mm512_slli_epi16(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm512_slli_epi32(a, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm512_slli_epi64(a, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return shift_left<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #endif
            return join(
                shift_left<Shift, false>(v.lo),
                shift_left<Shift, false>(v.hi)
            );
        }
    }

// !MARK

// MARK: Saturating Left Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_left(v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_left<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::rounding_shift_left(v, s);
    }
// !MARK

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_rounding_shift_left(v, s);
    }
// !MARK

// MARK: Vector shift left and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto vt = cast<result_t>(v);
        return shift_left<Shift>(vt);
    }
// !MARK

// MARK: Right shift
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::shift_right(v, rcast<std::make_unsigned_t<T>>(s));
        } else {
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m128)) {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_srav_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_srav_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_srav_epi64(to_vec(v), to_vec(s)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_srlv_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_srlv_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_srlv_epi64(to_vec(v), to_vec(s)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return shift_right(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }

            if constexpr (size == sizeof(__m256)) {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_srav_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_srav_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_srav_epi64(to_vec(v), to_vec(s)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_srlv_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_srlv_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_srlv_epi64(to_vec(v), to_vec(s)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return shift_right(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }

            if constexpr (size == sizeof(__m512)) {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_srav_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_srav_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_srav_epi64(to_vec(v), to_vec(s)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return emul::shift_right(v, s);
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_srlv_epi16(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_srlv_epi32(to_vec(v), to_vec(s)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_srlv_epi64(to_vec(v), to_vec(s)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return shift_right(from_vec<T>(fit_to_vec(v)), from_vec<T>(fit_to_vec(v))).lo;
            }
            return join(
                shift_right<false>(lhs.lo, s.lo),
                shift_right<false>(lhs.hi, s.lo)
            );
            #else
            return emul::shift_right(v, rcast<std::make_unsigned_t<T>>(s));
            #endif
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return shift_right(v, rcast<std::make_signed_t<T>>(s));
    }

    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        alignas(16) static constexpr int16_t mask0_16[9] = {0x0000, 0x0080, 0x00c0, 0x00e0, 0x00f0,  0x00f8, 0x00fc, 0x00fe, 0x00ff};
        alignas(16) static constexpr uint16_t mask1_16[9] = {0xffff, 0xff7f, 0xff3f, 0xff1f, 0xff0f,  0xff07, 0xff03, 0xff01, 0xff00};                        
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::shift_right<Shift>(v);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto zero = _mm_setzero_si128();
                        auto mask0 = _mm_set1_epi16(mask0_16[Shift]);
                        auto a_sign = _mm_cmpgt_epi8(zero, a);
                        auto r = _mm_srai_epi16(a, Shift);
                        auto a_sign_mask =  _mm_and_si128(mask0, a_sign);
                        r =  _mm_andnot_si128(mask0, r);
                        auto res = _mm_or_si128(r, a_sign_mask);
                        return from_vec<T>(res);
                    } else {
                        auto mask0 = _mm_set1_epi16(mask1_16[Shift]);
                        auto res = _mm_srli_epi16(a, Shift);
                        return from_vec<T>(_mm_and_si128(res, mask0));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    __m128i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm_srai_epi16(a, Shift);
                    } else {
                        res = _mm_srli_epi16(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    __m128i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm_srai_epi32(a, Shift);
                    } else {
                        res = _mm_srli_epi32(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    __m128i res;
                    if constexpr (std::is_signed_v<T>) {
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        res = _mm_srai_epi64(a, Shift);
                        return from_vec<T>(res);
                        #endif
                    } else {
                        res = _mm_srli_epi64(a, Shift);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto zero = _mm256_setzero_si256();
                        auto mask0 = _mm256_set1_epi16(mask0_16[Shift]);
                        auto a_sign = _mm256_cmpgt_epi8(zero, a);
                        auto r = _mm256_srai_epi16(a, Shift);
                        auto a_sign_mask = _mm256_and_si256(mask0, a_sign);
                        r =  _mm256_andnot_si256(mask0, r);
                        auto res = _mm256_or_si256(r, a_sign_mask);
                        return from_vec<T>(res);
                    } else {
                        auto mask0 = _mm256_set1_epi16(mask1_16[Shift]);
                        auto res = _mm256_srli_epi16(a, Shift);
                        return from_vec<T>(_mm256_and_si256(res, mask0));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    __m256i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm256_srai_epi16(a, Shift);
                    } else {
                        res = _mm256_srli_epi16(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    __m256i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm256_srai_epi32(a, Shift);
                    } else {
                        res = _mm256_srli_epi32(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    __m256i res;
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm256_srai_epi64(a, Shift);
                    } else {
                        res = _mm256_srli_epi64(a, Shift);
                    }
                    #else
                    auto t0 = _mm256_srli_epi64(a, Shift); // a >> Shift
                    if constexpr (std::is_signed_v<T>) {
                        auto zero = _mm256_setzero_si256();
                        auto sign_mask = _mm256_cmpgt_epi64(zero, a); // a < 0
                        auto add_mask = _mm256_slli_epi64(
                            _mm256_set1_epi64x(~0ULL), // 0xff...ff
                            64 - Shift
                        ); // 1s << (64- Shift)
                        auto corr = _mm256_and_si256(sign_mask, add_mask);
                        res = _mm256_or_si256(t0, corr);
                    } else {
                        res = t0;
                    }
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
                    if constexpr (std::is_signed_v<T>) {
                        auto mask0 = _mm512_set1_epi16(mask0_16[Shift]);
                        auto a_sign = _mm512_cmpgt_epi8(zero, a);
                        auto r = _mm512_srai_epi16(a, Shift);
                        auto a_sign_mask = _mm512_and_si512(mask0, a_sign);
                        r =  _mm512_andnot_si128(mask0, r);
                        auto res = _mm512_or_si512(r, a_sign_mask);
                        return from_vec<T>(res);
                    } else {
                        auto mask0 = _mm512_set1_epi16(mask1_16[Shift]);
                        auto res = _mm512_srli_epi16(a, Shift);
                        return from_vec<T>(_mm512_and_si512(res, mask0));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    __m512i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm512_srai_epi16(a, Shift);
                    } else {
                        res = _mm512_srli_epi16(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    __m512i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm512_srai_epi32(a, Shift);
                    } else {
                        res = _mm512_srli_epi32(a, Shift);
                    }
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    __m512i res;
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm512_srai_epi64(a, Shift);
                    } else {
                        res = _mm512_srli_epi64(a, Shift);
                    }
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

// MARK: Saturating Right Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_right(v, s);
    }
// !MARK


// MARK: Vector rounding shift right
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::rounding_shift_right(v, s);
    }

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::rounding_shift_right<Shift>(v);
        } else {
            alignas(16) static constexpr std::uint16_t mask2b[9] = {0x0000, 0x0101, 0x0202, 0x0404, 0x0808, 0x1010, 0x2020, 0x4040, 0x8080};;
            if constexpr (size == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto r = to_vec(shift_right<Shift>(v));
                    auto m1 = _mm_set1_epi16(mask2b[Shift]);
                    auto mb = _mm_and_si128(to_vec(v), m1);
                    mb = _mm_srli_epi16(mb, Shift - 1);
                    return from_vec<T>(_mm_add_epi8(r, mb));
                } else {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto vt = rcast<std::make_unsigned_t<T>>(v);
                    auto mb = shift_right<bits - 1>(
                        shift_left<bits - Shift>(vt)
                    );
                    auto res = shift_right<Shift>(v);
                    return add<true>(res, rcast<T>(mb));
                }
            } else if constexpr (size * 2 == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto a = fit_to_vec(v);
                    auto r = _mm_cvtepi8_epi16(a);
                    auto mb = _mm_slli_epi16(r, 16 - Shift);
                    mb = _mm_srli_epi16(mb, 16);
                    r = _mm_srai_epi16(r, Shift);
                    r = _mm_add_epi16(r, mb);
                    if constexpr (std::is_signed_v<T>) {
                        r = _mm_packs_epi16(r, r);
                    } else {
                        r = _mm_packus_epi16(r, r);
                    }
                    return from_vec<T>(r).lo;
                }
                if constexpr (Merge) {
                    return rounding_shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                if constexpr (sizeof(T) == 1) {
                    auto r = to_vec(shift_right<Shift>(v));
                    auto m1 = _mm256_set1_epi16(mask2b[Shift]);
                    auto mb = _mm256_and_si256(to_vec(v), m1);
                    mb = _mm256_srli_epi16(mb, Shift - 1);
                    return from_vec<T>(_mm256_add_epi8(r, mb));
                } else {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto vt = rcast<std::make_unsigned_t<T>>(v);
                    auto mb = shift_right<bits - 1>(
                        shift_left<bits - Shift>(vt)
                    );
                    auto res = shift_right<Shift>(v);
                    auto temp = add<true>(res, rcast<T>(mb));
                    return temp;
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                if constexpr (sizeof(T) == 1) {
                    auto a = fit_to_vec(v);
                    auto r = _mm256_cvtepi8_epi16(a);
                    auto mb = _mm256_slli_epi16(r, 16 - Shift);
                    mb = _mm256_srli_epi16(mb, 16);
                    r = _mm256_srai_epi16(r, Shift);
                    r = _mm256_add_epi16(r, mb);
                    if constexpr (std::is_signed_v<T>) {
                        r = _mm256_packs_epi16(r, r);
                    } else {
                        r = _mm256_packus_epi16(r, r);
                    }
                    return from_vec<T>(r).lo;
                }
                if constexpr (Merge) {
                    return rounding_shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                if constexpr (sizeof(T) == 1) {
                    auto r = shift_right<Shift>(v);
                    auto m1 = Vec<N, T>::load(mask2b[Shift]);
                    auto mb = bitwise_and(v, m1);
                    mb = shift_right<Shift - 1>(mb);
                    return from_vec<T>(_mm512_add_epi8(to_vec(r), to_vec(mb)));
                } else {
                    static constexpr auto bits = sizeof(T) * 8;
                    auto vt = rcast<std::make_unsigned_t<T>>(v);
                    auto mb = shift_right<bits - 1>(
                        shift_left<bits - Shift>(vt)
                    );
                    auto res = shift_right<Shift>(v);
                    return add<true>(res, rcast<T>(mb));
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                if constexpr (sizeof(T) == 1) {
                    auto a = fit_to_vec(v);
                    auto r = _mm512_cvtepi8_epi16(a);
                    auto mb = _mm512_slli_epi16(r, 16 - Shift);
                    mb = _mm512_srli_epi16(mb, 16);
                    r = _mm512_srai_epi16(r, Shift);
                    r = _mm512_add_epi16(r, mb);
                    if constexpr (std::is_signed_v<T>) {
                        r = _mm512_packs_epi16(r, r);
                    } else {
                        r = _mm512_packus_epi16(r, r);
                    }
                    return from_vec<T>(r).lo;
                }
                if constexpr (Merge) {
                    return rounding_shift_right<Shift>(from_vec<T>(fit_to_vec(v))).lo;
                }
            }
            #endif
            return join(
                rounding_shift_right<Shift, false>(v.lo),
                rounding_shift_right<Shift, false>(v.hi)
            );
        }
    }
// !MARK

// MARK: Vector saturating rounding shift right
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_rounding_shift_right(v, s);
    }
// !MARK

// MARK: Vector shift right and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto vt = cast<result_t>(v);
        return shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift right and accumulate
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right_accumulate(
        Vec<N, T> const& a,
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        auto shifted = rounding_shift_right<Shift>(v);
        return add<true>(a, shifted); 
    }
// !MARK

// MARK: Vector shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        auto shifted = shift_right<Shift>(v);
        return cast<result_t>(shifted);
    }
// !MARK

// MARK: Vector saturating shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_narrowing_shift_right<Shift>(v);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_unsigned_narrowing_shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector saturating rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_rounding_narrowing_shift_right<Shift>(v);
    }
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_rounding_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_rounding_unsigned_narrowing_shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        auto shifted = rounding_shift_right<Shift>(v);
        return cast<result_t>(shifted);
    }
// !MARK

// MARK: Vector shift left and insert
    /**
     * @brief It inserts 'Shift' amount of LSB of 'a' into 'b' shifted by 'Shift'.
     * @code
     * (b << Shift) | (a & ((1 << (Shift + 1)) - 1))
     * @codeend
     * @tparam Shift amount of shift
     * @param a masked LSB will be inserted into 'b'
     * @param b will be shifted by 'Shift'
    */

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto insert_shift_left(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        using utype = std::make_unsigned_t<T>;
        auto ta = rcast<utype>(a);
        auto tb = rcast<utype>(b);
        if constexpr (sizeof(T) == 1) {
            alignas(16) static constexpr std::uint8_t mask_right[8] = {0x0, 0x1, 0x3, 0x7, 0x0f, 0x1f, 0x3f, 0x7f};
            auto mask = Vec<N, utype>::load(mask_right[Shift]);
            auto b_shift = shift_left<Shift>(tb);
            auto a_masked = bitwise_and(ta, mask);
            return rcast<T>(bitwise_or(b_shift, a_masked));
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            auto b_shift = shift_left<Shift>(tb);
            auto a_c = shift_left<bits - Shift>(ta);
            a_c = shift_right<bits - Shift>(a_c);
            return rcast<T>(bitwise_or(b_shift, a_c));
        }
    }
 
// !MARK

// MARK: Vector shift right and insert
    /**
     * @brief It inserts 'Shift' amount of MSB of 'a' into 'b' shifted by 'Shift'.
     * @code
     * (b >> Shift) | (a & ((~T(0) << (sizeof(T) * 8 - Shift))))
     * @codeend
     * @tparam Shift amount of shift
     * @param a masked MSB will be inserted into 'b'
     * @param b will be shifted by 'Shift'
    */
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift <= (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto insert_shift_right(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        using utype = std::make_unsigned_t<T>;
        auto ta = rcast<utype>(a);
        auto tb = rcast<utype>(b);
        if constexpr (sizeof(T) == 1) {
            alignas(16) static constexpr std::uint8_t mask_right[9] = {0x0, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff};
            auto mask = Vec<N, utype>::load(mask_right[Shift]);
            auto b_shift = shift_right<Shift>(tb);
            auto a_masked = bitwise_and(ta, mask);
            return rcast<T>(bitwise_or(b_shift, a_masked));
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            auto b_shift = shift_right<Shift>(tb);
            auto a_c = shift_right<bits - Shift>(ta);
            a_c = shift_left<bits - Shift>(a_c);
            return rcast<T>(bitwise_or(b_shift, a_c));
        }
    }
// !MARK

} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_SHIFT_HPP
