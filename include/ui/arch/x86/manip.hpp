#ifndef AMT_UI_ARCH_X86_MANIP_HPP
#define AMT_UI_ARCH_X86_MANIP_HPP

#include "cast.hpp"
#include "logical.hpp"
#include "shift.hpp"
#include "bit.hpp"
#include "../emul/manip.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {
    // MARK: Copy vector lane
    template <unsigned ToLane, unsigned FromLane, bool Merge = true, std::size_t N, std::size_t M, typename T>
        requires (ToLane < N && FromLane < M)
    UI_ALWAYS_INLINE auto copy(
        Vec<N, T> const& to,
        Vec<M, T> const& from
    ) noexcept -> Vec<N, T> {
        using mtype = mask_inner_t<T>;
        if constexpr (N == 1 || M == 1) {
            return emul::copy<ToLane, FromLane>(to, from);
        } else if constexpr (N == M) {
            if constexpr (ToLane == FromLane) {
                constexpr auto create_mask = []<std::size_t... Is>(std::index_sequence<Is...>) {
                    auto v = Vec<N, mtype>{};
                    ((v[Is] = (Is == FromLane ? ~mtype(0) : mtype(0))),...);
                    return v;
                };
                auto from_mask = create_mask(std::make_index_sequence<M>{});
                auto tmp = bitwise_select(from_mask, from, to);
                return tmp;
            }
        }

        auto f = Vec<N, T>::load(from[FromLane]);
        return copy<ToLane, ToLane>(to, f);
    }
// !MARK

// MARK: Reverse bits within elements
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto reverse_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using mtype = mask_inner_t<T>;

        if constexpr (N == 1) {
            return emul::reverse_bits(v);
        } else {
            static constexpr auto bits = sizeof(v);
            constexpr auto rev_helper = []<std::size_t M, typename U>(auto&& self, Vec<M, U> const& a) -> Vec<M, U> {
                if constexpr (sizeof(U) == 2) {
                    return bitwise_or(
                        shift_right<8>(a),
                        shift_left<8>(a)
                    );
                } else if constexpr (sizeof(U) == 4) {
                    auto t = rcast<std::uint16_t>(a);
                    auto rev = rcast<U>(self(self, t));
                    return bitwise_or(
                        shift_right<16>(rev),
                        shift_left<16>(rev)
                    );
                } else if constexpr (sizeof(U) == 8) {
                    auto t = rcast<std::uint32_t>(a);
                    auto rev = rcast<U>(self(self, t));
                    return bitwise_or(
                        shift_right<32>(rev),
                        shift_left<32>(rev)
                    );
                }
            };
            if constexpr (bits == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    static const constexpr std::uint8_t table8[] = {
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
                    };
                    auto lookup = *reinterpret_cast<__m128i const*>(table8);
                    auto lo = _mm_and_si128(a, _mm_set1_epi8(0x0F));
                    auto hi = to_vec(shift_right<4>(from_vec<std::uint8_t>(a)));
                    auto res = _mm_or_si128(
                        to_vec(shift_left<4>(from_vec<std::uint8_t>(_mm_shuffle_epi8(lookup, lo)))),
                        _mm_shuffle_epi8(lookup, hi)
                    );
                    return from_vec<T>(res);
                } else {
                    auto res = rev_helper(rev_helper, rcast<mtype>(reverse_bits(rcast<std::uint8_t>(v))));
                    return rcast<T>(res);
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return reverse_bits(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (sizeof(T) == 1) {
                    auto a = to_vec(rcast<std::uint8_t>(v));
                    static const constexpr std::uint8_t table8[] = {
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
                    };
                    auto lookup = *reinterpret_cast<__m256i const*>(table8);
                    auto lo = _mm256_and_si256(a, _mm256_set1_epi8(0x0F));
                    auto hi = to_vec(shift_right<4>(from_vec<std::uint8_t>(a)));
                    auto res = _mm256_or_si256(
                        to_vec(shift_left<4>(from_vec<std::uint8_t>(_mm256_shuffle_epi8(lookup, lo)))),
                        _mm256_shuffle_epi8(lookup, hi)
                    );
                    return rcast<T>(from_vec<std::uint8_t>(res));
                } else {
                    auto res = rev_helper(rev_helper, rcast<mtype>(reverse_bits(rcast<std::uint8_t>(v))));
                    return rcast<T>(res);
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    static const constexpr std::uint8_t table8[] = {
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
                    };
                    auto lookup = *reinterpret_cast<__m512i const*>(table8);
                    auto lo = _mm512_and_si512(a, _mm512_set1_epi8(0x0F));
                    auto hi = to_vec(shift_right<4>(from_vec<std::uint8_t>(a)));
                    auto res = _mm256_or_si256(
                        to_vec(shift_left<4>(from_vec<std::uint8_t>(_mm512_permutexvar_epi8(lo, lookup)))),
                        _mm512_permutexvar_epi8(hi, lookup)
                    );
                    return from_vec<T>(res);
                } else {
                    auto res = rev_helper(rev_helper, rcast<mtype>(reverse_bits(rcast<std::uint8_t>(v))));
                    return rcast<T>(res);
                }
            }
            #endif

            return join(
                reverse_bits<false>(v.lo),
                reverse_bits<false>(v.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto reverse(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using mtype = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::reverse(v);
        } else {
            static constexpr auto bits = sizeof(v);
            if constexpr (bits == sizeof(__m128)) {
                auto a = to_vec(rcast<mtype>(v));
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr std::int8_t mask_rev_e8[16] = {
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0}; 
                    return rcast<T>(from_vec<mtype>(_mm_shuffle_epi8(a, *reinterpret_cast<__m128i const*>(mask_rev_e8))));
                } else if constexpr (sizeof(T) == 2) {
                    alignas(16) static constexpr std::int8_t mask_rev_e16[16] = {
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1
                    };
                    return rcast<T>(from_vec<mtype>(_mm_shuffle_epi8(a, *reinterpret_cast<__m128i const*>(mask_rev_e16))));
                } else if constexpr (sizeof(T) == 4) {
                    return rcast<T>(from_vec<mtype>(_mm_shuffle_epi32(
                        a, _MM_SHUFFLE(0, 1, 2, 3)
                    )));
                } else if constexpr (sizeof(T) == 8) {
                    return rcast<T>(from_vec<mtype>(_mm_shuffle_epi32(
                        a, _MM_SHUFFLE(1, 0, 3, 2)
                    )));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return reverse(
                    from_vec<T>(fit_to_vec(v))
                ).hi;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                auto a = to_vec(rcast<mtype>(v));
                if constexpr (sizeof(T) == 1) {
                    alignas(32) static constexpr std::int8_t mask_rev_e8[32] = {
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0,
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0
                    }; 
                    auto res = _mm256_shuffle_epi8(a, *reinterpret_cast<__m256i const*>(mask_rev_e8));
                    res = _mm256_permute2f128_si256(res, res, 0b01);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 2) {
                    alignas(32) static constexpr std::int8_t mask_rev_e16[32] = {
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1,
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1,
                    };
                    auto res = _mm256_shuffle_epi8(a, *reinterpret_cast<__m256i const*>(mask_rev_e16));
                    res = _mm256_permute2f128_si256(res, res, 0b01);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm256_shuffle_epi32(
                        a, _MM_SHUFFLE(0, 1, 2, 3)
                    );
                    res = _mm256_permute2f128_si256(res, res, 0b01);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm256_shuffle_epi32(
                        a, _MM_SHUFFLE(1, 0, 3, 2)
                    );
                    res = _mm256_permute2f128_si256(res, res, 0b01);
                    return rcast<T>(from_vec<mtype>(res));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(rcast<mtype>(v));
                if constexpr (sizeof(T) == 1) {
                    alignas(64) static constexpr std::int8_t mask_rev_e8[64] = {
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0,
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0,
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0,
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0
                    }; 
                    auto res = _mm512_shuffle_epi8(a, *reinterpret_cast<__m512i const*>(mask_rev_e8));
                    res = _mm512_shuffle_i64x2(res, res, 0b00011011);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 2) {
                    alignas(64) static constexpr std::int8_t mask_rev_e16[64] = {
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1,
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1,
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1,
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1
                    };
                    auto res = _mm512_shuffle_epi8(a, *reinterpret_cast<__m512i const*>(mask_rev_e16));
                    res = _mm512_shuffle_i64x2(res, res, 0b00011011);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto res = _mm_shuffle_epi32(
                        a, _MM_SHUFFLE(0, 1, 2, 3)
                    );
                    res = _mm512_shuffle_i64x2(res, res, 0b00011011);
                    return rcast<T>(from_vec<mtype>(res));
                } else if constexpr (sizeof(T) == 8) {
                    auto res = _mm_shuffle_epi32(
                        a, _MM_SHUFFLE(1, 0, 3, 2)
                    );
                    res = _mm512_shuffle_i64x2(res, res, 0b00011011);
                    return rcast<T>(from_vec<mtype>(res));
                }
            }
            #endif

            return join(
                reverse<false>(v.hi),
                reverse<false>(v.lo)
            );
        }
    }
// !MARK

// MARK: Zip
    namespace internal {
        struct zip_helper {
            template <std::size_t N, typename T>
                requires (N == 2)
            UI_ALWAYS_INLINE auto low(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept -> Vec<N, T> {
                if constexpr (sizeof(T) == 4) {
                    auto l = fit_to_vec(rcast<std::int32_t>(a)); 
                    auto r = fit_to_vec(rcast<std::int32_t>(a)); 
                    return rcast<T>(from_vec<T>(_mm_unpacklo_epi32(l, r)).lo.lo);
                } else if constexpr (sizeof(T) == 8) {
                    auto l = to_vec(rcast<std::int64_t>(a)); 
                    auto r = to_vec(rcast<std::int64_t>(b)); 
                    return rcast<T>(from_vec<std::int64_t>(_mm_unpacklo_epi64(l, r)));
                } 
                return { a[0], b[0] };
            }

            template <std::size_t N, typename T>
                requires (N == 2)
            UI_ALWAYS_INLINE auto high(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept -> Vec<N, T> {
                if constexpr (sizeof(T) == 4) {
                    auto l = fit_to_vec(rcast<std::int32_t>(a)); 
                    auto r = fit_to_vec(rcast<std::int32_t>(a)); 
                    l = _mm_shuffle_epi32(l, _MM_SHUFFLE(1, 1, 0, 0));
                    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 1, 0, 0));
                    return rcast<T>(from_vec<T>(_mm_unpackhi_epi32(l, r)).hi.lo);
                } else if constexpr (sizeof(T) == 8) {
                    auto l = to_vec(rcast<std::int64_t>(a)); 
                    auto r = to_vec(rcast<std::int64_t>(b)); 
                    return rcast<T>(from_vec<std::int64_t>(_mm_unpackhi_epi64(l, r)));
                } 
                return { a[1], b[1] };
            }

            template <std::size_t N, typename T>
                requires (N > 2)
            UI_ALWAYS_INLINE auto low(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept {
                static constexpr auto bits = sizeof(a);
                if constexpr (bits == sizeof(__m128)) {
                    if constexpr (sizeof(T) == 1) {
                        auto l = to_vec(rcast<std::int8_t>(a));
                        auto r = to_vec(rcast<std::int8_t>(b));
                        return rcast<T>(from_vec<std::int8_t>(_mm_unpacklo_epi8(l, r)));
                    } else if constexpr (sizeof(T) == 2) {
                        auto l = to_vec(rcast<std::int16_t>(a));
                        auto r = to_vec(rcast<std::int16_t>(b));
                        return rcast<T>(from_vec<std::int16_t>(_mm_unpacklo_epi16(l, r)));
                    } else if constexpr (sizeof(T) == 4) {
                        auto l = to_vec(rcast<std::int32_t>(a));
                        auto r = to_vec(rcast<std::int32_t>(b));
                        return rcast<T>(from_vec<std::int32_t>(_mm_unpacklo_epi32(l, r)));
                    } else if constexpr (sizeof(T) == 8) {
                        auto l = to_vec(rcast<std::int64_t>(a));
                        auto r = to_vec(rcast<std::int64_t>(b));
                        return rcast<T>(from_vec<std::int64_t>(_mm_unpacklo_epi64(l, r)));
                    }
                } else if constexpr (bits * 2 == sizeof(__m128)) {
                    return low(
                        join(a, b),
                        Vec<2 * N, T>{}
                    ).lo;
                }
                // TODO: Implement avx256 and 512
            }

            template <std::size_t N, typename T>
                requires (N > 2)
            UI_ALWAYS_INLINE auto high(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept {
                static constexpr auto bits = sizeof(a);
                if constexpr (bits == sizeof(__m128)) {
                    if constexpr (sizeof(T) == 1) {
                        auto l = to_vec(rcast<std::int8_t>(a));
                        auto r = to_vec(rcast<std::int8_t>(b));
                        return rcast<T>(from_vec<std::int8_t>(_mm_unpackhi_epi8(l, r)));
                    } else if constexpr (sizeof(T) == 2) {
                        auto l = to_vec(rcast<std::int16_t>(a));
                        auto r = to_vec(rcast<std::int16_t>(b));
                        return rcast<T>(from_vec<std::int16_t>(_mm_unpackhi_epi16(l, r)));
                    } else if constexpr (sizeof(T) == 4) {
                        auto l = to_vec(rcast<std::int32_t>(a));
                        auto r = to_vec(rcast<std::int32_t>(b));
                        return rcast<T>(from_vec<std::int32_t>(_mm_unpackhi_epi32(l, r)));
                    } else if constexpr (sizeof(T) == 8) {
                        auto l = to_vec(rcast<std::int64_t>(a));
                        auto r = to_vec(rcast<std::int64_t>(b));
                        return rcast<T>(from_vec<std::int64_t>(_mm_unpackhi_epi64(l, r)));
                    }
                } else if constexpr (bits * 2 == sizeof(__m128)) {
                    auto l = fit_to_vec(a);
                    auto r = fit_to_vec(b);

                    if constexpr (sizeof(T) == 1) {
                        l = _mm_shuffle_epi32(l, _MM_SHUFFLE(1, 1, 1, 1));
                        r = _mm_shuffle_epi32(r, _MM_SHUFFLE(1, 1, 1, 1));
                        return low(from_vec<T>(l), from_vec<T>(r)).lo;
                    } else if constexpr (sizeof(T) == 2) {
                        auto mask = _mm_setr_epi8(
                            0, 0, // Zero out first 16 bits
                            0, 0, // Zero out second 16 bits
                            0, 1,   // Move X0
                            2, 3,   // Move X1

                            4, 5,   // Keep X2
                            6, 7,   // Keep X3
                            0, 0, // Zero out last 16 bits
                            0, 0  // Zero out last 16 bits
                        );
                        l = _mm_shuffle_epi8(l, mask);
                        r = _mm_shuffle_epi8(r, mask);
                        return high(from_vec<T>(l), from_vec<T>(r)).hi;
                    }
                }
            }
        };

        template <std::size_t N, typename T>
        UI_ALWAYS_INLINE auto zipping_helper(
            Vec<N, T> const& a,
            Vec<N, T> const& b
        ) noexcept -> Vec<2 * N, T> {
            auto const zip = internal::zip_helper{};
            if constexpr (!(
                std::is_void_v<decltype(zip.low(a,b))> ||
                std::is_void_v<decltype(zip.high(a,b))>
            )) {
                return join(zip.low(a, b), zip.high(a, b));
            } else {
                using ret_low_t = decltype(zip.low(a.lo, b.lo));
                if constexpr (std::is_void_v<ret_low_t>) {
                    return join(zipping_helper(a.lo, b.lo), zipping_helper(a.hi, b.hi));
                } else {
                    using ret_high_t = decltype(zip.high(a.hi, b.hi));
                    if constexpr (std::is_void_v<ret_high_t>) {
                        return join(zip.low(a.lo, b.lo), zipping_helper(a.hi, b.hi));
                    } else {
                        return join(
                            join(zip.low(a.lo, b.lo), zip.high(a.lo, b.lo)),
                            join(zip.low(a.hi, b.hi), zip.high(a.hi, b.hi))
                        );
                    }
                }
            }
        }
    } // namespace internal

    /*
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_low(a, b) == Vec<4, int>::load(0, 4, 1, 5))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        auto const zip = internal::zip_helper{};
        using ret_low_t = decltype(zip.low(a, b));
        if constexpr (!std::is_void_v<ret_low_t>) {
            return internal::zip_helper{}.low(a, b);
        } else {
            return internal::zipping_helper(a.lo, b.lo);
        }
    }

    /**
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_low(a, b) == Vec<4, int>::load(2, 6, 3, 7))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        auto const zip = internal::zip_helper{};
        using ret_low_t = decltype(zip.high(a, b));
        if constexpr (!std::is_void_v<ret_low_t>) {
            return internal::zip_helper{}.high(a, b);
        } else {
            return internal::zipping_helper(a.hi, b.hi);
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto unzip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::unzip_low(a, b); 
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int8_t>(res));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = _mm_set_epi8(
                        13, 12, 9, 8, 5, 4, 1, 0,
                        13, 12, 9, 8, 5, 4, 1, 0 
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int32_t>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = _mm_shuffle_epi32(l, _MM_SHUFFLE(2, 0, 2, 0));
                    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(2, 0, 2, 0));
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int64_t>(res));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                auto tmp = unzip_low(from_vec<T>(l), from_vec<T>(r)); // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
                return join(tmp.lo.lo, tmp.hi.lo);
            }
            // TODO: implement avx256 and 512
            return join(unzip_low(a.lo, a.hi), unzip_low(b.lo, b.hi));
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto unzip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::unzip_high(a, b); 
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = _mm_set_epi8(
                         15, 13, 11, 9, 7, 5, 3, 1,
                         15, 13, 11, 9, 7, 5, 3, 1
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int8_t>(res));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = _mm_set_epi8(
                         15, 14, 11, 10, 7, 6, 3, 2,
                         15, 14, 11, 10, 7, 6, 3, 2
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int16_t>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = _mm_shuffle_epi32(l, _MM_SHUFFLE(3, 1, 3, 1));
                    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 1, 3, 1));
                    auto res = _mm_castpd_si128(_mm_blend_pd(
                        _mm_castsi128_pd(l),
                        _mm_castsi128_pd(r),
                        0b10
                    ));
                    return rcast<T>(from_vec<std::int32_t>(res));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                auto tmp = unzip_high(from_vec<T>(l), from_vec<T>(r)); // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
                return join(tmp.lo.lo, tmp.hi.lo);
            }

            // TODO: implement avx256 and 512
            return join(unzip_high(a.lo, a.hi), unzip_high(b.lo, b.hi));
        }
    }
// !MARK

// MARK: Transpose elements
    template <bool Merge = true, std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto transpose_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::transpose_low(a, b); 
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    return rcast<T>(from_vec<std::int8_t>(_mm_unpacklo_epi8(l, r)));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = _mm_set_epi8(
                        13, 12, 9, 8, 5, 4, 1, 0,
                        13, 12, 9, 8, 5, 4, 1, 0 
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    return rcast<T>(from_vec<std::int32_t>(_mm_unpacklo_epi16(l, r)));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = _mm_shuffle_epi32(l, _MM_SHUFFLE(3, 1, 2, 0));
                    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 1, 2, 0));
                    return rcast<T>(from_vec<std::int64_t>(_mm_unpacklo_epi32(l, r)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                return transpose_low(from_vec<T>(l), from_vec<T>(r)).lo; // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
            }
            // TODO: implement avx256/512
            return join(
                transpose_low<false>(a.lo, b.lo),
                transpose_low<false>(a.hi, b.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto transpose_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::transpose_high(a, b); 
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = _mm_set_epi8(
                         15, 13, 11, 9, 7, 5, 3, 1,
                         15, 13, 11, 9, 7, 5, 3, 1
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    return rcast<T>(from_vec<std::int8_t>(_mm_unpacklo_epi8(l, r)));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = _mm_set_epi8(
                         15, 14, 11, 10, 7, 6, 3, 2,
                         15, 14, 11, 10, 7, 6, 3, 2
                    );
                    l = _mm_shuffle_epi8(l, mask);
                    r = _mm_shuffle_epi8(r, mask);
                    return rcast<T>(from_vec<std::int32_t>(_mm_unpacklo_epi16(l, r)));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = _mm_shuffle_epi32(l, _MM_SHUFFLE(3, 1, 3, 1));
                    r = _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 1, 3, 1));
                    return rcast<T>(from_vec<std::int64_t>(_mm_unpacklo_epi32(l, r)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                return transpose_high(from_vec<T>(l), from_vec<T>(r)).lo; // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
            }

            // TODO: implement avx256/512
            return join(transpose_high(a.lo, b.lo), transpose_high(a.hi, b.hi));
        }
    }
// !MARK
} // namespace ui::x86

namespace ui {
// MARK: IntMask
    template <std::size_t N, typename T>
    inline constexpr IntMask<N, T>::IntMask(mask_t<N, T> const& m) noexcept {
        using namespace x86;

        using mtype = mask_inner_t<T>;
        if constexpr (is_packed) {
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 16) {
                    mask = static_cast<base_type>(_mm_movemask_epi8(to_vec(m)));
                    return;
                }

                if constexpr (N == 32) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    mask = static_cast<base_type>(_mm256_movemask_epi8(to_vec(m)));
                    return;
                    #endif
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 4) {
                    mask = static_cast<base_type>(_mm_movemask_ps(std::bit_cast<__m128>(m)));
                    return;
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 4) {
                    mask = static_cast<base_type>(_mm_movemask_pd(std::bit_cast<__m128d>(m)));
                    return;
                }
            }

            auto ext = rcast<mtype>(shift_right<7>(rcast<std::make_signed_t<T>>(m)));
            auto helper = [&ext]<std::size_t... Is>(std::index_sequence<Is...>) -> base_type {
                auto res = base_type{};
                ((res |= (base_type((ext[Is] & 1) << Is))),...);
                return res;
            };
            mask = helper(std::make_index_sequence<N>{});
        } else {
            auto tmp = rcast<std::uint16_t>(m);
            auto s = narrowing_shift_right<4>(tmp);
            mask = std::bit_cast<base_type>(rcast<std::uint64_t>(s));
        }
    }

// !MARK
} // namespace ui


#endif // AMT_UI_ARCH_X86_MANIP_HPP 
