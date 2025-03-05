#ifndef AMT_UI_ARCH_X86_BIT_HPP
#define AMT_UI_ARCH_X86_BIT_HPP

#include "cast.hpp"
#include "add.hpp"
#include "sub.hpp"
#include "logical.hpp"
#include "../emul/bit.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto popcount(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T>;

// MARK: Count leading zeros
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_zeros(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::count_leading_zeros(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_lzcnt_epi8(a));
                    #else
                    alignas(16) static constexpr std::int8_t mask_CLZ[16] = { /* 0 */ 4,/* 1 */ 3,/* 2 */ 2,/* 3 */ 2,
                                                         /* 4 */ 1,/* 5 */ 1,/* 6 */ 1,/* 7 */ 1,
                                                         /* 8 */ 0,/* 9 */ 0,/* a */ 0,/* b */ 0,
                                                         /* c */ 0,/* d */ 0,/* e */ 0,/* f */ 0 };
                    auto mask_low = _mm_set1_epi8(0x0f);
                    auto fs = _mm_set1_epi8(4);
                    auto low_clz = _mm_shuffle_epi8(*reinterpret_cast<__m128i const*>(mask_CLZ), a);
                    auto mask = _mm_srli_epi16(a, 4);
                    mask = _mm_and_si128(mask, mask_low);
                    auto high_clz = _mm_shuffle_epi8(*reinterpret_cast<__m128i const*>(mask_CLZ), mask);
                    mask = _mm_cmpeq_epi8(high_clz, fs);
                    low_clz = _mm_and_si128(low_clz, mask);
                    return from_vec<T>(_mm_add_epi8(low_clz, high_clz));
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_lzcnt_epi16(a));
                    #else
                    alignas(16) static constexpr std::int8_t mask8_sab[16] = { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
                    alignas(16) static constexpr std::uint16_t mask8_bit[8] = {0x00ff, 0x00ff, 0x00ff, 0x00ff,0x00ff, 0x00ff, 0x00ff, 0x00ff};

                    auto mb = *reinterpret_cast<__m128i const*>(mask8_bit);
                    auto c7 = _mm_srli_epi16(mb, 5);
                    auto res8x16 = to_vec(
                        std::bit_cast<Vec<N, T>>(count_leading_zeros(std::bit_cast<Vec<2 * N, std::int8_t>>(from_vec<T>(a))))
                    );
                    auto res8x16_swap = _mm_shuffle_epi8(res8x16, *reinterpret_cast<__m128i const*>(mask8_sab));
                    res8x16 = _mm_and_si128(res8x16, mb);

                    res8x16_swap = _mm_and_si128(res8x16_swap, mb);
                    c7 = _mm_cmpgt_epi16(res8x16_swap, c7);
                    res8x16 = _mm_and_si128(res8x16, c7);
                    return from_vec<T>(_mm_add_epi16(res8x16_swap, res8x16));
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_lzcnt_epi32(a));
                    #else
                    auto tmp = _mm_srli_epi32(a, 1);
                    auto res = _mm_or_si128(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = _mm_srli_epi32(res, 2);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = _mm_srli_epi32(res, 4);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = _mm_srli_epi32(res, 8);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = _mm_srli_epi32(res, 16);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_lzcnt_epi64(a));
                    #else
                    auto tmp = _mm_srli_epi32(a, 1);
                    auto res = _mm_or_si128(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = _mm_srli_epi64(res, 2);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = _mm_srli_epi64(res, 4);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = _mm_srli_epi64(res, 8);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = _mm_srli_epi64(res, 16);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    tmp = _mm_srli_epi64(res, 16);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    tmp = _mm_srli_epi64(res, 32);
                    res = _mm_or_si128(tmp, res); //atmp[i] |= (atmp[i] >> 32);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                    #endif
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                return count_leading_zeros(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm256_lzcnt_epi8(a));
                    #else
                    alignas(16) static constexpr std::int8_t mask_CLZ[32] = {
                        /* 0 */ 4,/* 1 */ 3,/* 2 */ 2,/* 3 */ 2,
                        /* 4 */ 1,/* 5 */ 1,/* 6 */ 1,/* 7 */ 1,
                        /* 8 */ 0,/* 9 */ 0,/* a */ 0,/* b */ 0,
                        /* c */ 0,/* d */ 0,/* e */ 0,/* f */ 0,

                        /* 0 */ 4,/* 1 */ 3,/* 2 */ 2,/* 3 */ 2,
                        /* 4 */ 1,/* 5 */ 1,/* 6 */ 1,/* 7 */ 1,
                        /* 8 */ 0,/* 9 */ 0,/* a */ 0,/* b */ 0,
                        /* c */ 0,/* d */ 0,/* e */ 0,/* f */ 0
                    };
                    auto mask_low = _mm256_set1_epi8(0x0f);
                    auto fs = _mm256_set1_epi8(4);
                    auto low_clz = _mm256_shuffle_epi8(*reinterpret_cast<__m256i const*>(mask_CLZ), a);
                    auto mask = _mm256_srli_epi16(a, 4);
                    mask = _mm256_and_si256(mask, mask_low);
                    auto high_clz = _mm256_shuffle_epi8(*reinterpret_cast<__m256i const*>(mask_CLZ), mask);
                    mask = _mm256_cmpeq_epi8(high_clz, fs);
                    low_clz = _mm256_and_si256(low_clz, mask);
                    return from_vec<T>(_mm256_add_epi8(low_clz, high_clz));
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm256_lzcnt_epi16(a));
                    #else
                    alignas(32) static constexpr std::int8_t mask8_sab256[32] = { 
                        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
                    };
                    alignas(32) static constexpr std::uint16_t mask8_bit256[16] = { 
                        0x00ff, 0x00ff, 0x00ff, 0x00ff,
                        0x00ff, 0x00ff, 0x00ff, 0x00ff,
                        0x00ff, 0x00ff, 0x00ff, 0x00ff,
                        0x00ff, 0x00ff, 0x00ff, 0x00ff
                    };

                    auto mb = *reinterpret_cast<const __m256i*>(mask8_bit256);
                    auto c7 = _mm256_srli_epi16(mb, 5);
                    auto res8x16 = to_vec(
                        std::bit_cast<Vec<N, T>>(count_leading_zeros(std::bit_cast<Vec<2 * N, std::int8_t>>(from_vec<T>(a))))
                    );
                    auto res8x16_swap = _mm256_shuffle_epi8(res8x16, *reinterpret_cast<const __m256i*>(mask8_sab256));
                    res8x16 = _mm256_and_si256(res8x16, mb);

                    res8x16_swap = _mm256_and_si256(res8x16_swap, mb);
                    c7 = _mm256_cmpgt_epi16(res8x16_swap, c7);
                    res8x16 = _mm256_and_si256(res8x16, c7);
                    return from_vec<T>(_mm256_add_epi16(res8x16_swap, res8x16));
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm256_lzcnt_epi32(a));
                    #else
                    auto tmp = _mm256_srli_epi32(a, 1);
                    auto res = _mm256_or_si256(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = _mm256_srli_epi32(res, 2);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = _mm256_srli_epi32(res, 4);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = _mm256_srli_epi32(res, 8);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = _mm256_srli_epi32(res, 16);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm256_lzcnt_epi64(a));
                    #else
                    auto tmp = _mm256_srli_epi32(a, 1);
                    auto res = _mm256_or_si256(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = _mm256_srli_epi64(res, 2);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = _mm256_srli_epi64(res, 4);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = _mm256_srli_epi64(res, 8);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = _mm256_srli_epi64(res, 16);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    tmp = _mm256_srli_epi64(res, 16);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    tmp = _mm256_srli_epi64(res, 32);
                    res = _mm256_or_si256(tmp, res); //atmp[i] |= (atmp[i] >> 32);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                    #endif
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                   return from_vec<T>(_mm512_lzcnt_epi8(a)); 
                } else if constexpr (sizeof(T) == 2) {
                   return from_vec<T>(_mm512_lzcnt_epi16(a)); 
                } else if constexpr (sizeof(T) == 4) {
                   return from_vec<T>(_mm512_lzcnt_epi32(a)); 
                } else if constexpr (sizeof(T) == 8) {
                   return from_vec<T>(_mm512_lzcnt_epi64(a)); 
                }
            }
            #endif

            return join(
                count_leading_zeros(v.lo),
                count_leading_zeros(v.hi)
            );
        }
    }
// !MARK

// MARK: Count leading sign bits
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_sign_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        constexpr auto fn = [](auto const& v_) {
            auto ffs = rcast<T>(cmp(v_, v_, op::equal_t{}));
            auto sign = Vec<N, T>::load(T(1) << (sizeof(T) * 8 - 1));
            auto ones = Vec<N, T>::load(1);
            auto mask = bitwise_and(v_, sign);
            mask = rcast<T>(cmp(mask, sign, op::equal_t{})); 
            auto neg = bitwise_xor(v_, ffs);
            neg = bitwise_and(neg, mask);
            auto pos = bitwise_notand(mask, v_);
            auto comb = bitwise_or(pos, neg);
            comb = count_leading_zeros(comb);
            return sub(comb, ones);
        };

        if constexpr (bits == sizeof(__m128)) {
            return fn(v);
        } else if constexpr (bits * 2 == sizeof(__m128)) {
            return count_leading_sign_bits(from_vec<T>(fit_to_vec(v))).lo;
        }

        return fn(v);
    }
// !MARK

// MARK: Population Count
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto popcount(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::popcount(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi8(a));
                    #else
                    alignas(16) static constexpr std::int8_t mask_POPCOUNT[16] = 
                        { /* 0 */ 0,/* 1 */ 1,/* 2 */ 1,/* 3 */ 2,
                          /* 4 */ 1,/* 5 */ 2,/* 6 */ 2,/* 7 */ 3,
                          /* 8 */ 1,/* 9 */ 2,/* a */ 2,/* b */ 3,
                          /* c */ 2,/* d */ 3,/* e */ 3,/* f */ 4};

                    auto mask_low = _mm_set1_epi8(0x0f);
                    auto mask = _mm_and_si128(a, mask_low);
                    auto mp = *reinterpret_cast<__m128i const*>(mask_POPCOUNT);
                    auto lowpopcnt = _mm_shuffle_epi8(mp, mask);
                    mask = _mm_srli_epi16(a, 4);
                    mask = _mm_and_si128(mask, mask_low);
                    auto hipopcnt = _mm_shuffle_epi8(mp, mask);
                    return from_vec<T>(_mm_add_epi8(lowpopcnt, hipopcnt));
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi16(a));
                    #else
                    auto mask1 = _mm_set1_epi16(0x5555);  // 0b0101010101010101
                    auto mask2 = _mm_set1_epi16(0x3333);  // 0b0011001100110011
                    auto mask4 = _mm_set1_epi16(0x0F0F);  // 0b0000111100001111

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm_srli_epi16(a, 1);
                    tmp = _mm_and_si128(tmp, mask1);
                    a = _mm_sub_epi16(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm_srli_epi16(a, 2);
                    tmp = _mm_and_si128(tmp, mask2);
                    a = _mm_and_si128(a, mask2);
                    a = _mm_add_epi16(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm_srli_epi16(a, 4);
                    a = _mm_add_epi16(a, tmp);
                    a = _mm_and_si128(a, mask4);

                    tmp = _mm_srli_epi16(a, 8);
                    a = _mm_add_epi16(a, tmp);
                    a = _mm_and_si128(a, _mm_set1_epi16(0x001F)); // Only need 5 bits per lane

                    return from_vec<T>(a);
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi32(a));
                    #else
                    auto mask1 = _mm_set1_epi32(0x55555555);
                    auto mask2 = _mm_set1_epi32(0x33333333);
                    auto mask4 = _mm_set1_epi32(0x0F0F0F0F);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm_srli_epi32(a, 1);
                    tmp = _mm_and_si128(tmp, mask1);
                    a = _mm_sub_epi16(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm_srli_epi32(a, 2);
                    tmp = _mm_and_si128(tmp, mask2);
                    a = _mm_and_si128(a, mask2);
                    a = _mm_add_epi16(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm_srli_epi32(a, 4);
                    a = _mm_add_epi16(a, tmp);
                    a = _mm_and_si128(a, mask4);

                    // Sum counts in 16-bit halves
                    a = _mm_add_epi32(a, _mm_srli_epi32(a, 8));
                    // Sum counts in 32-bit value
                    a = _mm_add_epi32(a, _mm_srli_epi32(a, 16));
                    a = _mm_and_si128(a, _mm_set1_epi32(0x003F)); // Only need 5 bits per lane
                    return from_vec<T>(a);
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi64(a));
                    #else
                    auto mask1 = _mm_set1_epi64x(0x5555'5555'5555'5555ll);
                    auto mask2 = _mm_set1_epi64x(0x3333'3333'3333'3333ll);
                    auto mask4 = _mm_set1_epi64x(0x0F0F'0F0F'0F0F'0F0Fll);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm_srli_epi64(a, 1);
                    tmp = _mm_and_si128(tmp, mask1);
                    a = _mm_sub_epi64(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm_srli_epi64(a, 2);
                    tmp = _mm_and_si128(tmp, mask2);
                    a = _mm_and_si128(a, mask2);
                    a = _mm_add_epi64(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm_srli_epi64(a, 4);
                    a = _mm_add_epi64(a, tmp);
                    a = _mm_and_si128(a, mask4);

                    // Sum counts in 16-bit chunk
                    a = _mm_add_epi64(a, _mm_srli_epi64(a, 8));
                    // Sum counts in 32-bit chunk
                    a = _mm_add_epi64(a, _mm_srli_epi64(a, 16));
                    // Sum counts in 64-bit value
                    a = _mm_add_epi64(a, _mm_srli_epi64(a, 32));

                    a = _mm_and_si128(a, _mm_set1_epi64x(0x007F)); // Only need 7 bits per lane
                    return from_vec<T>(a);
                    #endif
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                return popcount(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi8(a));
                    #else
                    alignas(32) static constexpr std::int8_t mask_POPCOUNT[32] = 
                        { /* 0 */ 0,/* 1 */ 1,/* 2 */ 1,/* 3 */ 2,
                          /* 4 */ 1,/* 5 */ 2,/* 6 */ 2,/* 7 */ 3,
                          /* 8 */ 1,/* 9 */ 2,/* a */ 2,/* b */ 3,
                          /* c */ 2,/* d */ 3,/* e */ 3,/* f */ 4,

                          /* 0 */ 0,/* 1 */ 1,/* 2 */ 1,/* 3 */ 2,
                          /* 4 */ 1,/* 5 */ 2,/* 6 */ 2,/* 7 */ 3,
                          /* 8 */ 1,/* 9 */ 2,/* a */ 2,/* b */ 3,
                          /* c */ 2,/* d */ 3,/* e */ 3,/* f */ 4 };

                    auto mask_low = _mm256_set1_epi8(0x0f);
                    auto mask = _mm256_and_si256(a, mask_low);
                    auto mp = *reinterpret_cast<const __m256i*>(mask_POPCOUNT);
                    auto lowpopcnt = _mm256_shuffle_epi8(mp, mask);
                    mask = _mm256_srli_epi16(a, 4);
                    mask = _mm256_and_si256(mask, mask_low);
                    auto hipopcnt = _mm256_shuffle_epi8(mp, mask);
                    return from_vec<T>(_mm256_add_epi8(lowpopcnt, hipopcnt));
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi16(a));
                    #else
                    auto mask1 = _mm256_set1_epi16(0x5555);  // 0b0101010101010101
                    auto mask2 = _mm256_set1_epi16(0x3333);  // 0b0011001100110011
                    auto mask4 = _mm256_set1_epi16(0x0F0F);  // 0b0000111100001111

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm256_srli_epi16(a, 1);
                    tmp = _mm256_and_si256(tmp, mask1);
                    a = _mm256_sub_epi16(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm256_srli_epi16(a, 2);
                    tmp = _mm256_and_si256(tmp, mask2);
                    a = _mm256_and_si256(a, mask2);
                    a = _mm256_add_epi16(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm256_srli_epi16(a, 4);
                    a = _mm256_add_epi16(a, tmp);
                    a = _mm256_and_si256(a, mask4);

                    tmp = _mm256_srli_epi16(a, 8);
                    a = _mm256_add_epi16(a, tmp);
                    a = _mm256_and_si256(a, _mm256_set1_epi16(0x001F)); // Only need 5 bits per lane

                    return from_vec<T>(a);
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi32(a));
                    #else
                    auto mask1 = _mm256_set1_epi32(0x55555555);
                    auto mask2 = _mm256_set1_epi32(0x33333333);
                    auto mask4 = _mm256_set1_epi32(0x0F0F0F0F);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm256_srli_epi32(a, 1);
                    tmp = _mm256_and_si256(tmp, mask1);
                    a = _mm256_sub_epi32(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm256_srli_epi32(a, 2);
                    tmp = _mm256_and_si256(tmp, mask2);
                    a = _mm256_and_si256(a, mask2);
                    a = _mm256_add_epi32(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm256_srli_epi32(a, 4);
                    a = _mm256_add_epi32(a, tmp);
                    a = _mm256_and_si256(a, mask4);

                    // Sum counts in 16-bit halves
                    a = _mm256_add_epi32(a, _mm256_srli_epi32(a, 8));
                    // Sum counts in 32-bit value
                    a = _mm256_add_epi32(a, _mm256_srli_epi32(a, 16));
                    a = _mm256_and_si256(a, _mm256_set1_epi32(0x003F)); // Only need 5 bits per lane
                    return from_vec<T>(a);
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return from_vec<T>(_mm_popcnt_epi64(a));
                    #else
                    auto mask1 = _mm256_set1_epi64x(0x5555'5555'5555'5555ll);
                    auto mask2 = _mm256_set1_epi64x(0x3333'3333'3333'3333ll);
                    auto mask4 = _mm256_set1_epi64x(0x0F0F'0F0F'0F0F'0F0Fll);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = _mm256_srli_epi64(a, 1);
                    tmp = _mm256_and_si256(tmp, mask1);
                    a = _mm256_sub_epi64(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = _mm256_srli_epi64(a, 2);
                    tmp = _mm256_and_si256(tmp, mask2);
                    a = _mm256_and_si256(a, mask2);
                    a = _mm256_add_epi64(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = _mm256_srli_epi64(a, 4);
                    a = _mm256_add_epi64(a, tmp);
                    a = _mm256_and_si256(a, mask4);

                    // Sum counts in 16-bit chunk
                    a = _mm256_add_epi64(a, _mm256_srli_epi64(a, 8));
                    // Sum counts in 32-bit chunk
                    a = _mm256_add_epi64(a, _mm256_srli_epi64(a, 16));
                    // Sum counts in 64-bit chunk
                    a = _mm256_add_epi64(a, _mm256_srli_epi64(a, 32));

                    a = _mm256_and_si256(a, _mm256_set1_epi64x(0x007F));
                    return from_vec<T>(a);
                    #endif
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                   return from_vec<T>(_mm512_popcnt_epi8(a)); 
                } else if constexpr (sizeof(T) == 2) {
                   return from_vec<T>(_mm512_popcnt_epi16(a)); 
                } else if constexpr (sizeof(T) == 4) {
                   return from_vec<T>(_mm512_popcnt_epi32(a)); 
                } else if constexpr (sizeof(T) == 8) {
                   return from_vec<T>(_mm512_popcnt_epi64(a)); 
                }
            }
            #endif

            return join(
                popcount(v.lo),
                popcount(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise clear
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_clear(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return bitwise_notand(b, a);
    }
// !MARK

// MARK: Bitwise select
    template <std::size_t N, typename T>
        requires (std::is_arithmetic_v<T>)
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(cond);
        if constexpr (N == 1) {
            return emul::bitwise_select(cond, true_, false_);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                auto c = to_vec(cond);
                auto t = to_vec(true_);
                auto f = to_vec(false_);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_blendv_ps(f, t, _mm_castsi128_ps(c)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_blendv_ps(f, t, _mm_castsi128_pd(c)));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<T>(_mm_blendv_epi8(f, t, c));
                } else {
                    auto t0 = bitwise_and(cond, rcast<mask_inner_t<T>>(true_));
                    auto t1 = bitwise_notand(cond, rcast<mask_inner_t<T>>(false_));
                    return rcast<T>(bitwise_or(t0, t1));
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                return bitwise_select(
                    from_vec<mask_inner_t<T>>(fit_to_vec(cond)),
                    from_vec<T>(fit_to_vec(true_)),
                    from_vec<T>(fit_to_vec(false_))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                auto c = to_vec(cond);
                auto t = to_vec(true_);
                auto f = to_vec(false_);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_blendv_ps(f, t, _mm_castsi128_ps(c)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_blendv_ps(f, t, _mm_castsi128_pd(c)));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<T>(_mm256_blendv_epi8(f, t, c));
                } else {
                    auto t0 = bitwise_and(cond, rcast<mask_inner_t<T>>(true_));
                    auto t1 = bitwise_notand(cond, rcast<mask_inner_t<T>>(false_));
                    return rcast<T>(bitwise_or(t0, t1));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto t0 = bitwise_and(cond, rcast<mask_inner_t<T>>(true_));
                auto t1 = bitwise_notand(cond, rcast<mask_inner_t<T>>(false_));
                return rcast<T>(bitwise_or(t0, t1));
            }
            #endif

            return join(
                bitwise_select(cond.lo, true_.lo, false_.lo),
                bitwise_select(cond.hi, true_.hi, false_.hi)
            );
        }
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_BIT_HPP
