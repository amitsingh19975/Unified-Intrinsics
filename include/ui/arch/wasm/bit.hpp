#ifndef AMT_UI_ARCH_WASM_BIT_HPP
#define AMT_UI_ARCH_WASM_BIT_HPP

#include "cast.hpp"
#include "../emul/bit.hpp"
#include "logical.hpp"
#include "add.hpp"
#include "sub.hpp"
#include <wasm_simd128.h>

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto popcount(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T>;

// MARK: Count leading zeros
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_zeros(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::count_leading_zeros(v);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr std::int8_t mask_CLZ[16] = {
                        /* 0 */ 4,/* 1 */ 3,/* 2 */ 2,/* 3 */ 2,
                        /* 4 */ 1,/* 5 */ 1,/* 6 */ 1,/* 7 */ 1,
                        /* 8 */ 0,/* 9 */ 0,/* a */ 0,/* b */ 0,
                        /* c */ 0,/* d */ 0,/* e */ 0,/* f */ 0
                    };
                    auto mask_low = wasm_i8x16_const_splat(0x0f);
                    auto fs = wasm_i8x16_const_splat(4);
                    auto lookup = *reinterpret_cast<v128_t const*>(mask_CLZ);
                    auto low_clz = wasm_i8x16_swizzle(lookup, a);
                    auto mask = wasm_u16x8_shr(a, 4);
                    mask = wasm_v128_and(mask, mask_low);
                    auto high_clz = wasm_i8x16_swizzle(lookup, mask);
                    mask = wasm_i8x16_eq(high_clz, fs);
                    low_clz = wasm_v128_and(low_clz, mask);
                    return from_vec<T>(wasm_i8x16_add(low_clz, high_clz));
                } else if constexpr (sizeof(T) == 2) {
                    alignas(16) static constexpr std::int8_t mask8_sab[16] = {
                        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
                    };
                    alignas(16) static constexpr std::uint16_t mask8_bit[8] = {
                        0x00ff, 0x00ff, 0x00ff, 0x00ff,0x00ff, 0x00ff, 0x00ff, 0x00ff
                    };

                    auto mb = *reinterpret_cast<v128_t const*>(mask8_bit);
                    auto c7 = wasm_u16x8_shr(mb, 5);
                    auto res8x16 = to_vec(
                        std::bit_cast<Vec<N, T>>(count_leading_zeros(std::bit_cast<Vec<2 * N, std::int8_t>>(from_vec<T>(a))))
                    );
                    auto res8x16_swap = wasm_i8x16_swizzle(res8x16, *reinterpret_cast<v128_t const*>(mask8_sab));
                    res8x16 = wasm_v128_and(res8x16, mb);

                    res8x16_swap = wasm_v128_and(res8x16_swap, mb);
                    c7 = wasm_i16x8_ge(res8x16_swap, c7);
                    res8x16 = wasm_v128_and(res8x16, c7);
                    return from_vec<T>(wasm_i16x8_add(res8x16_swap, res8x16));
                } else if constexpr (sizeof(T) == 4) {
                    auto tmp = wasm_u32x4_shr(a, 1);
                    auto res = wasm_v128_or(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = wasm_u32x4_shr(res, 2);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = wasm_u32x4_shr(res, 4);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = wasm_u32x4_shr(res, 8);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = wasm_u32x4_shr(res, 16);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                } else if constexpr (sizeof(T) == 8) {
                    auto tmp = wasm_u32x4_shr(a, 1);
                    auto res = wasm_v128_or(tmp, a); //atmp[i] |= (atmp[i] >> 1);

                    tmp = wasm_u64x2_shr(res, 2);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 2);
                    tmp = wasm_u64x2_shr(res, 4);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 4);
                    tmp = wasm_u64x2_shr(res, 8);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 8);
                    tmp = wasm_u64x2_shr(res, 16);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 16);
                    tmp = wasm_u64x2_shr(res, 32);
                    res = wasm_v128_or(tmp, res); //atmp[i] |= (atmp[i] >> 32);
                    auto n = bitwise_not(from_vec<T>(res));
                    return popcount(n);
                }
            } else if constexpr (size * 2 == sizeof(v128_t)) {
               return count_leading_zeros(from_vec<T>(fit_to_vec(v))).lo; 
            }
            return join(
                count_leading_zeros<false>(v.lo),
                count_leading_zeros<false>(v.hi)
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
            auto sign = Vec<N, T>::load(static_cast<T>(T(1) << (sizeof(T) * 8 - 1)));
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

        if constexpr (bits == sizeof(v128_t)) {
            return fn(v);
        } else if constexpr (bits * 2 == sizeof(v128_t)) {
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
            if constexpr (bits == sizeof(v128_t)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    return from_vec<T>(wasm_i8x16_popcnt(a));
                } else if constexpr (sizeof(T) == 2) {
                    auto mask1 = wasm_i16x8_const_splat(0x5555);  // 0b0101010101010101
                    auto mask2 = wasm_i16x8_const_splat(0x3333);  // 0b0011001100110011
                    auto mask4 = wasm_i16x8_const_splat(0x0F0F);  // 0b0000111100001111

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = wasm_u16x8_shr(a, 1);
                    tmp = wasm_v128_and(tmp, mask1);
                    a = wasm_i16x8_sub(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = wasm_u16x8_shr(a, 2);
                    tmp = wasm_v128_and(tmp, mask2);
                    a = wasm_v128_and(a, mask2);
                    a = wasm_i16x8_add(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = wasm_u16x8_shr(a, 4);
                    a = wasm_i16x8_add(a, tmp);
                    a = wasm_v128_and(a, mask4);

                    tmp = wasm_u16x8_shr(a, 8);
                    a = wasm_i16x8_add(a, tmp);
                    a = wasm_v128_and(a, wasm_i16x8_const_splat(0x001F)); // Only need 5 bits per lane

                    return from_vec<T>(a);
                } else if constexpr (sizeof(T) == 4) {
                    auto mask1 = wasm_i32x4_const_splat(0x55555555);
                    auto mask2 = wasm_i32x4_const_splat(0x33333333);
                    auto mask4 = wasm_i32x4_const_splat(0x0F0F0F0F);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = wasm_u32x4_shr(a, 1);
                    tmp = wasm_v128_and(tmp, mask1);
                    a = wasm_i32x4_sub(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = wasm_u32x4_shr(a, 2);
                    tmp = wasm_v128_and(tmp, mask2);
                    a = wasm_v128_and(a, mask2);
                    a = wasm_i32x4_add(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = wasm_u32x4_shr(a, 4);
                    a = wasm_i32x4_add(a, tmp);
                    a = wasm_v128_and(a, mask4);

                    // Sum counts in 16-bit halves
                    a = wasm_i32x4_add(a, wasm_u32x4_shr(a, 8));
                    // Sum counts in 32-bit value
                    a = wasm_i32x4_add(a, wasm_u32x4_shr(a, 16));
                    a = wasm_v128_and(a, wasm_i32x4_const_splat(0x003F)); // Only need 5 bits per lane
                    return from_vec<T>(a);
                } else if constexpr (sizeof(T) == 8) {
                    auto mask1 = wasm_i64x2_const_splat(0x5555'5555'5555'5555ll);
                    auto mask2 = wasm_i64x2_const_splat(0x3333'3333'3333'3333ll);
                    auto mask4 = wasm_i64x2_const_splat(0x0F0F'0F0F'0F0F'0F0Fll);

                    // Step 1: Count bits in 2-bit chunks
                    auto tmp = wasm_u64x2_shr(a, 1);
                    tmp = wasm_v128_and(tmp, mask1);
                    a = wasm_i64x2_sub(a, tmp);

                    // Step 2: Count bits in 4-bit chunks
                    tmp = wasm_u64x2_shr(a, 2);
                    tmp = wasm_v128_and(tmp, mask2);
                    a = wasm_v128_and(a, mask2);
                    a = wasm_i64x2_add(a, tmp);

                    // Step 3: Count bits in 8-bit chunks
                    tmp = wasm_u64x2_shr(a, 4);
                    a = wasm_i64x2_add(a, tmp);
                    a = wasm_v128_and(a, mask4);

                    // Sum counts in 16-bit chunk
                    a = wasm_i64x2_add(a, wasm_u64x2_shr(a, 8));
                    // Sum counts in 32-bit chunk
                    a = wasm_i64x2_add(a, wasm_u64x2_shr(a, 16));
                    // Sum counts in 64-bit value
                    a = wasm_i64x2_add(a, wasm_u64x2_shr(a, 32));

                    a = wasm_v128_and(a, wasm_i64x2_const_splat(0x007F)); // Only need 7 bits per lane
                    return from_vec<T>(a);
                }
            } else if constexpr (bits * 2 == sizeof(v128_t)) {
                return popcount(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

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
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(cond);
        if constexpr (N == 1) {
            return emul::bitwise_select(cond, true_, false_);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto c = to_vec(cond);
                auto t = to_vec(true_);
                auto f = to_vec(false_);
                return from_vec<T>(wasm_v128_bitselect(t, f, c));
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                return bitwise_select(
                    from_vec<mask_inner_t<T>>(fit_to_vec(cond)),
                    from_vec<T>(fit_to_vec(true_)),
                    from_vec<T>(fit_to_vec(false_))
                ).lo;
            }
            return join(
                bitwise_select<false>(cond.lo, true_.lo, false_.lo),
                bitwise_select<false>(cond.hi, true_.hi, false_.hi)
            );
        }
    }
// !MARK

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_BIT_HPP
