#ifndef AMT_UI_ARCH_WASM_LOAD_HPP
#define AMT_UI_ARCH_WASM_LOAD_HPP

#include "cast.hpp"
#include "../emul/load.hpp"
#include "basic.hpp"
#include "ui/base.hpp"
#include <array>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <wasm_simd128.h>

namespace ui::wasm {
    template <std::size_t N, typename T, bool Merge = true>
    UI_ALWAYS_INLINE auto load(T val) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(T) * N;
        if constexpr (N == 1) {
            return emul::load<N>(val);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(wasm_f32x4_splat(val));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(wasm_f64x2_splat(val));
                    }
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return rcast<T>(load<N>(std::bit_cast<std::uint16_t>(val)));
                } else {
                    using utype = std::make_unsigned_t<T>;
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_u32x4_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_u64x2_splat(static_cast<utype>(val)));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t)) {
                return load<2 * N, T>(val).lo;
            }

            auto t = load<N / 2, T, false>(val);
            return join(t, t);
        }
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return load<N>(v[Lane]);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(
        Vec<N, T> const& v
    ) noexcept -> Vec<N * 2, T> {
        using ret_t = Vec<2 * N, T>;
        static constexpr auto size = sizeof(v);
        if constexpr (size * 2 == sizeof(v128_t)) {
            return std::bit_cast<ret_t>(wasm_i64x2_splat(std::bit_cast<std::int64_t>(v)));
        }

        return ret_t(v, v);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b
    ) noexcept {
        using namespace constants;
        static constexpr auto size = N * sizeof(T);
        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0); // [d0, d1, d2, d3, ...]
                    auto t1 = wasm_v128_load(data + 16);

                    auto tmp0 = wasm_i8x16_shuffle(
                        t0, t1,
                        0, 2, 4, 6, 8, 10, 12, 14,
                        16, 18, 20, 22, 24, 26, 28, 30
                    );

                    auto tmp1 = wasm_i8x16_shuffle(
                        t0, t1,
                        1, 3, 5, 7, 9, 11, 13, 15,
                        17, 19, 21, 23, 25, 27, 29, 31
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0); // 8 elements
                    auto t1 = wasm_v128_load(data +  8); // 8 elements
                    auto tmp0 = wasm_i16x8_shuffle(
                        t0, t1,
                        0, 2, 4, 6,
                        8, 10, 12, 14
                    );

                    auto tmp1 = wasm_i16x8_shuffle(
                        t0, t1,
                        1, 3, 5, 7,
                        9, 11, 13, 15
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data + 0); // 4 elements
                    auto t1 = wasm_v128_load(data + 4); // 4 elements
                    auto tmp0 = wasm_i32x4_shuffle(
                        t0, t1,
                        0, 2, 4, 6
                    );

                    auto tmp1 = wasm_i32x4_shuffle(
                        t0, t1,
                        1, 3, 5, 7
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    return;
                } else if constexpr (sizeof(T) == 8) {
                    auto t0 = wasm_v128_load(data + 0); // 2 elements
                    auto t1 = wasm_v128_load(data + 2); // 2 elements
                    auto tmp0 = wasm_i64x2_shuffle(
                        t0, t1,
                        0, 2
                    );
                    auto tmp1 = wasm_i64x2_shuffle(
                        t0, t1,
                        1, 3
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    return;
                }
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0); // [d0, d1, d2, d3, ...]

                    auto tmp0 = wasm_i8x16_shuffle(
                        t0, t0,
                        0, 2, 4, 6, 8, 10, 12, 14,
                        0, 2, 4, 6, 8, 10, 12, 14
                    );

                    auto tmp1 = wasm_i8x16_shuffle(
                        t0, t0,
                        1, 3, 5, 7, 9, 11, 13, 15,
                        1, 3, 5, 7, 9, 11, 13, 15
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0);

                    auto tmp0 = wasm_i16x8_shuffle(
                        t0, t0,
                        0, 2, 4, 6,
                        0, 2, 4, 6
                    );

                    auto tmp1 = wasm_i16x8_shuffle(
                        t0, t0,
                        1, 3, 5, 7,
                        1, 3, 5, 7
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data + 0); // 4 elements

                    auto tmp0 = wasm_i32x4_shuffle(
                        t0, t0,
                        0, 2, 0, 2
                    );

                    auto tmp1 = wasm_i32x4_shuffle(
                        t0, t0,
                        1, 3, 1, 3
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    return;
                }
            }

            strided_load(data, a.lo, b.lo);
            strided_load(data + N / 2 * 2, a.hi, b.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c
    ) noexcept {
        using namespace constants;
        using namespace internal;

        static constexpr auto size = N * sizeof(T);
        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0); // [d0, d1, d2, d3, ...]
                    auto t1 = wasm_v128_load(data + 16);
                    auto t2 = wasm_v128_load(data + 32);

                    // [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 15][16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31][32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48]
                    // [0, -  -  3  -  -  6  -  -  9   -   -  12   -   -  15][ -  -   18  -   -   21, -   -   24  -   -   27  -   -   30   -  -   33  -   -   36  -   -   39  -   -   42  -   -   46
                    auto tmp0 = wasm_i8x16_shuffle(
                        t0, t1,
                         0,  3,  6,  9, 12, 15,
                        18, 21, 24, 27, 30,
                        0,  0,  0,  0,  0
                    );
                    tmp0 = wasm_i8x16_shuffle(
                        tmp0, t2,
                         0,  1,  2,  3,  4, 5, 6, 7, 8, 9, 10,
                        17, 20, 23, 26, 29
                    );

                    auto tmp1 = wasm_i8x16_shuffle(
                        t0, t1,
                         0 + 1,  3 + 1,  6 + 1,  9 + 1, 12 + 1, 15 + 1,
                        18 + 1, 21 + 1, 24 + 1, 27 + 1, 30 + 1,
                        0,  0,  0,  0,  0
                    );
                    tmp1 = wasm_i8x16_shuffle(
                        tmp1, t2,
                         0,  1,  2,  3,  4, 5, 6, 7, 8, 9, 10,
                        17 + 1, 20 + 1, 23 + 1, 26 + 1, 29 + 1
                    );

                    // [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 15][16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31][32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48]
                    // [-  -  2  -  -  5  -  -  8 -   -   11   -  -   14   -  -   17   -  -   20  -   -   23  -   -   26   -  -   29  -   -   32  -   -      
                    auto tmp2 = wasm_i8x16_shuffle(
                        t0, t1,
                        2, 5, 8, 11, 14,
                        17, 20, 23, 26, 29,
                        0, 0, 0, 0, 0, 0
                    );
                    tmp2 = wasm_i8x16_shuffle(
                        tmp2, t2,
                         0,  1,  2,  3,  4, 5, 6, 7, 8, 9,
                        16, 19, 22, 25, 28, 31
                    );
                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0); // 8 elements
                    auto t1 = wasm_v128_load(data +  8); // 8 elements
                    auto t2 = wasm_v128_load(data + 16); // 8 elements
                    auto tmp0 = wasm_i16x8_shuffle(
                        t0, t1,
                        0,  3,  6,
                        9, 12, 15,
                        0, 0
                    );
                    tmp0 = wasm_i16x8_shuffle(
                        tmp0, t2,
                        0, 1, 2, 3, 4, 5,
                        10, 13 
                    );

                    auto tmp1 = wasm_i16x8_shuffle(
                        t0, t1,
                        1, 4, 7,
                        10, 13,
                        0,  0, 0
                    );
                    tmp1 = wasm_i16x8_shuffle(
                        tmp1, t2,
                        0, 1, 2, 3, 4,
                        8, 11, 14
                    );

                    // [0, 1, 2, 3, 4, 5, 6 ,7][8, 9, 10, 11, 12, 13, 14]
                    auto tmp2 = wasm_i16x8_shuffle(
                        t0, t1,
                        2, 5,
                        8, 11, 14,
                        0,  0, 0
                    );
                    tmp2 = wasm_i16x8_shuffle(
                        tmp2, t2,
                        0, 1, 2, 3, 4,
                        9, 12, 15
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data + 0); // 4 elements
                    auto t1 = wasm_v128_load(data + 4); // 4 elements
                    auto t2 = wasm_v128_load(data + 8); // 4 elements
                    auto tmp0 = wasm_i32x4_shuffle(
                        t0, t1,
                        0, 3, 6, 6
                    );
                    tmp0 = wasm_i32x4_shuffle(
                        tmp0, t2,
                        0, 1, 2, 5
                    );

                    auto tmp1 = wasm_i32x4_shuffle(
                        t0, t1,
                        1, 4, 7, 7
                    );
                    tmp1 = wasm_i32x4_shuffle(
                        tmp1, t2,
                        0, 1, 2, 6
                    );

                    auto tmp2 = wasm_i32x4_shuffle(
                        t0, t1,
                        2, 5, 5, 5
                    );
                    tmp2 = wasm_i32x4_shuffle(
                        tmp2, t2,
                        0, 1, 4, 7
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    return;
                } else if constexpr (sizeof(T) == 8) {
                    auto t0 = wasm_v128_load(data + 0); // 2 elements
                    auto t1 = wasm_v128_load(data + 2); // 2 elements
                    auto t2 = wasm_v128_load(data + 4); // 2 elements
                    auto tmp0 = wasm_i64x2_shuffle(
                        t0, t1,
                        0, 3
                    );
                    auto tmp1 = wasm_i64x2_shuffle(
                        t0, t2,
                        1, 2
                    );
                    auto tmp2 = wasm_i64x2_shuffle(
                        t1, t2,
                        0, 3
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    return;
                }
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0); // [d0, d1, d2, d3, ...]
                    auto t1 = wasm_v128_load64_zero(data + 16);

                    auto tmp0 = wasm_i8x16_shuffle(
                        t0, t1,
                         0,  3,  6,  9, 12, 15,
                        18, 21, 24, 27, 30,
                        0,  0,  0,  0,  0
                    );

                    auto tmp1 = wasm_i8x16_shuffle(
                        t0, t1,
                         0 + 1,  3 + 1,  6 + 1,  9 + 1, 12 + 1, 15 + 1,
                        18 + 1, 21 + 1, 24 + 1, 27 + 1, 30 + 1,
                        0,  0,  0,  0,  0
                    );

                    auto tmp2 = wasm_i8x16_shuffle(
                        t0, t1,
                        2, 5, 8, 11, 14,
                        17, 20, 23, 26, 29,
                        0, 0, 0, 0, 0, 0
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load64_zero(data + 8);

                    auto tmp0 = wasm_i16x8_shuffle(
                        t0, t1,
                        0,  3,  6,
                        9, 12, 15,
                        0, 0
                    );

                    auto tmp1 = wasm_i16x8_shuffle(
                        t0, t1,
                        1, 4, 7,
                        10, 13,
                        0,  0, 0
                    );

                    auto tmp2 = wasm_i16x8_shuffle(
                        t0, t1,
                        2, 5,
                        11, 14, 15,
                        0,  0, 0
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data + 0); // 4 elements
                    auto t1 = wasm_v128_load(data + 2); // 2 elements

                    auto tmp0 = wasm_i32x4_shuffle(
                        t0, t1,
                        0, 3, 6, 6
                    );

                    auto tmp1 = wasm_i32x4_shuffle(
                        t0, t1,
                        1, 4, 7, 7
                    );

                    auto tmp2 = wasm_i32x4_shuffle(
                        t0, t1,
                        2, 5, 5, 5
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    return;
                }
            }
            strided_load(data, a.lo, b.lo, c.lo);
            strided_load(data + N / 2 * 3, a.hi, b.hi, c.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c,
        Vec<N, T>& UI_RESTRICT d
    ) noexcept {
        static constexpr auto size = N * sizeof(T);
        using namespace constants;
        using namespace internal;

        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
            d.val = data[3];
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load(data + 16);
                    auto t2 = wasm_v128_load(data + 32);
                    auto t3 = wasm_v128_load(data + 48);

                    auto a_lo = wasm_i8x16_shuffle(t0, t1,
                         0,  4,  8, 12,
                        16, 20, 24, 28,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto a_hi = wasm_i8x16_shuffle(t2, t3,
                         0,  4,  8, 12,
                        16, 20, 24, 28,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto tmp0 = wasm_i8x16_shuffle(a_lo, a_hi,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        16, 17, 18, 19, 20, 21, 22, 23
                    );
                    auto b_lo = wasm_i8x16_shuffle(t0, t1,
                         1,  5,  9, 13,
                        17, 21, 25, 29,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto b_hi = wasm_i8x16_shuffle(t2, t3,
                         1,  5,  9, 13,
                        17, 21, 25, 29,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto tmp1 = wasm_i8x16_shuffle(b_lo, b_hi,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        16, 17, 18, 19, 20, 21, 22, 23
                    );

                    auto c_lo = wasm_i8x16_shuffle(t0, t1,
                         2,  6, 10, 14,
                        18, 22, 26, 30,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto c_hi = wasm_i8x16_shuffle(t2, t3,
                         2,  6, 10, 14,
                        18, 22, 26, 30,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto tmp2 = wasm_i8x16_shuffle(c_lo, c_hi,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        16, 17, 18, 19, 20, 21, 22, 23
                    );

                    auto d_lo = wasm_i8x16_shuffle(t0, t1,
                         3,  7, 11, 15,
                        19, 23, 27, 31,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto d_hi = wasm_i8x16_shuffle(t2, t3,
                         3,  7, 11, 15,
                        19, 23, 27, 31,
                         0,  0,  0,  0, 0, 0, 0, 0
                    );
                    auto tmp3 = wasm_i8x16_shuffle(d_lo, d_hi,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        16, 17, 18, 19, 20, 21, 22, 23
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    d = from_vec<T>(tmp3);
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load(data +  8);
                    auto t2 = wasm_v128_load(data + 16);
                    auto t3 = wasm_v128_load(data + 24);

                    auto a_lo = wasm_i16x8_shuffle(t0, t1, 
                        0, 4,
                        8, 12,
                        0, 0, 0, 0
                    );
                    auto a_hi = wasm_i16x8_shuffle(t2, t3, 
                        0, 4,
                        8, 12,
                        0, 0, 0, 0
                    );
                    auto tmp0 = wasm_i16x8_shuffle(a_lo, a_hi,
                        0, 1, 2, 3,
                        8, 9, 10, 11
                    );

                    auto b_lo = wasm_i16x8_shuffle(t0, t1, 
                        1, 5,
                        9, 13,
                        0, 0, 0, 0
                    );
                    auto b_hi = wasm_i16x8_shuffle(t2, t3, 
                        1, 5,
                        9, 13,
                        0, 0, 0, 0
                    );
                    auto tmp1 = wasm_i16x8_shuffle(b_lo, b_hi,
                        0, 1, 2, 3,
                        8, 9, 10, 11
                    );

                    auto c_lo = wasm_i16x8_shuffle(t0, t1, 
                        2, 6,
                        10, 14,
                        0, 0, 0, 0
                    );
                    auto c_hi = wasm_i16x8_shuffle(t2, t3, 
                        2, 6,
                        10, 14,
                        0, 0, 0, 0
                    );
                    auto tmp2 = wasm_i16x8_shuffle(c_lo, c_hi,
                        0, 1, 2, 3,
                        8, 9, 10, 11
                    );

                    auto d_lo = wasm_i16x8_shuffle(t0, t1, 
                        3, 7,
                        11, 15,
                        0, 0, 0, 0
                    );
                    auto d_hi = wasm_i16x8_shuffle(t2, t3, 
                        3, 7,
                        11, 15,
                        0, 0, 0, 0
                    );
                    auto tmp3 = wasm_i16x8_shuffle(d_lo, d_hi,
                        0, 1, 2, 3,
                        8, 9, 10, 11
                    );

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    d = from_vec<T>(tmp3);
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load(data +  4);
                    auto t2 = wasm_v128_load(data +  8);
                    auto t3 = wasm_v128_load(data + 12);

                    auto a_lo = wasm_i32x4_shuffle(t0, t1, 0, 4, 0, 0);
                    auto a_hi = wasm_i32x4_shuffle(t2, t3, 0, 4, 0, 0);
                    auto tmp0 = wasm_i32x4_shuffle(a_lo, a_hi, 0, 1, 4, 5);

                    auto b_lo = wasm_i32x4_shuffle(t0, t1, 1, 5, 0, 0);
                    auto b_hi = wasm_i32x4_shuffle(t2, t3, 1, 5, 0, 0);
                    auto tmp1 = wasm_i32x4_shuffle(b_lo, b_hi, 0, 1, 4, 5);

                    auto c_lo = wasm_i32x4_shuffle(t0, t1, 2, 6, 0, 0);
                    auto c_hi = wasm_i32x4_shuffle(t2, t3, 2, 6, 0, 0);
                    auto tmp2 = wasm_i32x4_shuffle(c_lo, c_hi, 0, 1, 4, 5);

                    auto d_lo = wasm_i32x4_shuffle(t0, t1, 3, 7, 0, 0);
                    auto d_hi = wasm_i32x4_shuffle(t2, t3, 3, 7, 0, 0);
                    auto tmp3 = wasm_i32x4_shuffle(d_lo, d_hi, 0, 1, 4, 5);

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    d = from_vec<T>(tmp3);
                    return;
                } else if constexpr (sizeof(T) == 8) {
                    auto t0 = wasm_v128_load(data + 0); // 2 elements
                    auto t1 = wasm_v128_load(data + 2); // 2 elements
                    auto t2 = wasm_v128_load(data + 4); // 2 elements
                    auto t3 = wasm_v128_load(data + 6); // 2 elements

                    auto tmp0 = wasm_i64x2_shuffle(t0, t2, 0, 2);
                    auto tmp1 = wasm_i64x2_shuffle(t0, t2, 1, 3);
                    auto tmp2 = wasm_i64x2_shuffle(t1, t3, 0, 2);
                    auto tmp3 = wasm_i64x2_shuffle(t1, t3, 1, 3);

                    a = from_vec<T>(tmp0);
                    b = from_vec<T>(tmp1);
                    c = from_vec<T>(tmp2);
                    d = from_vec<T>(tmp3);
                    return;
                }
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load(data + 16);

                    auto tmp0 = wasm_i8x16_shuffle(
                        t0, t1,
                        0,  4,  8, 12,
                        16, 20, 24, 28,
                        0,  0,  0,  0,  0, 0, 0, 0
                    );

                    auto tmp1 = wasm_i8x16_shuffle(
                        t0, t1,
                        1,  5,  9, 13,
                        17, 21, 25, 29,
                        0,  0,  0,  0,  0, 0, 0, 0
                    );

                    auto tmp2 = wasm_i8x16_shuffle(
                        t0, t1,
                        2,  6, 10, 14,
                        18, 22, 26, 30,
                        0,  0,  0,  0,  0, 0, 0, 0
                    );

                    auto tmp3 = wasm_i8x16_shuffle(
                        t0, t1,
                        3,  7, 11, 15,
                        19, 23, 27, 31,
                        0,  0,  0,  0,  0, 0, 0, 0
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    d = from_vec<T>(tmp3).lo;
                    return;
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_load(data +  0);
                    auto t1 = wasm_v128_load(data +  8);

                    auto tmp0 = wasm_i16x8_shuffle(t0, t1,
                        0,  4,
                        8, 12,
                        0, 0, 0, 0
                    );
                    auto tmp1 = wasm_i16x8_shuffle(t0, t1,
                        1,  5,
                        9, 13,
                        0, 0, 0, 0
                    );
                    auto tmp2 = wasm_i16x8_shuffle(t0, t1,
                        2,  6,
                        10, 14,
                        0, 0, 0, 0
                    );
                    auto tmp3 = wasm_i16x8_shuffle(t0, t1,
                        3,  7,
                        11, 15,
                        0, 0, 0, 0
                    );

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    d = from_vec<T>(tmp3).lo;
                    return;
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_load(data + 0);
                    auto t1 = wasm_v128_load(data + 4);

                    auto tmp0 = wasm_i32x4_shuffle(t0, t1, 0, 4, 0, 0);
                    auto tmp1 = wasm_i32x4_shuffle(t0, t1, 1, 5, 0, 0);
                    auto tmp2 = wasm_i32x4_shuffle(t0, t1, 2, 6, 0, 0);
                    auto tmp3 = wasm_i32x4_shuffle(t0, t1, 3, 7, 0, 0);

                    a = from_vec<T>(tmp0).lo;
                    b = from_vec<T>(tmp1).lo;
                    c = from_vec<T>(tmp2).lo;
                    d = from_vec<T>(tmp3).lo;
                    return;
                }
            }
            strided_load(data, a.lo, b.lo, c.lo, d.lo);
            strided_load(data + N / 2 * 4, a.hi, b.hi, c.hi, d.hi);
        }
    }
} // namespace ui::wasm

#undef SWAP_HI_LOW_32

#endif // AMT_UI_ARCH_WASM_LOAD_HPP
