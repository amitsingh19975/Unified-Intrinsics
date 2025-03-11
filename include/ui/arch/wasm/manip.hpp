#ifndef AMT_UI_ARCH_WASM_MANIP_HPP
#define AMT_UI_ARCH_WASM_MANIP_HPP

#include "cast.hpp"
#include "../emul/manip.hpp"
#include "logical.hpp"
#include "shift.hpp"
#include "bit.hpp"
#include "ui/arch/basic.hpp"
#include <cstdint>
#include <type_traits>
#include <wasm_simd128.h>

namespace ui::wasm {
    // MARK: Copy vector lane
    template <unsigned ToLane, unsigned FromLane, bool Merge = true, std::size_t N, std::size_t M, typename T>
        requires (ToLane < N && FromLane < M)
    UI_ALWAYS_INLINE auto copy(
        Vec<N, T> const& to,
        Vec<M, T> const& from
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(to);

        if constexpr (size < sizeof(v128_t)) {
            return emul::copy<ToLane, FromLane>(to, from);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_replace_lane(to_vec(to), ToLane, from[FromLane]));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_replace_lane(to_vec(to), ToLane, from[FromLane]));
                } else if constexpr (internal::is_fp16<T>) {
                    return from_vec<T>(wasm_u16x8_replace_lane(to_vec(to), ToLane, std::bit_cast<std::uint16_t>(from[FromLane])));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_i64x2_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_u32x4_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_u64x2_replace_lane(to_vec(to), ToLane, from[FromLane]));
                    }
                }
            }
            if constexpr (ToLane < N / 2) {
                return join(
                    copy<ToLane, FromLane>(to.lo, from),
                    to.hi
                );
            } else {
                return join(
                    to.lo,
                    copy<ToLane - N / 2, FromLane>(to.hi, from)
                );
            }
        }
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
            if constexpr (bits == sizeof(v128_t)) {
                auto a = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    static const constexpr std::uint8_t table8[] = {
                        0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                        0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
                    };
                    auto lookup = *reinterpret_cast<v128_t const*>(table8);
                    auto lo = wasm_v128_and(a, wasm_i8x16_const_splat(0x0F));
                    auto hi = to_vec(shift_right<4>(from_vec<std::uint8_t>(a)));
                    auto res = wasm_v128_or(
                        to_vec(shift_left<4>(from_vec<std::uint8_t>(wasm_i8x16_swizzle(lookup, lo)))),
                        wasm_i8x16_swizzle(lookup, hi)
                    );
                    return from_vec<T>(res);
                } else {
                    auto res = rev_helper(rev_helper, rcast<mtype>(reverse_bits(rcast<std::uint8_t>(v))));
                    return rcast<T>(res);
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return reverse_bits(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

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
            if constexpr (bits == sizeof(v128_t)) {
                auto a = to_vec(rcast<mtype>(v));
                if constexpr (sizeof(T) == 1) {
                    alignas(16) static constexpr std::int8_t mask_rev_e8[16] = {
                        15,14,13,12,11,10,9,8,
                         7, 6, 5, 4, 3, 2,1,0}; 
                    return rcast<T>(from_vec<mtype>(wasm_i8x16_swizzle(a, *reinterpret_cast<v128_t const*>(mask_rev_e8))));
                } else if constexpr (sizeof(T) == 2) {
                    alignas(16) static constexpr std::int8_t mask_rev_e16[16] = {
                        14,15, 12,13, 10,11, 8,9,
                         6, 7,  4, 5,  2, 3, 0,1
                    };
                    return rcast<T>(from_vec<mtype>(wasm_i8x16_swizzle(a, *reinterpret_cast<v128_t const*>(mask_rev_e16))));
                } else if constexpr (sizeof(T) == 4) {
                    return rcast<T>(from_vec<mtype>(wasm_i32x4_shuffle(
                        a, a,
                        3, 2, 1, 0
                    )));
                } else if constexpr (sizeof(T) == 8) {
                    return rcast<T>(from_vec<mtype>(wasm_i32x4_shuffle(
                        a, a, 2, 3, 0, 1
                    )));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return reverse(
                    from_vec<T>(fit_to_vec(v))
                ).hi;
            }

            return join(
                reverse<false>(v.hi),
                reverse<false>(v.lo)
            );
        }
    }
// !MARK

// MARK: Zip
    namespace internal {
        UI_ALWAYS_INLINE static auto unpacklo_i8(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i8x16_shuffle(
                a, b,
                0, 16, 1, 17, 2, 18, 3, 19,
                4, 20, 5, 21, 6, 22, 7, 23
            );
        }

        UI_ALWAYS_INLINE static auto unpacklo_i16(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i16x8_shuffle(
                a, b,
                0, 8, 1, 9, 2, 10, 3, 11
            );
        }

        UI_ALWAYS_INLINE static auto unpacklo_i32(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i32x4_shuffle(a, b, 0, 4, 1, 5);
        }

        UI_ALWAYS_INLINE static auto unpacklo_i64(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i64x2_shuffle(a, b, 0, 2);
        }

        UI_ALWAYS_INLINE static auto unpackhi_i8(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i8x16_shuffle(
                a, b,
                8, 24, 9, 25, 10, 26, 11, 27,
                12, 28, 13, 29, 14, 30, 15, 31
            );
        }

        UI_ALWAYS_INLINE static auto unpackhi_i16(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i16x8_shuffle(
                a, b,
                4, 12, 5, 13, 6, 14, 7, 15
            );
        }

        UI_ALWAYS_INLINE static auto unpackhi_i32(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i32x4_shuffle(a, b, 2, 6, 3, 7);
        }

        UI_ALWAYS_INLINE static auto unpackhi_i64(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i64x2_shuffle(a, b, 1, 3);
        }

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
                    return rcast<T>(from_vec<T>(unpacklo_i32(l, r)).lo.lo);
                } else if constexpr (sizeof(T) == 8) {
                    auto l = to_vec(rcast<std::int64_t>(a)); 
                    auto r = to_vec(rcast<std::int64_t>(b)); 
                    return rcast<T>(from_vec<std::int64_t>(unpacklo_i64(l, r)));
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
                    l = wasm_i32x4_shuffle(l, l, 0, 0, 1, 1);
                    r = wasm_i32x4_shuffle(r, r, 0, 0, 1, 1);
                    return rcast<T>(from_vec<T>(unpackhi_i32(l, r)).hi.lo);
                } else if constexpr (sizeof(T) == 8) {
                    auto l = to_vec(rcast<std::int64_t>(a)); 
                    auto r = to_vec(rcast<std::int64_t>(b)); 
                    return rcast<T>(from_vec<std::int64_t>(unpackhi_i64(l, r)));
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
                if constexpr (bits == sizeof(v128_t)) {
                    if constexpr (sizeof(T) == 1) {
                        auto l = to_vec(rcast<std::int8_t>(a));
                        auto r = to_vec(rcast<std::int8_t>(b));
                        return rcast<T>(from_vec<std::int8_t>(unpacklo_i8(l, r)));
                    } else if constexpr (sizeof(T) == 2) {
                        auto l = to_vec(rcast<std::int16_t>(a));
                        auto r = to_vec(rcast<std::int16_t>(b));
                        return rcast<T>(from_vec<std::int16_t>(unpacklo_i16(l, r)));
                    } else if constexpr (sizeof(T) == 4) {
                        auto l = to_vec(rcast<std::int32_t>(a));
                        auto r = to_vec(rcast<std::int32_t>(b));
                        return rcast<T>(from_vec<std::int32_t>(unpacklo_i32(l, r)));
                    } else if constexpr (sizeof(T) == 8) {
                        auto l = to_vec(rcast<std::int64_t>(a));
                        auto r = to_vec(rcast<std::int64_t>(b));
                        return rcast<T>(from_vec<std::int64_t>(unpacklo_i64(l, r)));
                    }
                } else if constexpr (bits * 2 == sizeof(v128_t)) {
                    return low(
                        join(a, b),
                        Vec<2 * N, T>{}
                    ).lo;
                }
            }

            template <std::size_t N, typename T>
                requires (N > 2)
            UI_ALWAYS_INLINE auto high(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept {
                static constexpr auto bits = sizeof(a);
                if constexpr (bits == sizeof(v128_t)) {
                    if constexpr (sizeof(T) == 1) {
                        auto l = to_vec(rcast<std::int8_t>(a));
                        auto r = to_vec(rcast<std::int8_t>(b));
                        return rcast<T>(from_vec<std::int8_t>(unpackhi_i8(l, r)));
                    } else if constexpr (sizeof(T) == 2) {
                        auto l = to_vec(rcast<std::int16_t>(a));
                        auto r = to_vec(rcast<std::int16_t>(b));
                        return rcast<T>(from_vec<std::int16_t>(unpackhi_i16(l, r)));
                    } else if constexpr (sizeof(T) == 4) {
                        auto l = to_vec(rcast<std::int32_t>(a));
                        auto r = to_vec(rcast<std::int32_t>(b));
                        return rcast<T>(from_vec<std::int32_t>(unpackhi_i32(l, r)));
                    } else if constexpr (sizeof(T) == 8) {
                        auto l = to_vec(rcast<std::int64_t>(a));
                        auto r = to_vec(rcast<std::int64_t>(b));
                        return rcast<T>(from_vec<std::int64_t>(unpackhi_i64(l, r)));
                    }
                } else if constexpr (bits * 2 == sizeof(v128_t)) {
                    auto l = fit_to_vec(a);
                    auto r = fit_to_vec(b);

                    if constexpr (sizeof(T) == 1) {
                        l = wasm_i32x4_shuffle(l, l, 1, 1, 1, 1);
                        r = wasm_i32x4_shuffle(r, r, 1, 1, 1, 1);
                        return low(from_vec<T>(l), from_vec<T>(r)).lo;
                    } else if constexpr (sizeof(T) == 2) {
                        auto mask = wasm_i8x16_const(
                            0, 0, // Zero out first 16 bits
                            0, 0, // Zero out second 16 bits
                            0, 1,   // Move X0
                            2, 3,   // Move X1

                            4, 5,   // Keep X2
                            6, 7,   // Keep X3
                            0, 0, // Zero out last 16 bits
                            0, 0  // Zero out last 16 bits
                        );
                        l = wasm_i8x16_swizzle(l, mask);
                        r = wasm_i8x16_swizzle(r, mask);
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
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = wasm_i8x16_const(
                        0, 2, 4, 6, 8, 10, 12, 14,
                        0, 2, 4, 6, 8, 10, 12, 14
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int8_t>(res));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = wasm_i8x16_make(
                        0, 1, 4, 5, 8, 9, 12, 13,
                        0, 1, 4, 5, 8, 9, 12, 13
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int32_t>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = wasm_i32x4_shuffle(l, l, 0, 2, 0, 2);
                    r = wasm_i32x4_shuffle(r, r, 0, 2, 0, 2);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int64_t>(res));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                auto tmp = unzip_low(from_vec<T>(l), from_vec<T>(r)); // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
                return join(tmp.lo.lo, tmp.hi.lo);
            }
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
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = wasm_i8x16_make(
                        1, 3, 5, 7, 9, 11, 13, 15,
                        1, 3, 5, 7, 9, 11, 13, 15
                    );

                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int8_t>(res));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = wasm_i8x16_make(
                        2, 3, 6, 7, 10, 11, 14, 15,
                        2, 3, 6, 7, 10, 11, 14, 15
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int16_t>(res));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));

                    l = wasm_i32x4_shuffle(l, l, 1, 3, 1, 3);
                    r = wasm_i32x4_shuffle(r, r, 1, 3, 1, 3);
                    auto blendMask = wasm_i64x2_const(0, ~std::int64_t{});
                    auto res = wasm_v128_bitselect(r, l, blendMask);
                    return rcast<T>(from_vec<std::int32_t>(res));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
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
        using namespace internal;
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::transpose_low(a, b); 
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = wasm_i8x16_make(
                        0, 2, 4, 6, 8, 10, 12, 14,
                        0, 2, 4, 6, 8, 10, 12, 14
                    );

                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    return rcast<T>(from_vec<std::int8_t>(unpacklo_i8(l, r)));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = wasm_i8x16_make(
                        0, 1, 4, 5, 8, 9, 12, 13,
                        0, 1, 4, 5, 8, 9, 12, 13
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    return rcast<T>(from_vec<std::int32_t>(unpacklo_i16(l, r)));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = wasm_i32x4_shuffle(l, l, 0, 2, 1, 3);
                    r = wasm_i32x4_shuffle(r, r, 0, 2, 1, 3);
                    return rcast<T>(from_vec<std::int64_t>(unpacklo_i32(l, r)));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                return transpose_low(from_vec<T>(l), from_vec<T>(r)).lo; // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
            }

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
        using namespace internal;
        static constexpr auto bits = sizeof(a);
        if constexpr (N == 2) {
            return emul::transpose_high(a, b); 
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(rcast<std::int8_t>(a));
                    auto r = to_vec(rcast<std::int8_t>(b));
                    auto mask = wasm_i8x16_make(
                        1, 3, 5, 7, 9, 11, 13, 15,
                        1, 3, 5, 7, 9, 11, 13, 15
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    return rcast<T>(from_vec<std::int8_t>(unpacklo_i8(l, r)));
                } else if constexpr (sizeof(T) == 2) {
                    auto l = to_vec(rcast<std::int16_t>(a));
                    auto r = to_vec(rcast<std::int16_t>(b));
                    auto mask = wasm_i8x16_make(
                        2, 3, 6, 7, 10, 11, 14, 15,
                        2, 3, 6, 7, 10, 11, 14, 15
                    );
                    l = wasm_i8x16_swizzle(l, mask);
                    r = wasm_i8x16_swizzle(r, mask);
                    return rcast<T>(from_vec<std::int32_t>(unpacklo_i16(l, r)));
                } else if constexpr (sizeof(T) == 4) {
                    auto l = to_vec(rcast<std::int32_t>(a));
                    auto r = to_vec(rcast<std::int32_t>(b));
                    l = wasm_i32x4_shuffle(l, l, 1, 3, 1, 3);
                    r = wasm_i32x4_shuffle(r, r, 1, 3, 1, 3);
                    return rcast<T>(from_vec<std::int64_t>(unpacklo_i32(l, r)));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                auto l = fit_to_vec(a); // [a0, a1, a2, a3, a4, ...] => [a0, a2, a4, a6, ...]
                auto r = fit_to_vec(b); // [b0, b1, b2, b3, b4, ...] => [b0, b2, b4, b6, ...]
                return transpose_high(from_vec<T>(l), from_vec<T>(r)).lo; // [a0, b0, a2, b2, a4, b4, a6, b6 ...]
            }

            return join(
                transpose_high<false>(a.lo, b.lo),
                transpose_high<false>(a.hi, b.hi)
            );
        }
    }
// !MARK

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_MANIP_HPP
