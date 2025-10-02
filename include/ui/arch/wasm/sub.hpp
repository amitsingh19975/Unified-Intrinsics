#ifndef AMT_UI_ARCH_WASM_SUB_HPP
#define AMT_UI_ARCH_WASM_SUB_HPP

#include "cast.hpp"
#include "../emul/sub.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_sub(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_sub(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(
                        cast<float>(lhs),
                        cast<float>(rhs)
                    ));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_sub(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_sub(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_sub(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_i64x2_sub(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return sub(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

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
            if constexpr (size == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = wasm_i8x16_const_splat(
                            static_cast<std::int8_t>(128)
                        );
                        l = wasm_i8x16_add(l, mx);
                        r = wasm_i8x16_add(r, mx);
                    }
                    auto avg = wasm_u8x16_avgr(l, r);
                    auto res = wasm_i8x16_sub(l, avg);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto mx = wasm_i16x8_const_splat(static_cast<std::int16_t>(0x8000));
                        l = wasm_i16x8_add(l, mx);
                        r = wasm_i16x8_add(r, mx);
                    }
                    auto avg = wasm_u16x8_avgr(l, r);
                    auto res = wasm_i16x8_sub(l, avg);
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
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return halving_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

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
            if constexpr (size == sizeof(v128_t)) {
                auto res = to_vec(sub(lhs, rhs));
                if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = wasm_i16x8_shr(res, 8);
                        res = wasm_i8x16_narrow_i16x8(s, s);
                    } else {
                        auto s = wasm_u16x8_shr(res, 8);
                        res = wasm_u8x16_narrow_i16x8(s, s);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (std::is_signed_v<T>) {
                        auto s = wasm_i32x4_shr(res, 8);
                        res = wasm_i16x8_narrow_i32x4(s, s);
                    } else {
                        auto s = wasm_u32x4_shr(res, 8);
                        res = wasm_u16x8_narrow_i32x4(s, s);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 8) {
                    res =  wasm_i32x4_shuffle(res, res, 1, 3, 0, 2);
                    return from_vec<result_t>(res).lo;
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return high_narrowing_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

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
            if constexpr (size == sizeof(v128_t)) {
                auto a = to_vec(lhs);
                auto b = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(wasm_i8x16_sub_sat(a, b));
                    } else {
                        return from_vec<T>(wasm_u8x16_sub_sat(a, b));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (std::is_signed_v<T>) {
                        return from_vec<T>(wasm_i16x8_sub_sat(a, b));
                    } else {
                        return from_vec<T>(wasm_u16x8_sub_sat(a, b));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    v128_t res;
                    if constexpr (std::is_signed_v<T>) {
                        auto mask = wasm_i32x4_const_splat(0X7FFF'FFFF);
                        res = wasm_i32x4_sub(a, b);
                        auto res_sat = wasm_u32x4_shr(a, 31);
                        res_sat = wasm_i32x4_add(res_sat, mask);
                        auto res_xor_a = wasm_v128_xor(res, a);
                        auto b_xor_a = wasm_v128_xor(b, a);
                        res_xor_a = wasm_v128_and(b_xor_a, res_xor_a);
                        res_xor_a = wasm_i32x4_shr(res_xor_a, 31);
                        res_sat = wasm_v128_and(res_xor_a, res_sat);
                        res = wasm_v128_andnot(res, res_xor_a);
                        return from_vec<T>(wasm_v128_or(res, res_sat));
                    } else {
                        auto min = wasm_u32x4_min(a, b);
                        auto mask = wasm_i32x4_eq(min, b);
                        res = to_vec(sub(lhs, rhs));
                        return from_vec<T>(wasm_v128_and(res, mask));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (std::is_signed_v<T>) {
                        return emul::sat_sub(lhs, rhs);
                    } else {
                        auto mask = wasm_i32x4_const(0x0, int_max_mask, 0x0, int_max_mask);
                        auto res = to_vec(sub(lhs, rhs));
                        auto suba = to_vec(sub(lhs, from_vec<T>(mask)));
                        auto subb = to_vec(sub(rhs, from_vec<T>(mask)));
                        auto c = wasm_i64x2_gt(suba, subb);
                        return from_vec<T>(wasm_v128_and(res, c));
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return sat_sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

            return join(
                sat_sub<false>(lhs.lo, rhs.lo),
                sat_sub<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Subtaction with carry
    template <std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        T a,
        T b,
        T carry = {}
    ) noexcept -> std::pair<T /*result*/, T /*carry*/> {
        return emul::subc(a, b, carry);
    }

    template <std::size_t N, std::integral T>
        requires (std::is_unsigned_v<T>)
    UI_ALWAYS_INLINE auto subc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T carry = {}
    ) noexcept -> std::pair<Vec<N, T>, T /*carry*/> {
        return emul::subc(a, b, carry);
    }
// !MARK
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_SUB_HPP
