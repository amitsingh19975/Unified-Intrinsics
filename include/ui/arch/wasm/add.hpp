#ifndef AMT_UI_ARCH_WASM_ADD_HPP
#define AMT_UI_ARCH_WASM_ADD_HPP

#include "cast.hpp"
#include "../emul/add.hpp"
#include "logical.hpp"
#include "shift.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::add(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_add(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_add(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(add(
                        cast<float>(lhs),
                        cast<float>(rhs)
                    ));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_add(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_add(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_add(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_i64x2_add(l, r));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return add(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                add<false>(lhs.lo, rhs.lo),
                add<false>(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        using result_t = internal::widening_result_t<T>;
        return add(cast<result_t>(lhs), cast<result_t>(rhs));
    }

// MARK: Narrowing Addition
    template <bool Round = false, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (Round) {
            auto t0 = bitwise_and(lhs, rhs);
            auto t1 = bitwise_xor(lhs, rhs);
            auto t2 = shift_right<1>(t1); 
            return add(t0, t2);
        } else {
            auto tmp = sub<true>(rhs, lhs);
            tmp = shift_right<1>(tmp);
            return add(lhs, tmp);
        }
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
        static constexpr auto bits = sizeof(T) * N;
        using result_t = internal::narrowing_result_t<T>;
        if constexpr (N == 1) {
            return emul::high_narrowing_add(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto sum = add<false>(lhs, rhs);
                if constexpr (sizeof(T) == 2) {
                    sum = shift_right<8>(sum);
                    auto v = to_vec(sum);
                    v128_t res;
                    if constexpr (std::is_signed_v<T>) {
                        res = wasm_i8x16_narrow_i16x8(v, v);
                    } else {
                        res = wasm_u8x16_narrow_i16x8(v, v);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 4) {
                    sum = shift_right<16>(sum);
                    auto v = to_vec(sum);
                    v128_t res;
                    if constexpr (std::is_signed_v<T>) {
                        res = wasm_i16x8_narrow_i32x4(v, v);
                    } else {
                        res = wasm_u16x8_narrow_i32x4(v, v);
                    }
                    return from_vec<result_t>(res).lo;
                } else if constexpr (sizeof(T) == 8) {
                    return cast<result_t>(sum);
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return high_narrowing_add(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

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
        static constexpr auto bits = sizeof(T) * N;
        if constexpr (N == 1) {
            return emul::sat_add(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_add_sat(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_add_sat(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = wasm_i32x4_const_splat(0x7FFF'FFFF);
                        auto res = wasm_i32x4_add(l, r);
                        // (~(l ^ r)) & (l ^ res) => checks if carry

                        // 1. Get the sign bit
                        auto res_sat = wasm_u32x4_shr(l, 31);

                        // 2. add sign bit. If sign bit is 1, res_sat will be INT32_MIN; otherwise, INT32_MAX
                        res_sat = wasm_i32x4_add(res_sat, sign);

                        // 3. l ^ res
                        auto res_xor_l = wasm_v128_xor(res, l);
                        // 4. r ^ l
                        auto r_xor_l = wasm_v128_xor(r, l);
                        // 5. ~(l^r) & (l & res)
                        res_xor_l = wasm_v128_andnot(res_xor_l, r_xor_l);
                        // 6. Gets carry bit
                        res_xor_l = wasm_i32x4_shr(res_xor_l, 31);
                        // 7. (l ^ res) & (INT32_MIN or INT32_MAX)
                        res_sat = wasm_v128_and(res_xor_l, res_sat);
                        res = wasm_v128_andnot(res, res_xor_l);
                        res = wasm_v128_or(res, res_sat); 
                        return from_vec<T>(res);
                    }
                } else {
                    static constexpr auto sign_mask = static_cast<std::int32_t>(0x8000'0000);
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_add_sat(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_add_sat(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = wasm_i32x4_const_splat(sign_mask);
                        auto sum = wasm_i32x4_add(l, r);
                        auto subsum = wasm_i32x4_sub(sum, sign);
                        auto suba = wasm_i32x4_sub(l, sign); 
                        auto c = wasm_i32x4_gt(suba, subsum);
                        auto res = wasm_v128_or(sum, c);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto sign = wasm_i32x4_make(sign_mask, 0, sign_mask, 0);
                        auto sum = wasm_i64x2_add(l, r);
                        auto subsum = wasm_i64x2_sub(sum, sign);
                        auto suba = wasm_i64x2_sub(l, sign); 
                        auto c = wasm_i64x2_gt(suba, subsum);
                        auto res = wasm_v128_or(sum, c);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return sat_add(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                sat_add<false>(lhs.lo, rhs.lo),
                sat_add<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

    template <bool Merge = true, std::size_t N, std::integral T, std::integral U>
        requires (std::is_signed_v<T> != std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        auto sum = add(lhs, rcast<T>(rhs));
        // Case 1: T is signed and U unsigned
        //      lhs + rhs <= max(T)
        //      lhs <= max(T) - rhs
        //      return max(T) if lhs > max(T) - rhs
        if constexpr (std::is_signed_v<T>) {
            auto mx = Vec<N, U>::load(std::numeric_limits<T>::max());

            // Check if rhs is too large (> INT_MAX)
            auto rhs_too_large = cmp(rhs, mx, op::greater_t{});

            // Check if the addition would overflow (lhs > max(T) - rhs)
            auto safe_max = sub<true>(mx, rhs);
            auto would_overflow = cmp(lhs, rcast<T>(safe_max), op::greater_t{});

            // Combine overflow conditions
            auto overflow_mask = rcast<T>(bitwise_or(rhs_too_large, would_overflow));

            // Use max(T) for overflow cases, otherwise use sum
            return bitwise_or(
                bitwise_and(overflow_mask, rcast<T>(mx)),
                bitwise_notand(overflow_mask, sum)
            );
        } else {
            // Case 2: T is unsigned and U is signed
            auto mx = Vec<N, T>::load(std::numeric_limits<T>::max());

            // Check if rhs is negative
            auto rhs_negative = cmp(rhs, op::less_zero_t{});

            // For negative rhs, we need to do a subtraction instead
            // If rhs is negative and |rhs| <= lhs, then result is lhs - |rhs|
            auto abs_rhs = bitwise_select(rhs_negative, negate(rhs), rhs);
            auto safe_subtraction = cmp(lhs, rcast<T>(abs_rhs), op::greater_equal_t{});

            // For negative rhs, compute lhs - |rhs| directly instead of using sum
            auto neg_result = sub<true>(lhs, rcast<T>(abs_rhs));

            // If rhs is positive, check if would overflow
            auto safe_max = sub<true>(mx, rcast<T>(rhs));
            auto positive_overflow = bitwise_notand(
                rhs_negative,
                cmp(lhs, safe_max, op::greater_t{})
            );

            // Create overflow mask for positive rhs case
            auto overflow_mask = positive_overflow;

            // Select between:
            // - mx when overflow_mask is true (positive rhs overflow)
            // - neg_result when rhs is negative and safe_subtraction is true
            // - 0 when rhs is negative and safe_subtraction is false
            // - sum otherwise (positive rhs, no overflow)

            auto zero = Vec<N, T>{};
            auto result_for_neg = bitwise_select(safe_subtraction, neg_result, zero);

            // First select between regular sum and max based on overflow mask
            auto pos_result = bitwise_or(
                bitwise_and(overflow_mask, mx),
                bitwise_notand(overflow_mask, sum)
            );

            // Then select between positive and negative cases
            return bitwise_select(rhs_negative, result_for_neg, pos_result);
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
        static constexpr auto size = sizeof(T) * N;
        if constexpr (N == 2) {
            return emul::padd(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto sl = wasm_i32x4_shuffle(l, l, 1, 0, 3, 2); 
                    auto sr = wasm_i32x4_shuffle(r, r, 1, 0, 3, 2); 
                    auto s0 = wasm_f32x4_add(sl, l); 
                    auto s1 = wasm_f32x4_add(sr, r); 
                    auto res = wasm_i32x4_shuffle(s0, s1, 0, 2, 4, 6);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(padd(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    if constexpr (sizeof(T) == 2) {
                        auto sl = wasm_i16x8_shuffle(l, l, 1, 0, 3, 2, 5, 4, 7, 6);
                        auto sr = wasm_i16x8_shuffle(r, r, 1, 0, 3, 2, 5, 4, 7, 6);
                        auto s0 = wasm_i16x8_add(sl, l); 
                        auto s1 = wasm_i16x8_add(sr, r); 
                        auto res = wasm_i16x8_shuffle(s0, s1, 0, 2, 4, 6, 8, 10, 12, 14);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto sl = wasm_i32x4_shuffle(l, l, 1, 0, 3, 2); 
                        auto sr = wasm_i32x4_shuffle(r, r, 1, 0, 3, 2); 
                        auto s0 = wasm_i32x4_add(sl, l); 
                        auto s1 = wasm_i32x4_add(sr, r); 
                        auto res = wasm_i32x4_shuffle(s0, s1, 0, 2, 4, 6);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                if constexpr (sizeof(T) == 1) {
                    auto l = to_vec(cast<internal::widening_result_t<T>>(from_vec<T>(fit_to_vec(lhs))).lo);
                    auto r = to_vec(cast<internal::widening_result_t<T>>(from_vec<T>(fit_to_vec(rhs))).lo);
                    auto tmp = to_vec(padd<false>(from_vec<T>(l), from_vec<T>(r)));
                    auto res = wasm_i8x16_swizzle(tmp, *reinterpret_cast<v128_t const*>(constants::mask8_16_even_odd));
                    return from_vec<T>(res).lo;
                }
                if constexpr (Merge) {
                    return padd(
                        join(lhs, rhs),
                        Vec<2 * N, T>{}
                    ).lo;
                }
            }

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
        static constexpr auto size = sizeof(T) * N;
        if constexpr (N == 1) {
            return emul::fold(v, op);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto a = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    auto upper_half = wasm_i32x4_shuffle(a, a, 2, 3, 2, 3);
                    a = wasm_f32x4_add(a, upper_half);
                    auto shuffled_a = wasm_i32x4_shuffle(a, a, 1, 1, 1, 1);
                    a = wasm_f32x4_add(a, shuffled_a);
                    return wasm_f32x4_extract_lane(a, 0);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return static_cast<T>(fold(cast<float>(v), op));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        auto b = wasm_i8x16_shuffle(a, a,
                            8, 9, 10, 11, 12, 13, 14, 15,
                            8, 9, 10, 11, 12, 13, 14, 15
                        );
                        auto sum = wasm_i8x16_add(a, b);
                        auto res = wasm_i16x8_extadd_pairwise_i8x16(sum);
                        return static_cast<T>(wasm_i16x8_extract_lane(res, 0));
                    } else if constexpr (sizeof(T) == 2) {
                        auto b = wasm_i16x8_shuffle(a, a, 4, 5, 6, 7, 4, 5, 6, 7);
                        auto sum = wasm_i16x8_add(a, b);
                        auto shf = wasm_i32x4_shuffle(sum, sum, 1, 1, 1, 1);
                        sum = wasm_i16x8_add(sum, shf);
                        auto shifted = wasm_u32x4_shr(sum, 16);
                        sum = wasm_i16x8_add(sum, shifted);
                        return static_cast<T>(wasm_i16x8_extract_lane(sum, 0));
                    } else if constexpr (sizeof(T) == 4) {
                        auto b = wasm_i32x4_shuffle(a, a, 2, 3, 2, 3);
                        auto sum = wasm_i32x4_add(a, b);
                        auto shf = wasm_i32x4_shuffle(sum, sum, 1, 1, 1, 1);
                        sum = wasm_i32x4_add(sum, shf);
                        return static_cast<T>(wasm_i32x4_extract_lane(sum, 0));
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return fold(
                    from_vec<T>(fit_to_vec(v)),
                    op
                );
            }

            return fold<false>(v.lo, op) + fold<false>(v.hi, op);
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

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_ADD_HPP
