#ifndef AMT_UI_ARCH_WASM_CMP_HPP
#define AMT_UI_ARCH_WASM_CMP_HPP

#include "cast.hpp"
#include "../emul/cmp.hpp"
#include "ui/base.hpp"
#include <type_traits>

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

// MARK: Bitwise equal and 'and' test
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::equal_t op
    ) noexcept -> mask_t<N, T> {
        static constexpr auto bits = sizeof(lhs);
        using result_t = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(wasm_f32x4_eq(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(wasm_f64x2_eq(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(
                        cast<float>(lhs),
                        cast<float>(rhs),
                        op
                    ));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_i8x16_eq(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_i16x8_eq(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_i32x4_eq(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(wasm_i64x2_eq(l, r));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return cmp(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs)),
                    op
                ).lo;
            }

            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    /**
     * @return (lhs & rhs) != 0
     */
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::and_test_t op
    ) noexcept -> mask_t<N, T> {
        static constexpr auto bits = sizeof(lhs);
        using result_t = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    auto t0 = wasm_v128_and(l, r);
                    auto z = wasm_i8x16_const_splat(0);
                    auto res = wasm_i8x16_ne(t0, z);
                    return from_vec<result_t>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto t0 = wasm_v128_and(l, r);
                    auto z = wasm_i16x8_const_splat(0);
                    auto res = wasm_i16x8_ne(t0, z);
                    return from_vec<result_t>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto t0 = wasm_v128_and(l, r);
                    auto z = wasm_i32x4_const_splat(0);
                    auto res = wasm_i32x4_ne(t0, z);
                    return from_vec<result_t>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto t0 = wasm_v128_and(l, r);
                    auto z = wasm_i64x2_const_splat(0);
                    auto res = wasm_i64x2_ne(t0, z);
                    return from_vec<result_t>(res);
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return cmp(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs)),
                    op
                ).lo;
            }

            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }
// !MARK

// MARK:  Greater than or equal to
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        static constexpr auto bits = sizeof(lhs);
        using result_t = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(wasm_f32x4_ge(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(wasm_f64x2_ge(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(
                        cast<float>(lhs),
                        cast<float>(rhs),
                        op
                    ));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_i8x16_ge(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_i16x8_ge(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_i32x4_ge(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(wasm_i64x2_ge(l, r));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_u8x16_ge(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_u16x8_ge(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_u32x4_ge(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        auto mx = Vec<N, T>{ std::max(lhs[0], rhs[0]), std::max(lhs[1], rhs[1]) }; 
                        return cmp(mx, lhs, op::equal_t{});
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return cmp(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs)),
                    op
                ).lo;
            }

            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::greater_equal_t{}); 
    }
// !MARK

// MARK: Less than or equal to
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::less_equal_t op
    ) noexcept -> mask_t<N, T> {
        static constexpr auto bits = sizeof(lhs);
        using result_t = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(wasm_f32x4_le(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(wasm_f64x2_le(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(
                        cast<float>(lhs),
                        cast<float>(rhs),
                        op
                    ));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_i8x16_le(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_i16x8_le(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_i32x4_le(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(wasm_i64x2_le(l, r));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_u8x16_le(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_u16x8_le(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_u32x4_le(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        auto mx = Vec<N, T>{ std::min(lhs[0], rhs[0]), std::min(lhs[1], rhs[1]) }; 
                        return cmp(mx, lhs, op::equal_t{});
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return cmp(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs)),
                    op
                ).lo;
            }

            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::less_equal_t{});
    }
// !MARK

// MARK: Greater Than
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_t op
    ) noexcept -> mask_t<N, T> {
        static constexpr auto bits = sizeof(lhs);
        using result_t = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(wasm_f32x4_gt(l, r)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(wasm_f64x2_gt(l, r)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(
                        cast<float>(lhs),
                        cast<float>(rhs),
                        op
                    ));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_i8x16_gt(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_i16x8_gt(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_i32x4_gt(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<result_t>(wasm_i64x2_gt(l, r));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<result_t>(wasm_u8x16_gt(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<result_t>(wasm_u16x8_gt(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<result_t>(wasm_u32x4_gt(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        auto sign = wasm_i64x2_const_splat(static_cast<std::int64_t>(0x8000'0000'0000'0000ULL));
                        auto a = wasm_i64x2_sub(l, sign);
                        auto b = wasm_i64x2_sub(r, sign);
                        return from_vec<result_t>(wasm_i64x2_gt(a, b));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return cmp(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs)),
                    op
                ).lo;
            }

            return join(
                cmp<false>(lhs.lo, rhs.lo, op),
                cmp<false>(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::greater_t{});
    }
// !MARK

// MARK: Less Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::less_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(rhs, lhs, op::greater_t{});
    }
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_zero_t op
    ) noexcept -> mask_t<N, T> {
        return cmp(v, Vec<N, T>{}, op::less_t{});
    }
// !MARK

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T>;

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T>;

// MARK: Absolute greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::greater_equal_t{}
            );
        } else {
            return cmp(
                abs<true>(lhs),
                abs<true>(rhs),
                op::greater_equal_t{}
            );
        }
    }
// !MARK

// MARK: Absolute less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_equal_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::less_equal_t{}
            );
        } else {
            return cmp(
                abs<true>(lhs),
                abs<true>(rhs),
                op::less_equal_t{}
            );
        }
    }
// !MARK

// MARK: Absolute greater than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::greater_t{}
            );
        } else {
            return cmp(
                abs<true>(lhs),
                abs<true>(rhs),
                op::greater_t{}
            );
        }
    }
// !MARK

// MARK: Absolute less than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::less_t{}
            );
        } else {
            return cmp(
                abs<true>(lhs),
                abs<true>(rhs),
                op::less_t{}
            );
        }
    }
// !MARK

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_CMP_HPP
