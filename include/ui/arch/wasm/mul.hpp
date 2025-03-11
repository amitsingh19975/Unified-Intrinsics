#ifndef AMT_UI_ARCH_WASM_MUL_HPP
#define AMT_UI_ARCH_WASM_MUL_HPP

#include "cast.hpp"
#include "../emul/mul.hpp"
#include "add.hpp"
#include "ui/base.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

// MARK: Multiplication
    namespace internal {
        template <bool Merge = true, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto mul_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs
        ) noexcept -> Vec<N, T> {
            static constexpr auto size = sizeof(lhs);
            if constexpr (N == 1) {
                return emul::mul(lhs, rhs);
            } else {
                if constexpr (size == sizeof(v128_t)) {
                    if constexpr (std::same_as<T, float>) {
                        return from_vec<T>(wasm_f32x4_mul(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (std::same_as<T, double>) {
                        return from_vec<T>(wasm_f64x2_mul(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return cast<T>(mul_helper(cast<float>(lhs), cast<float>(rhs)));
                    } else {
                        if constexpr (sizeof(T) == 1) {
                            auto a = to_vec(lhs);
                            auto b = to_vec(rhs);
                            auto even = wasm_i16x8_mul(a, b);
                            auto odd = wasm_i16x8_mul(
                                wasm_u16x8_shr(a, 8),
                                wasm_u16x8_shr(b, 8)
                            );
                            auto mx = wasm_i16x8_const_splat(0xff);
                            auto res = wasm_v128_xor(
                                wasm_i16x8_shl(odd, 8),
                                wasm_v128_and(even, mx) 
                            );
                            return from_vec<T>(res);
                        } else if constexpr (sizeof(T) == 2) {
                            return from_vec<T>(wasm_i16x8_mul(to_vec(lhs), to_vec(rhs)));
                        } else if constexpr (sizeof(T) == 4) {
                            return from_vec<T>(wasm_i32x4_mul(to_vec(lhs), to_vec(rhs)));
                        } else if constexpr (sizeof(T) == 8) {
                            return from_vec<T>(wasm_i64x2_mul(to_vec(lhs), to_vec(rhs)));
                        }
                    }
                } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                    return mul_helper(
                        from_vec<T>(fit_to_vec(lhs)),
                        from_vec<T>(fit_to_vec(rhs))
                    ).lo;
                }

                return join(
                    mul_helper<false>(lhs.lo, rhs.lo),
                    mul_helper<false>(lhs.hi, rhs.hi)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
       return internal::mul_helper(lhs, rhs); 
    }
// !MARK

// MARK: Multiply-Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (::ui::internal::is_fp16<T>) {
            auto a = cast<float>(acc);
            auto l = cast<float>(lhs);
            auto r = cast<float>(rhs);
            return cast<T>(add(
                a,
                mul(l, r)
            ));
        } else {
            return add(
                acc,
                mul(lhs, rhs)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (::ui::internal::is_fp16<T>) {
            auto a = cast<float>(acc);
            auto l = cast<float>(lhs);
            auto r = cast<float>(rhs);
            return cast<T>(sub(
                a,
                mul(l, r)
            ));
        } else {
            return sub(
                acc,
                mul(lhs, rhs)
            );
        }
    }

    template <typename Op, std::size_t N, std::integral T>
        requires (sizeof(T) < 8 && (std::same_as<Op, op::add_t> || std::same_as<Op, op::sub_t>))
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        Op op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto l = cast<result_t>(lhs);
        auto r = cast<result_t>(rhs);
        return mul_acc(acc, l, r, op);
    }
// !MARK

    namespace internal {
        template <bool Merge = true, std::size_t N, std::floating_point T, typename Op>
            requires (std::same_as<Op, op::add_t> || std::same_as<Op, op::sub_t>)
        UI_ALWAYS_INLINE auto fused_mul_helper(
            Vec<N, T> const& acc,
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            Op op 
        ) noexcept -> Vec<N, T> {
            #ifdef UI_EMPSCRIPTEN_WASM_RELAXED_SIMD
            static constexpr auto size = sizeof(lhs);
            if constexpr (N == 1) {
                return emul::fused_mul_acc(acc, lhs, rhs, op);
            } else {
                if constexpr (size == sizeof(v128_t)) {
                    auto a = to_vec(acc);
                    auto l = to_vec(lhs);
                    auto r = to_vec(rhs);
                    if constexpr (std::same_as<T, float>) {
                        v128_t res;
                        if constexpr (std::same_as<Op, op::add_t>) {
                            res = wasm_f32x4_relaxed_madd(l, r, a);
                        } else {
                            res = wasm_f32x4_relaxed_nmadd(l, r, a);
                        }
                        return from_vec<T>(res);
                    } else if constexpr (std::same_as<T, double>) {
                        v128_t res;
                        if constexpr (std::same_as<Op, op::add_t>) {
                            res = wasm_f64x2_relaxed_madd(l, r, a);
                        } else {
                            res = wasm_f64x2_relaxed_nmadd(l, r, a);
                        }
                        return from_vec<T>(res);
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return cast<T>(fused_mul_helper(
                            cast<float>(acc),
                            cast<float>(lhs),
                            cast<float>(rhs),
                            op
                        ));
                    }
                } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                    return fused_mul_helper(
                        from_vec<T>(fit_to_vec(lhs)),
                        from_vec<T>(fit_to_vec(rhs)),
                        op
                    ).lo;
                }

                return join(
                    fused_mul_helper<false>(acc.lo, lhs.lo, rhs.lo, op),
                    fused_mul_helper<false>(acc.hi, lhs.hi, rhs.hi, op)
                );
            }
            #else
            return mul_acc(acc, lhs, rhs, op);
            #endif
        }
    } // namespace internal

// MARK: Fused-Multiply-Accumulate
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return internal::fused_mul_helper(acc, lhs, rhs, op);
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return internal::fused_mul_helper(acc, lhs, rhs, op);
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return fused_mul_acc(acc, a, temp, op);
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return fused_mul_acc(acc, a, temp, op);
    }
// !MARK

// MARK: Widening Multiplication
    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto l = cast<result_t>(lhs);
        auto r = cast<result_t>(rhs);
        return mul(l, r);
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar
    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return mul_acc(a, b, temp, op);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return mul_acc(a, b, temp, op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(c);
        return mul_acc(a, b, temp, op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(c);
        return mul_acc(a, b, temp, op);
    }
// !MARK

// MARK: Multiplication with scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, T> {
        return mul(v, Vec<N, T>::load(c));
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return mul(a, Vec<N, T>::load(v[Lane]));
    }
// !MARK

// MARK: Multiplication with scalar and widen
    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul(cast<result_t>(v), static_cast<result_t>(c));
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul(cast<result_t>(a), static_cast<result_t>(v[Lane]));
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar and widen
    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul_acc(
            a,
            cast<result_t>(v),
            Vec<N, result_t>::load(static_cast<result_t>(c)),
            op
        );
    }

    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul_acc(
            a,
            cast<result_t>(v),
            Vec<N, result_t>::load(static_cast<result_t>(c)),
            op
        );
    }
// !MARK

// MARK: Fused multiply-accumulate by scalar
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return fused_mul_acc(
            a,
            b,
            Vec<N, T>::load(c),
            op
        );
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return fused_mul_acc(
            a,
            b,
            Vec<N, T>::load(c),
            op
        );
    }
// !MARK
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_MUL_HPP
