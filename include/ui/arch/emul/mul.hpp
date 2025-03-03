#ifndef AMT_ARCH_EMUL_MUL_HPP
#define AMT_ARCH_EMUL_MUL_HPP

#include "cast.hpp"
#include "ui/arch/basic.hpp"
#include "ui/base.hpp"
#include <concepts>

namespace ui::emul {

// MARK: Multiplication
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return static_cast<T>(l * r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Multiply-Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, T> {
        return map([](auto a, auto l, auto r) {
            return static_cast<T>(a + (l * r));
        }, acc, lhs, rhs);
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([](auto a, auto l, auto r) -> result_t {
            return static_cast<result_t>(a + (static_cast<result_t>(l) * static_cast<result_t>(r)));
        }, acc, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, T> {
        return map([](auto a, auto l, auto r) {
            return static_cast<T>(a - (l * r));
        }, acc, lhs, rhs);
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([](auto a, auto l, auto r) -> result_t {
            return static_cast<result_t>(a - (static_cast<result_t>(l) * static_cast<result_t>(r)));
        }, acc, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(acc, rhs, lhs, op);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(acc, a, Vec<N, T>::load(v[Lane]), op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(acc, rhs, lhs, op);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(acc, a, Vec<N, T>::load(v[Lane]), op);
    }
// !MARK

// MARK: Widening Multiplication
    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE static constexpr auto widening_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([](auto l, auto r) -> result_t {
            return static_cast<result_t>(l) * static_cast<result_t>(r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, T> {
        return map([c](auto a_, auto l) {
            return static_cast<T>(a_ + (l * c));
        }, a, b);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(a, b, v[Lane], op);
    }
// !MARK

// MARK: Vector multiply-subtract by scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, T> {
        return map([c](auto a_, auto l) {
            return static_cast<T>(a_ - (l * c));
        }, a, b);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(a, b, v[Lane], op);
    }
// !MARK

// MARK: Multiplication with scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, T> {
        return map([c](auto v_) {
            return static_cast<T>(v_ * c);
        }, v);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return mul(a, v[Lane]);
    }
// !MARK

// MARK: Multiplication with scalar and widen
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto widening_mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([c](auto v_) -> result_t {
            return static_cast<result_t>(static_cast<result_t>(v_) * static_cast<result_t>(c));
        }, v);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE static constexpr auto widening_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept {
        return widening_mul(a, v[Lane]);
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar and widen
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([c](auto a_, auto v_) -> result_t {
            return static_cast<result_t>(a_ + static_cast<result_t>(v_) * static_cast<result_t>(c));
        }, a, v);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([c](auto a_, auto v_) -> result_t {
            return static_cast<result_t>(a_ - static_cast<result_t>(v_) * static_cast<result_t>(c));
        }, a, v);
    }
// !MARK

// MARK: Fused multiply-accumulate by scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(a, b, c, op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return mul_acc(a, b, c, op);
    }
// !MARK

} // namespace ui::emul

#endif // AMT_ARCH_EMUL_MUL_HPP
