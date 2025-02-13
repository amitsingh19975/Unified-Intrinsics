#ifndef AMT_ARCH_EMUL_RECIPROCAL_HPP
#define AMT_ARCH_EMUL_RECIPROCAL_HPP

#include "cast.hpp"
#include <concepts>
#include "../../modular_inv.hpp"

namespace ui::emul {

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return maths::BinaryReciprocal{}.estimate(v_);
        }, v);
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto reciprocal_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto e_) {
            return maths::internal::calculate_reciprocal(v_, e_);
        }, v, e);
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto sqrt_inv_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return maths::BinaryReciprocal{}.sqrt_inv(v_);
        }, v);
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto sqrt_inv_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto e_) {
            return maths::internal::calculate_sqrt_inv(v_, e_);
        }, v, e);
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto exponent_reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            auto fp = fp::decompose_fp(v_);
            return maths::BinaryReciprocal{}.estimate(T(1) << fp.exponent);
        }, v);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_RECIPROCAL_HPP
