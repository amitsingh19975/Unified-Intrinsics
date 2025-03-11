#ifndef AMT_ARCH_EMUL_RECIPROCAL_HPP
#define AMT_ARCH_EMUL_RECIPROCAL_HPP

#include "cast.hpp"
#include "../../modular_inv.hpp"

namespace ui::emul {

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (std::integral<T>) {
                return maths::BinaryReciprocal{}.iestimate(v_);
            } else {
                return maths::BinaryReciprocal{}.estimate(v_);
            }
        }, v);
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto reciprocal_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        if constexpr (std::integral<T>) {
            return e;
        } else {
            return map([](auto v_, auto e_) {
                return maths::internal::calculate_reciprocal(v_, e_);
            }, v, e);
        }
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto sqrt_inv_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (std::integral<T>) {
                return maths::BinaryReciprocal{}.isqrt_inv(v_);
            } else {
                return maths::BinaryReciprocal{}.sqrt_inv(v_);
            }
        }, v);
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto sqrt_inv_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto e_) {
            if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return T(maths::internal::calculate_sqrt_inv(float(v_), float(e_)));
            } else {
                return maths::internal::calculate_sqrt_inv(v_, e_);
            }
        }, v, e);
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto exponent_reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) -> T {
            if (v_ == 0) v_ = std::numeric_limits<T>::min();
            auto fp = fp::decompose_fp(v_);
            using e_t = decltype(fp.exponent);
            e_t exp = fp.exponent;
            auto res = fp::compose_fp(fp::FloatingPointRep<T>{ .sign = 0, .exponent = exp, .mantissa = 0 });
            auto temp = maths::BinaryReciprocal{}.estimate(res);
            temp = maths::internal::calculate_reciprocal<4, true>(res, temp);
            return static_cast<T>(temp);
        }, v);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_RECIPROCAL_HPP
