#ifndef AMT_ARCH_EMUL_BIT_HPP
#define AMT_ARCH_EMUL_BIT_HPP

#include "cast.hpp"
#include <bit>
#include <type_traits>

namespace ui::emul {

// MARK: Count leading sign bits
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto count_leading_sign_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
	constexpr auto helper = [](T val) -> T {
            auto tmp = static_cast<std::make_signed_t<T>>(val);
            auto t0 = static_cast<std::make_unsigned_t<T>>(val);
            auto v0 = static_cast<std::make_unsigned_t<T>>(tmp < 0 ? ~t0 : t0);
            return static_cast<T>(std::countl_zero(v0) - 1);
	};
	return map([helper](auto v_) { return helper(v_); }, v);
    }
// !MARK

// MARK: Count leading zeros
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto count_leading_zeros(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
	return map([](auto v_) {
	    return static_cast<T>(std::countl_zero(static_cast<std::make_unsigned_t<T>>(v_)));
	}, v);
    }
// !MARK

// MARK: Population Count
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto popcount(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
	return map([](auto v_) {
	    return static_cast<T>(std::popcount(static_cast<std::make_unsigned_t<T>>(v_)));
	}, v);
    }
// !MARK

// MARK: Bitwise clear
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_clear(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
	return map([](auto a_, auto b_){
	    return static_cast<T>(a_ & (~b_));
	}, a, b);
    }
// !MARK

// MARK: Bitwise select
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T> {
	return map([](auto a_, auto b_, auto c_) {
            static constexpr auto max = std::numeric_limits<mask_inner_t<T>>::max();
	    return static_cast<T>(a_ == max ? b_ : c_);
	}, cond, true_, false_);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_BIT_HPP
