#ifndef AMT_ARCH_EMUL_ABS_HPP
#define AMT_ARCH_EMUL_ABS_HPP

#include "cast.hpp"

namespace ui::emul {

// MARK: Difference
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
	return map([](auto l, auto r) {
	    return static_cast<T>(
		l > r ? (l - r) : (r - l)
	    );  
	}, lhs, rhs);
    }
// !MARK

// MARK: Widening Difference
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
	using result_t = internal::widening_result_t<T>;
	
	return map([](auto l, auto r) {
	    return static_cast<result_t>(
		l > r ? (l - r) : (r - l)
	    );  
	}, lhs, rhs);
    }
// !MARK

// MARK: Absolute difference and Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
	return map([](auto a, auto l, auto r){
	    return static_cast<T>(a + (l > r ? l - r : r - l));
	}, acc, lhs, rhs);
    }
// !MARK

// MARK: Absolute Value
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
	using std::abs;
	using ui::abs;
	return map([](auto v) {
	    return static_cast<T>(abs(v));
	}, v);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
	return map([](auto v) {
            using type = std::conditional_t<std::is_signed_v<T>, std::int16_t, std::uint64_t>;
            static constexpr auto min = static_cast<type>(std::numeric_limits<T>::min());
            static constexpr auto max = static_cast<type>(std::numeric_limits<T>::max());
            return static_cast<T>(std::clamp(std::abs(static_cast<type>(v)), min, max));
	}, v);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_ABS_HPP
