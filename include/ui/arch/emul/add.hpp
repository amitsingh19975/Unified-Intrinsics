#ifndef AMT_UI_ARCH_EMUL_ADD_HPP
#define AMT_UI_ARCH_EMUL_ADD_HPP

#include "cast.hpp"
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ui::emul {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
		return map([](auto l, auto r) { return static_cast<T>(l + r); }, lhs, rhs);
	}

// MARK: Widening Addition
    template <std::size_t N, std::integral T, std::integral U>
    UI_ALWAYS_INLINE static constexpr auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
		using result_t = internal::widening_result_t<T, U>;
		return map([](auto l, auto r) { 
			return static_cast<result_t>(l) + static_cast<result_t>(r);
		}, lhs, rhs);
	}
// !MAKR

// MARK: Halving Widening Addition
    template <bool Round, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
		using acc_t = internal::widening_result_t<T>;
		return map([](auto l, auto r) {
			return internal::halving_round_helper<Round, acc_t>(l, r, op::add_t{});
		}, lhs, rhs); 
	}
// !MAKR

// MARK: High-bit Narrowing Addition
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto high_narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::narrowing_result_t<T>> {
		using result_t = internal::narrowing_result_t<T>;
		return map([](auto l, auto r) {
			return (static_cast<result_t>((l + r) >> (sizeof(result_t) * 8)));
		}, lhs, rhs); 
	}
// !MAKR

// MARK: Saturation Addition
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
	) noexcept -> Vec<N, T> {
		using type = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
		auto sum = static_cast<std::int64_t>(lhs) + static_cast<std::int64_t>(rhs);
		static constexpr auto min = static_cast<type>(std::numeric_limits<T>::min());
		static constexpr auto max = static_cast<type>(std::numeric_limits<T>::max());
		return static_cast<T>(
			std::clamp<type>(sum, min, max)
		);
	}
// !MAKR

// MARK: Pairwise Addition
    template <std::size_t N, std::integral T>
		requires (N != 1)
    UI_ALWAYS_INLINE static constexpr auto padd(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
	) noexcept -> Vec<N, T> {
		if constexpr (N == 2) {
			return { lhs[0] + lhs[1], rhs[0] + rhs[1] };
		} else {
			return join(
				padd(lhs.lo, rhs.lo),
				padd(lhs.hi, rhs.hi)
			);
		}
	}

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto padd(
        Vec<N, T> const& v
	) noexcept -> T {
		if constexpr (N == 1) return v.val;
		else return padd(v.lo) + padd(v.hi);
	}
// !MAKR

// MARK: Pairwise Addition
    template <std::size_t N, std::integral T>
		requires (N == 1)
    UI_ALWAYS_INLINE static constexpr auto widening_padd(
        Vec<N, T> const& v
	) noexcept {
        using result_t = internal::widening_result_t<T>;
		using ret_t = Vec<N / 2, result_t>;

		if constexpr (N == 2) {
			return ret_t{ 
                .val = static_cast<result_t>(v.lo) + static_cast<result_t>(v.hi)
			 };
		} else {
			return join(
				widening_padd(v.lo),
				widening_padd(v.hi)
			);
		}
	}

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto widening_padd(
        Vec<    N, internal::widening_result_t<T>> const& x,
		Vec<2 * N, T> v
	) noexcept -> Vec<N, internal::widening_result_t<T>> {
		using result_t = internal::widening_result_t<T>;
		if constexpr (N == 1) {
			return {
				.val = x.val + static_cast<result_t>(v.lo.val) + static_cast<result_t>(v.hi.val)	
			};
		} else {
            return join(
                widening_padd(x.lo, v.lo),
                widening_padd(x.hi, v.hi)
            );
		}
	}
// !MAKR

// MARK: Addition across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::add_t op
    ) noexcept -> T {
		constexpr auto helper = []<std::size_t... Is>(
			std::index_sequence<Is...>,
			Vec<N, T> const& v
		) -> T {
			return static_cast<T>((v[Is] +...));
		};
		return helper(std::make_index_sequence<N>{}, v);
	}

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::add_t op
    ) noexcept -> T {
		using result_t = internal::widening_result_t<T>;
		constexpr auto helper = []<std::size_t... Is>(
			std::index_sequence<Is...>,
			Vec<N, T> const& v
		) -> result_t {
			return static_cast<result_t>((static_cast<result_t>(v[Is]) +...));
		};
		return helper(std::make_index_sequence<N>{}, v);
	}
// !MAKR
} // namespace ui::emul

#endif // AMT_UI_ARCH_EMUL_ADD_HPP
