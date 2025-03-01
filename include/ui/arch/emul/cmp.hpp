#ifndef AMT_ARCH_EMUL_CMP_HPP
#define AMT_ARCH_EMUL_CMP_HPP

#include "cast.hpp"
#include <bit>
#include <limits>
#include <type_traits>
#include "abs.hpp"
#include "ui/base.hpp"

namespace ui::emul {

// MARK: Bitwise equal and 'and' test
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return l == r ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }

    /**
     * @return (lhs & rhs) == 0
     */
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::and_test_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return (l & r) ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }
// !MARK

// MARK: Greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return l >= r ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto v_) {
            return (v_ >= T{0}) ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, v);
    }
// !MARK

// MARK: Less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::less_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return l <= r ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto v_) {
            return (v_ <= T{0}) ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, v);
    }
// !MARK

// MARK: Greater Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::greater_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return l > r ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::greater_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto v_) {
            return (v_ > T{0}) ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, v);
    }
// !MARK

// MARK: Less Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::less_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto l, auto r) {
            return l < r ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& v,
        [[maybe_unused]] op::less_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;

        return map([](auto v_) {
            return (v_ < T{0}) ? std::numeric_limits<result_t>::max(): result_t{}; 
        }, v);
    }
// !MARK

// MARK: Absolute greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::greater_equal_t{}
            );
        } else {
            using ui::abs;
            using std::abs;
            return map([](auto l, auto r) {
                return (abs(l) >= abs(r)) ? std::numeric_limits<result_t>::max() : result_t{};
            }, lhs, rhs);
        }
    }
// !MARK

// MARK: Absolute less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::less_equal_t{}
            );
        } else {
            using ui::abs;
            using std::abs;
            return map([](auto l, auto r) {
                return (abs(l) <= abs(r)) ? std::numeric_limits<result_t>::max() : result_t{};
            }, lhs, rhs);
        }
    }
// !MARK

// MARK: Absolute greater than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_greater_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::greater_t{}
            );
        } else {
            using ui::abs;
            using std::abs;
            return map([](auto l, auto r) {
                return (abs(l) > abs(r)) ? std::numeric_limits<result_t>::max() : result_t{};
            }, lhs, rhs);
        }
    }
// !MARK

// MARK: Absolute greater than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::abs_less_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        if constexpr (std::integral<T>) {
            return cmp(
                sat_abs(lhs),
                sat_abs(rhs),
                op::less_t{}
            );
        } else {
            using ui::abs;
            using std::abs;
            return map([](auto l, auto r) {
                return (abs(l) < abs(r)) ? std::numeric_limits<result_t>::max() : result_t{};
            }, lhs, rhs);
        }
    }
// !MARK

} // namespace ui::emul

#endif // AMT_ARCH_EMUL_CMP_HPP
