#ifndef AMT_ARCH_EMUL_MINMAX_HPP
#define AMT_ARCH_EMUL_MINMAX_HPP

#include "cast.hpp"
#include "ui/base.hpp"
#include <concepts>
#include <utility>

namespace ui::emul {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto max(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return std::max(l, r);
        }, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto min(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return std::min(l, r);
        }, lhs, rhs);
    }

    /**
     * @return number-maximum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto maxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return internal::maxnm(l, r);
        }, lhs, rhs);
    }

    /**
     * @return number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto minnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return internal::minnm(l, r);
        }, lhs, rhs);
    }

// MARK: Pairwise Maximum
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto pmax(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, auto const& l, auto const& r) {
            auto res = Vec<N, T>{};
            ((
                res[Is] = std::max(l[2 * Is], l[2 * Is + 1]),
                res[Is + N / 2] = std::max(r[2 * Is], r[2 * Is + 1])
            ),...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::pmax_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else if constexpr (N == 2) return std::max(v[0], v[1]);
        else return std::max(fold(v.lo, op), fold(v.hi, op));
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto pmaxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, auto const& l, auto const& r) {
            auto res = Vec<N, T>{};
            ((
                res[Is] = internal::maxnm(l[2 * Is], l[2 * Is + 1]),
                res[Is + N / 2] = internal::maxnm(r[2 * Is], r[2 * Is + 1])
            ),...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, lhs, rhs);
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::pmaxnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else if constexpr (N == 2) return internal::maxnm(v[0], v[1]);
        else return internal::maxnm(fold(v.lo, op), fold(v.hi, op));
    }
// !MARK

// MARK: Pairwise Minimum
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto pmin(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, auto const& l, auto const& r) {
            auto res = Vec<N, T>{};
            ((
                res[Is] = std::min(l[2 * Is], l[2 * Is + 1]),
                res[Is + N / 2] = std::min(r[2 * Is], r[2 * Is + 1])
            ),...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, lhs, rhs);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::pmin_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else if constexpr (N == 2) return std::min(v[0], v[1]);
        else return std::min(fold(v.lo, op), fold(v.hi, op));
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto pminnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, auto const& l, auto const& r) {
            auto res = Vec<N, T>{};
            ((
                res[Is] = internal::minnm(l[2 * Is], l[2 * Is + 1]),
                res[Is + N / 2] = internal::minnm(r[2 * Is], r[2 * Is + 1])
            ),...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, lhs, rhs);
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::pminnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else if constexpr (N == 2) return internal::minnm(v[0], v[1]);
        else return internal::minnm(fold(v.lo, op), fold(v.hi, op));
    }
// !MARK

// MARK: Maximum/Maximum across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::max_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else return std::max(fold(v.lo, op), fold(v.hi, op));
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::maxnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else return internal::maxnm(fold(v.lo, op), fold(v.hi, op));
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::min_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else return std::min(fold(v.lo, op), fold(v.hi, op));
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::minnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else return internal::minnm(fold(v.lo, op), fold(v.hi, op));
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_MINMAX_HPP
