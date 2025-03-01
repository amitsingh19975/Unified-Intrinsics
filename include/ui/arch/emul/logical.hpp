#ifndef AMT_ARCH_EMUL_LOGICAL_HPP
#define AMT_ARCH_EMUL_LOGICAL_HPP

#include "cast.hpp"
#include <type_traits>

namespace ui::emul {

// MARK: Negation
    template <std::size_t N, typename T>
        requires (std::is_floating_point_v<T> || std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) { return static_cast<T>(-v_); }, v);
    }
// !MARK

    template <std::size_t N, std::integral T>
        requires (std::is_signed_v<T>)
    UI_ALWAYS_INLINE static constexpr auto sat_negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) -> T { 
            static constexpr auto min = std::numeric_limits<T>::min();
            static constexpr auto max = std::numeric_limits<T>::max();
            return static_cast<T>((v_ == min) ? max : -v_);
        }, v);
    }

// MARK: Bitwise Not
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_not(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) { return static_cast<T>(~v_); }, v); 
    }
// !MARK

// MARK: Bitwise Not
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_and(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>(l & r); }, lhs, rhs);
    }
// !MARK

// MARK: Bitwise OR
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_or(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>(l | r); }, lhs, rhs);
    }
// !MARK

// MARK: Bitwise XOR
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>(l ^ r); }, lhs, rhs);
    }
// !MARK

// MARK: Bitwise Or-Not lhs | ~rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_ornot(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>(l | (~r)); }, lhs, rhs);
    }
// !MARK

// MARK: Bitwise And-Not ~lhs & rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto bitwise_notand(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>((~l) & r); }, lhs, rhs);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_LOGICAL_HPP
