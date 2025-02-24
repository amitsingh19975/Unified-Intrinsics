#ifndef AMT_ARCH_EMUL_SHIFT_HPP
#define AMT_ARCH_EMUL_SHIFT_HPP

#include "cast.hpp"
#include <concepts>
#include <type_traits>

namespace ui::emul {

// MARK: Left Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            return static_cast<T>(v_ << s_);
        }, v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE static constexpr auto shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return static_cast<T>(v_ << Shift);
        }, v);
    }
// !MARK

// MARK: Saturating Left Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            if constexpr (sizeof(T) == 8) {
                static constexpr auto max = std::numeric_limits<T>::max();
                using type = std::make_unsigned_t<T>;
                if (static_cast<type>(max) - (type(1) << s_) >= v_) return static_cast<T>(v_ << s_);
                return max;
            } else {
                auto temp = static_cast<std::int64_t>(v_) << s_;
                static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
                static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
                return static_cast<T>(std::clamp(temp, min, max));
            }
        }, v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE static constexpr auto sat_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (sizeof(T) == 8) {
                static constexpr auto max = std::numeric_limits<T>::max();
                using type = std::make_unsigned_t<T>;
                if (static_cast<type>(max) - (type(1) << Shift) >= v_) return static_cast<T>(v_ << Shift);
                return max;
            } else {
                auto temp = static_cast<std::int64_t>(v_) << Shift;
                static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
                static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
                return static_cast<T>(std::clamp(temp, min, max));
            }
        }, v);
    }
// !MARK

// MARK: Vector rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            if (s_ < 0) {
                auto temp = static_cast<std::int64_t>(v_);
                auto shift = -s_;
                temp += (1 << (shift - 1));
                return static_cast<T>(temp >> shift);
            }
            return static_cast<T>(v_ >> s_);
        }, v, s);
    }
// !MARK

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            if (s_ < 0) {
                auto temp = static_cast<std::int64_t>(v_);
                static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
                static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
                auto shift = -s_;
                temp += (1 << (shift - 1));
                return static_cast<T>(std::clamp(temp >> shift, min, max));
            }
            if constexpr (sizeof(T) == 8) {
                static constexpr auto max = std::numeric_limits<T>::max();
                using type = std::make_unsigned_t<T>;
                if (static_cast<type>(max) - (type(1) << s_) >= v_) return static_cast<T>(v_ << s_);
                return max;
            } else {
                auto temp = static_cast<std::int64_t>(v_) << s_;
                static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
                static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
                return static_cast<T>(std::clamp(temp, min, max));
            }
        }, v, s);
    }
// !MARK

// MARK: Vector shift left and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE static constexpr auto widening_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([](auto v_) { return static_cast<result_t>(static_cast<result_t>(v_) << Shift); }, v);
    }
// !MARK

// MARK: Vector shift left and insert
    /**
     * @brief It inserts 'Shift' amount of LSB of 'a' into 'b' shifted by 'Shift'.
     * @code
     * (b << Shift) | (a & ((1 << (Shift + 1)) - 1))
     * @codeend
     * @tparam Shift amount of shift
     * @param a masked LSB will be inserted into 'b'
     * @param b will be shifted by 'Shift'
    */
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE static constexpr auto insert_shift_left(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return map([](auto a_, auto b_) {
            if constexpr (Shift + 1 == sizeof(T) * 8) {
                static constexpr T mask = static_cast<T>(~T(0));
                return static_cast<T>((b_ << Shift) | (a_ & mask));
            } else {
                static constexpr T mask = static_cast<T>((std::size_t(1) << (Shift + 1)) - 1); 
                return static_cast<T>((b_ << Shift) | (a_ & mask));
            }
        }, a, b);
    }
// !MARK

// MARK: Right Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            return static_cast<T>(v_ >> s_);
        }, v, s);
    }
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return static_cast<T>(v_ >> Shift);
        }, v);
    }
// !MARK

// MARK: Vector rounding shift left
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto rounding_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            auto temp = static_cast<std::int64_t>(v_);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return static_cast<T>(temp >> Shift);
        }, v);
    }
// !MARK

// MARK: Vector rounding shift right and accumulate
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto rounding_shift_right_accumulate(
        Vec<N, T> const& a,
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto a_, auto v_) {
            auto temp = static_cast<std::int64_t>(v_);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return static_cast<T>(a_ + static_cast<T>(temp >> Shift));
        }, a, v);
    }
// !MARK

// MARK: Vector shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE static constexpr auto narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        return map([](auto v_) {
            return static_cast<result_t>(v_ >> Shift);
        }, v);
    }
// !MARK

// MARK: Vector saturating shift right and narrow   
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        return map([](auto v_) {
            auto temp = static_cast<std::int64_t>(v_) >> Shift;
            static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<result_t>::max());
            static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<result_t>::min());
            return static_cast<result_t>(std::clamp(temp, min, max));
        }, v);
    }
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = std::make_unsigned_t<internal::narrowing_result_t<T>>;
        return map([](auto v_) {
            auto temp = static_cast<std::uint64_t>(v_) >> Shift;
            static constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<result_t>::max());
            static constexpr auto min = static_cast<std::uint64_t>(std::numeric_limits<result_t>::min());
            return static_cast<result_t>(std::clamp(temp, min, max));
        }, v);
    }
// !MARK

// MARK: Vector saturating rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        return map([](auto v_) {
            static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
            static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
            auto temp = static_cast<std::int64_t>(v_);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return static_cast<result_t>(std::clamp(temp >> Shift, min, max));
        }, v);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_rounding_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = std::make_unsigned_t<internal::narrowing_result_t<T>>;
        return map([](auto v_) {
            static constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<T>::max());
            static constexpr auto min = static_cast<std::uint64_t>(std::numeric_limits<T>::min());
            auto temp = static_cast<std::uint64_t>(v_);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return static_cast<result_t>(std::clamp(temp >> Shift, min, max));
        }, v);
    }
// !MARK

// MARK: Vector rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        return map([](auto v_) {
            auto temp = static_cast<std::int64_t>(v_);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return static_cast<result_t>(temp >> Shift);
        }, v);
    }
// !MARK

// MARK: Vector shift right and insert
    /**
     * @brief It inserts 'Shift' amount of MSB of 'a' into 'b' shifted by 'Shift'.
     * @code
     * (b >> Shift) | (a & ((~T(0) << (sizeof(T) * 8 - Shift))))
     * @codeend
     * @tparam Shift amount of shift
     * @param a masked MSB will be inserted into 'b'
     * @param b will be shifted by 'Shift'
    */
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift <= (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto insert_shift_right(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return map([](auto a_, auto b_) {
            static constexpr T mask = static_cast<T>(
                Shift != sizeof(T) * 8
                    ? (~T(0) << (sizeof(T) * 8 - Shift))
                    : 0
            );
            return static_cast<T>((b_ >> Shift) | (a_ & mask));
        }, a, b);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SHIFT_HPP
