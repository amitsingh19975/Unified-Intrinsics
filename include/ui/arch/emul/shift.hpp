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
    namespace internal {
        using namespace ::ui::internal;

        template <std::integral T>
        UI_ALWAYS_INLINE static constexpr auto sat_shift_left_helper(
            T v,
            std::make_unsigned_t<T> s
        ) noexcept -> T {
            static constexpr auto bits = sizeof(T) * 8;
            if constexpr (std::is_signed_v<T>) {
                static constexpr auto lane = bits - 1;
                using utype = std::make_unsigned_t<T>;
                T const limit = T(1) << (lane - s);
                return static_cast<T>(
                        (v >= limit || v <= -limit)
                            ? static_cast<T>(static_cast<T>(static_cast<utype>(v) >> lane) + (T(1) << lane) - 1)
                            : (v << s)
                    );
            } else {
                static constexpr auto max = std::numeric_limits<T>::max();
                return static_cast<T>((T(1) << (bits - s)) <= v ? max : static_cast<T>(v << s));
            }
        }

        template <std::size_t Shift, std::integral T>
        UI_ALWAYS_INLINE static constexpr auto sat_shift_left_helper(
            T v
        ) noexcept -> T {
            static constexpr auto bits = sizeof(T) * 8;
            if constexpr (std::is_signed_v<T>) {
                static constexpr auto lane = bits - 1;
                using utype = std::make_unsigned_t<T>;
                static constexpr T limit = T(1) << (lane - Shift);
                return static_cast<T>(
                        (v >= limit || v <= -limit)
                            ? static_cast<T>(static_cast<T>(static_cast<utype>(v) >> lane) + (T(1) << lane) - 1)
                            : (v << Shift)
                    );
            } else {
                static constexpr auto max = std::numeric_limits<T>::max();
                return static_cast<T>((T(1) << (bits - Shift)) <= v ? max : static_cast<T>(v << Shift));
            }
        }
    }
    
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            return internal::sat_shift_left_helper(v_, s_);
        }, v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE static constexpr auto sat_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return internal::sat_shift_left_helper<Shift>(v_);
        }, v);
    }
// !MARK

// MARK: Vector rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            return static_cast<T>(v_ << s_);
        }, v, s);
    }
// !MARK

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return sat_shift_left(v, s);
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
     * @code
     * mask = (1 << Shift) - 1
     * (a & mask) | (b << Shift) & ~mask
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
            static constexpr auto mask = static_cast<std::make_unsigned_t<T>>(T(1) << Shift) - 1; 
            auto a0 = static_cast<T>(a_ & mask);
            auto b0 = static_cast<T>((b_ << Shift) & ~mask);
            return static_cast<T>(a0 | b0);
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
    
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            if constexpr (std::is_signed_v<T>) {
                return static_cast<T>(s_ >= (sizeof(T) * 8 - 1) ? T(-1) : (v_ >> s_));
            } else {
                return static_cast<T>(v_ >> s_);
            }
        }, v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE static constexpr auto sat_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (std::is_signed_v<T>) {
                return static_cast<T>(Shift >= (sizeof(T) * 8 - 1) ? T(-1) : (v_ >> Shift));
            } else {
                return static_cast<T>(v_ >> Shift);
            }
        }, v);
    }
// !MARK

// MARK: Vector rounding shift right
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            auto temp = static_cast<std::int64_t>(v_);
            auto shift = s_;
            temp += (1 << (shift - 1));
            return static_cast<T>(temp >> shift);
        }, v, s);
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

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return map([](auto v_, auto s_) {
            auto const round_bit = (1 << (s_ - 1));
            #ifndef UI_HAS_INT128
            if constexpr (sizeof(T) == 8) {
                if (v_ == 0) return T(0);
                constexpr unsigned lane = sizeof(T) * 8 - 1;
                return static_cast<T>(s_ > lane
                    ? 0
                    : (v_ >> s_) + ((v_ & round_bit) >> (s_ - 1)));
            } else 
            #endif
            {
                #ifdef UI_HAS_INT128
                using type64 = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
                using type128 = std::conditional_t<std::is_signed_v<T>, ui::int128_t, ui::uint128_t>;
                using type = std::conditional_t<sizeof(T) == 8, type128, type64>;
                #else
                using type = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
                #endif
                auto temp = static_cast<type>(v_) >> s_;
                temp += (v_ & round_bit) >> (s_ - 1);
                return static_cast<T>(temp);
            }
        }, v, s);
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
        requires (Shift > 0 && Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto insert_shift_right(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return map([](auto a_, auto b_) {
            using utype = std::make_unsigned_t<T>;
            static constexpr auto bits = sizeof(T) * 8;
            static constexpr auto adj_shift = bits - Shift;
            static constexpr utype mask = (utype(1) << adj_shift) - 1;
            auto ta = (static_cast<utype>(a_) >> adj_shift) << adj_shift;
            utype tb = static_cast<utype>(b_) >> Shift;
            auto res = static_cast<T>(ta | tb);
            return res;
        }, a, b);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SHIFT_HPP
