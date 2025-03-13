#ifndef AMT_UI_VEC_OP_HPP
#define AMT_UI_VEC_OP_HPP

#include "base_vec.hpp"
#include "arch/arch.hpp"
#include "ui/arch/arm/mul.hpp"

namespace ui {
    template <std::size_t N, typename T>
    inline constexpr auto Vec<N, T>::load(T val) noexcept -> Vec<N, T> {
        return ui::load<N, T>(val);
    }

    template <std::size_t N, typename T>
    template <unsigned Lane, std::size_t M>
    inline constexpr auto Vec<N, T>::load(Vec<M, T> const& v) noexcept -> Vec<N, T> {
        return ui::load<N, Lane>(v);
    }
} // namespace ui

// MARK: not
template <std::size_t N, std::integral T>
    requires (!std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator!(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_not(op);
}

template <std::size_t N, std::integral T>
    requires (!std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator~(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_not(op);
}
// !MARK

// MARK: ==
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator==(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::equal_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator==(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    return lhs == ui::Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator==(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    return rhs == lhs;
}
// !MARK

// MARK: !=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator!=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return !cmp(lhs, rhs, op::equal_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator!=(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    return lhs != ui::Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator!=(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    return !(lhs == rhs);
}
// !MARK

// MARK: <=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::less_equal_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator<=(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs <= Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator<=(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) <= rhs;
}
// !MARK

// MARK: <
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::less_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator<(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs < Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator<(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) < rhs;
}
// !MARK

// MARK: >=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::greater_equal_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator>=(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs >= Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator>=(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) >= rhs;
}
// !MARK

// MARK: >
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::greater_t{});
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator>(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs > Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator>(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) > rhs;
}
// !MARK

// MARK: - (negate)
template <std::size_t N, typename T>
    requires (std::floating_point<T> || std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return negate(op);
}
// !MARK

// MARK: ^
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator^(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_xor(lhs, rhs);
}

template <std::size_t N, std::integral T, std::integral U>
UI_ALWAYS_INLINE constexpr auto operator^(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs ^ Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, std::integral T, std::integral U>
UI_ALWAYS_INLINE constexpr auto operator^(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) ^ rhs;
}
// !MARK

// MARK: +
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator+(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return add(lhs, rhs);
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator+(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs + Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator+(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) + rhs;
}
// !MARK

// MARK: -
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return sub(lhs, rhs);
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs - Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator-(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) - rhs;
}
// !MARK

// MARK: *
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator*(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return mul(lhs, rhs);
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator*(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs * Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator*(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) * rhs;
}
// !MARK


// MARK: /
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator/(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return div(lhs, rhs);
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator/(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs / Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator/(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) / rhs;
}
// !MARK

// MARK: &
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator&(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_and(lhs, rhs);
}

template <std::size_t N, std::integral T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator&(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs & Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, std::integral T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator&(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) & rhs;
}
// !MARK

// MARK: |
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator|(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_or(lhs, rhs);
}

template <std::size_t N, std::integral T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator|(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs | Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, std::integral T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator|(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) | rhs;
}
// !MARK

// MARK: <<
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator<<(ui::Vec<N, T> const& lhs, ui::Vec<N, std::make_unsigned_t<T>> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return shift_left(lhs, rhs);
}

template <std::size_t N, std::integral T, std::convertible_to<std::make_unsigned_t<T>> U>
UI_ALWAYS_INLINE constexpr auto operator<<(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    using type = std::make_unsigned_t<T>;
    return lhs << Vec<N, T>::load(static_cast<type>(rhs));
}

template <std::size_t N, std::integral T, std::convertible_to<std::make_unsigned_t<T>> U>
UI_ALWAYS_INLINE constexpr auto operator<<(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    using type = std::make_unsigned_t<T>;
    return Vec<N, T>::load(static_cast<type>(lhs)) << rhs;
}
// !MARK

// MARK: >>
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator>>(ui::Vec<N, T> const& lhs, ui::Vec<N, std::make_unsigned_t<T>> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return shift_right(lhs, rhs);
}

template <std::size_t N, std::integral T, std::convertible_to<std::make_unsigned_t<T>> U>
UI_ALWAYS_INLINE constexpr auto operator>>(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    using type = std::make_unsigned_t<T>;
    return lhs >> Vec<N, T>::load(static_cast<type>(rhs));
}

template <std::size_t N, std::integral T, std::convertible_to<std::make_unsigned_t<T>> U>
UI_ALWAYS_INLINE constexpr auto operator>>(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    using type = std::make_unsigned_t<T>;
    return Vec<N, T>::load(static_cast<type>(lhs)) >> rhs;
}
// !MARK

// MARK: %
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator%(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return rem(lhs, rhs);
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator%(ui::Vec<N, T> const& lhs, U const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs % Vec<N, T>::load(static_cast<T>(rhs));
}

template <std::size_t N, typename T, std::convertible_to<T> U>
UI_ALWAYS_INLINE constexpr auto operator%(U const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(static_cast<T>(lhs)) % rhs;
}
// !MARK


namespace ui {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto if_then_else(
        mask_t<N, T> const& cond,
        Vec<N, T> const& then,
        Vec<N, T> const& else_
    ) noexcept -> Vec<N, T> {
        return bitwise_select(cond, then, else_);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto any(
        Vec<N, T> const& v
    ) noexcept -> bool {
        return fold(rcast<std::make_unsigned_t<T>>(v) > 0, op::max_t{}) > 0;
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto all(
        Vec<N, T> const& v
    ) noexcept -> bool {
        return fold(rcast<std::make_unsigned_t<T>>(v) > 0, op::min_t{}) > 0;
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto ceil(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return round<std::float_round_style::round_toward_infinity>(v); 
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto floor(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return round<std::float_round_style::round_toward_neg_infinity>(v); 
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto trunc(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return round<std::float_round_style::round_toward_zero>(v); 
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto frac(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return v - floor(v);
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE static constexpr auto div255(Vec<N, std::uint16_t> const& v) noexcept -> Vec<N, std::uint8_t> {
        return cast<uint8_t>((v + 127) / 255);
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE static constexpr auto approx_scale(Vec<N, std::uint8_t> const& x, Vec<N, std::uint8_t> const& y) noexcept -> Vec<N, std::uint8_t> {
        auto X = cast<std::uint16_t>(x);
        auto Y = cast<std::uint16_t>(y);
        auto tmp = fused_mul_acc(X, X, Y); // X + X * Y
        return cast<std::uint8_t>(tmp / 256);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto dot(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> T {
        return fold(x * y, op::add_t{});
    }

    template <typename T>
    UI_ALWAYS_INLINE static constexpr auto cross(
        Vec<2, T> const& x,
        Vec<2, T> const& y
    ) noexcept -> T {
        auto res = x * shuffle<1,0>(y);
        return res[0] - res[1];
    }

    template <std::size_t N, typename T>
        requires (std::is_floating_point_v<T>)
    UI_ALWAYS_INLINE static constexpr auto isfinite(
        Vec<N, T> const& v
    ) noexcept -> bool {
        return std::isfinite(dot(v, Vec<N, T>::load(0)));
    }

    using float2  = Vec< 2, float>;
    using float4  = Vec< 4, float>;
    using float8  = Vec< 8, float>;

    using half2  = Vec< 2, float16>;
    using half4  = Vec< 4, float16>;
    using half8  = Vec< 8, float16>;

    using bhalf2  = Vec< 2, bfloat16>;
    using bhalf4  = Vec< 4, bfloat16>;
    using bhalf8  = Vec< 8, bfloat16>;

    using double2 = Vec< 2, double>;
    using double4 = Vec< 4, double>;
    using double8 = Vec< 8, double>;

    using byte2   = Vec< 2,  std::uint8_t>;
    using byte4   = Vec< 4,  std::uint8_t>;
    using byte8   = Vec< 8,  std::uint8_t>;
    using byte16  = Vec< 16, std::uint8_t>;

    using int2    = Vec< 2, std::int32_t>;
    using int4    = Vec< 4, std::int32_t>;
    using int8    = Vec< 8, std::int32_t>;

    using ushort2 = Vec< 2, std::uint16_t>;
    using ushort4 = Vec< 4, std::uint16_t>;
    using ushort8 = Vec< 8, std::uint16_t>;

    using uint2   = Vec< 2, std::uint32_t>;
    using uint4   = Vec< 4, std::uint32_t>;
    using uint8   = Vec< 8, std::uint32_t>;

    using long2   = Vec< 2, std::int64_t>;
    using long4   = Vec< 4, std::int64_t>;
    using long8   = Vec< 8, std::int64_t>;

    namespace native {

#if UI_NATIVE_SIZE == 16 || !defined(UI_NATIVE_SIZE)
        static constexpr std::size_t NativeSizeFactor = 1ul;
#elif UI_NATIVE_SIZE == 32
        static constexpr std::size_t NativeSizeFactor = 2ul;
#elif UI_NATIVE_SIZE == 64
        static constexpr std::size_t NativeSizeFactor = 4ul;
#endif
        using f16  = Vec< 8 * NativeSizeFactor, float16>;
        using bf16 = Vec< 8 * NativeSizeFactor, bfloat16>;
        using f32  = Vec< 4 * NativeSizeFactor, float>;
        using f64  = Vec< 2 * NativeSizeFactor, double>;
        using u8   = Vec<16 * NativeSizeFactor, std::uint8_t>;
        using u16  = Vec< 8 * NativeSizeFactor, std::uint16_t>;
        using u32  = Vec< 4 * NativeSizeFactor, std::uint32_t>;
        using u64  = Vec< 2 * NativeSizeFactor, std::uint64_t>;
        using i8   = Vec<16 * NativeSizeFactor, std::int8_t>;
        using i16  = Vec< 8 * NativeSizeFactor, std::int16_t>;
        using i32  = Vec< 4 * NativeSizeFactor, std::int32_t>;
        using i64  = Vec< 2 * NativeSizeFactor, std::int64_t>;

    } // namespace native
}

#endif // AMT_UI_VEC_OP_HPP
