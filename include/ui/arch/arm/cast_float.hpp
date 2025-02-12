#ifndef AMT_UI_ARCH_ARM_CAST_FLOAT_HPP
#define AMT_UI_ARCH_ARM_CAST_FLOAT_HPP

#include "cast.hpp"
#include "../../float.hpp"
#include "logical.hpp"
#include "minmax.hpp"
#include "mul.hpp"
#include "load.hpp"
#include "add.hpp"
#include "shift.hpp"
#include "sub.hpp"
#include "bit.hpp"
#include "cmp.hpp"
#include <bit>
#include <cstdint>

namespace ui {

    template <std::size_t N>
    static inline auto cast_float32_to_float16(
        Vec<N, float> const& v
    ) noexcept -> Vec<N, float16> {
        using namespace arm::neon;
        #define I(V) std::bit_cast<Vec<N, std::uint32_t>>(V)
        #define F(V) std::bit_cast<Vec<N, float>>(V)

        auto bits = I(v);
        auto sign = bitwise_and(bits, load<N, std::uint32_t>(0x8000'0000));
        auto abs = min(bitwise_xor(bits, sign), load<N, std::uint32_t>(0x4780'0000));

        auto magic = bitwise_and(
            I(
                max(
                    mul(F(abs), load<N, float>(1 << 13)),
                    load<N, float>(0.5f)
                )
            ),
            load<N, std::uint32_t>(0xff << 23)
        );
        auto rounded = I(add(F(abs), F(magic)));
        auto shifts = load<N, std::uint32_t>((127 - 15 + 13 + 1) << 10);
        auto exp = sub(shift_right<13>(magic), shifts);

        auto f16 = add(rounded, exp);
        return std::bit_cast<Vec<N, float16>>(
            cast<std::uint16_t>(bitwise_or(
                shift_right<16>(sign),
                f16
            ))
        );
        #undef I
        #undef F
    }

    template <std::size_t N>
    static inline constexpr auto cast_float16_to_float32(
        Vec<N, float16> const& v
    ) noexcept -> Vec<N, float> {
        using namespace arm::neon;
        #define I(V) std::bit_cast<Vec<N, std::uint32_t>>(V)
        #define F(V) std::bit_cast<Vec<N, float>>(V)
        auto wide = cast<std::uint32_t>(std::bit_cast<Vec<N, std::uint16_t>>(v));
        auto sign = bitwise_and(wide, load<N, std::uint32_t>(0x8000));
        auto abs = bitwise_xor(wide, sign);
        auto inf_or_nan = bitwise_and(
            cmp(abs, load<N, std::uint32_t>(31 << 10), op::greater_equal_t{}),
            load<N, std::uint32_t>(0xFF << 23)
        );
        auto is_norm = cmp(abs, load<N, std::uint32_t>(0x3FF), op::greater_t{});
        auto sub = I(mul(cast<float>(abs), load<N, float>(1.f / (1 << 24))));
        auto norm = add(
            shift_left<13>(abs),
            load<N, std::uint32_t>((127 - 15) << 23)
        );
        auto finite = bitwise_select(is_norm, norm, sub);
        return F(
            bitwise_or(
                bitwise_or(
                    shift_left<16>(sign),
                    finite
                ),
                inf_or_nan
            )
        );
        #undef I
        #undef F
    }

    template <std::size_t N>
    static inline constexpr auto cast_float32_to_bfloat16(
        Vec<N, float> const& v
    ) noexcept -> Vec<N, bfloat16> {
        using namespace arm::neon;
        auto temp = std::bit_cast<Vec<N, std::uint32_t>>(v);
        auto const m0 = load<N, std::uint32_t>(0x7FFFFF);
        auto shifted = shift_right<16>(temp);
        auto b0 = load<N, std::uint32_t>(0x7FFF);
        auto b1 = add(b0, bitwise_and(temp, load<N, std::uint32_t>(1)));
        return std::bit_cast<Vec<N, bfloat16>>(cast<uint16_t>(bitwise_or(
            shifted,
            shift_right<16>(
                bitwise_and(
                    add(temp, b1),
                    m0
                )
            )
        )));
    }

    template <std::size_t N>
    static inline constexpr auto cast_bfloat16_to_float32(
        Vec<N, bfloat16> const& v
    ) noexcept -> Vec<N, float> {
        using namespace arm::neon;
        auto temp = std::bit_cast<Vec<N, std::uint16_t>>(v);
        auto wide = cast<std::uint32_t>(temp);
        return std::bit_cast<Vec<N, float>>(
            shift_left<16>(wide)
        );
    }

} // namespace ui

#endif // AMT_UI_ARCH_ARM_CAST_FLOAT_HPP
