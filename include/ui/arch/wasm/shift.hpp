#ifndef AMT_UI_ARCH_WASM_SHIFT_HPP
#define AMT_UI_ARCH_WASM_SHIFT_HPP

#include "cast.hpp"
#include "../emul/shift.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

// MARK: Left shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::shift_left(v, s); 
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::shift_left(v, rcast<std::make_unsigned_t<T>>(s)); 
    }

    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::shift_left<Shift>(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto m = to_vec(v);
                if constexpr (sizeof(T) == 1) {
                    auto res = wasm_i8x16_shl(m, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 2) {
                    auto res = wasm_i16x8_shl(m, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 4) {
                    auto res = wasm_i32x4_shl(m, Shift);
                    return from_vec<T>(res);
                } else if constexpr (sizeof(T) == 8) {
                    auto res = wasm_i64x2_shl(m, Shift);
                    return from_vec<T>(res);
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return shift_left<Shift>(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            return join(
                shift_left<Shift, false>(v.lo),
                shift_left<Shift, false>(v.hi)
            );
        }
    }
// !MARK

// MARK: Saturating Left Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_left(v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_left<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::rounding_shift_left(v, s);
    }
// !MARK

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_rounding_shift_left(v, s);
    }
// !MARK

// MARK: Vector shift left and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto vt = cast<result_t>(v);
        return shift_left<Shift>(vt);
    }
// !MARK

// MARK: Right shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::shift_right(v, rcast<std::make_unsigned_t<T>>(s));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::shift_right(v, s);
    }

    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::shift_right<Shift>(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto m = to_vec(v);
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        auto res = wasm_i8x16_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto res = wasm_i16x8_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto res = wasm_i32x4_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto res = wasm_i64x2_shr(m, Shift);
                        return from_vec<T>(res);
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        auto res = wasm_u8x16_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 2) {
                        auto res = wasm_u16x8_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 4) {
                        auto res = wasm_u32x4_shr(m, Shift);
                        return from_vec<T>(res);
                    } else if constexpr (sizeof(T) == 8) {
                        auto res = wasm_u64x2_shr(m, Shift);
                        return from_vec<T>(res);
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return shift_right<Shift>(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            return join(
                shift_right<Shift, false>(v.lo),
                shift_right<Shift, false>(v.hi)
            );
        }
    }
// !MARK

// MARK: Saturating Right Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_shift_right(v, s);
    }
// !MARK

// MARK: Vector rounding shift right
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::rounding_shift_right(v, s);
    }

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <unsigned Shift, bool Merge = true, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(T) * 8;
        auto vt = rcast<std::make_unsigned_t<T>>(v);
        auto mb = shift_right<bits - 1>(
            shift_left<bits - Shift>(vt)
        );
        auto res = shift_right<Shift>(v);
        return add<true>(res, rcast<T>(mb));
    }
// !MARK

// MARK: Vector saturating rounding shift right
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return emul::sat_rounding_shift_right(v, s);
    }
// !MARK
// MARK: Vector shift right and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto vt = cast<result_t>(v);
        return shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift right and accumulate
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right_accumulate(
        Vec<N, T> const& a,
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        auto shifted = rounding_shift_right<Shift>(v);
        return add<true>(a, shifted); 
    }
// !MARK

// MARK: Vector shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        auto shifted = shift_right<Shift>(v);
        return cast<result_t>(shifted);
    }
// !MARK

// MARK: Vector saturating shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_narrowing_shift_right<Shift>(v);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_unsigned_narrowing_shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector saturating rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_rounding_narrowing_shift_right<Shift>(v);
    }
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_rounding_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        return emul::sat_rounding_unsigned_narrowing_shift_right<Shift>(v);
    }
// !MARK

// MARK: Vector rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>;
        auto shifted = rounding_shift_right<Shift>(v);
        return cast<result_t>(shifted);
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
    UI_ALWAYS_INLINE auto insert_shift_left(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        using utype = std::make_unsigned_t<T>;
        auto ta = rcast<utype>(a);
        auto tb = rcast<utype>(b);
        if constexpr (sizeof(T) == 1) {
            alignas(16) static constexpr std::uint8_t mask_right[8] = {0x0, 0x1, 0x3, 0x7, 0x0f, 0x1f, 0x3f, 0x7f};
            auto mask = Vec<N, utype>::load(mask_right[Shift]);
            auto b_shift = shift_left<Shift>(tb);
            auto a_masked = bitwise_and(ta, mask);
            return rcast<T>(bitwise_or(b_shift, a_masked));
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            auto b_shift = shift_left<Shift>(tb);
            auto a_c = shift_left<bits - Shift>(ta);
            a_c = shift_right<bits - Shift>(a_c);
            return rcast<T>(bitwise_or(b_shift, a_c));
        }
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
        using utype = std::make_unsigned_t<T>;
        auto ta = rcast<utype>(a);
        auto tb = rcast<utype>(b);
        if constexpr (sizeof(T) == 1) {
            alignas(16) static constexpr std::uint8_t mask_right[9] = {0x0, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff};
            auto mask = Vec<N, utype>::load(mask_right[Shift]);
            auto b_shift = shift_right<Shift>(tb);
            auto a_masked = bitwise_and(ta, mask);
            return rcast<T>(bitwise_or(b_shift, a_masked));
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            auto b_shift = shift_right<Shift>(tb);
            auto a_c = shift_right<bits - Shift>(ta);
            a_c = shift_left<bits - Shift>(a_c);
            return rcast<T>(bitwise_or(b_shift, a_c));
        }
    }
// !MARK

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_SHIFT_HPP
