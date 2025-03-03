#ifndef AMT_UI_BITS_HPP
#define AMT_UI_BITS_HPP

#include "arch/arch.hpp"
#include <type_traits>

namespace ui::bits {

    template <unsigned R, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto rotate_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(T) * 8;
        static constexpr auto S = R % bits;
        if constexpr (S == 0) return v;
        else {
            // Ex: 1010111 rotate by 3
            //  0111000 | 0000101 => 0111101
            using utype = std::make_unsigned_t<T>;
            auto vt = rcast<utype>(v);

            auto lhs = shift_left<S>(vt);
            auto rhs = shift_right<bits - S>(vt);
            auto res = bitwise_or(lhs, rhs);
            return rcast<T>(res);
        }
    }

    template <unsigned R, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto rotate_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(T) * 8;
        static constexpr auto S = R % bits;
        if constexpr (S == 0) return v;
        else {
            // Ex: 1010111 rotate by 3
            //  0001010 | 1110000 => 1111010
            using utype = std::make_unsigned_t<T>;
            auto vt = rcast<utype>(v);

            auto lhs = shift_right<S>(vt);
            auto rhs = shift_left<bits - S>(vt);
            auto res = bitwise_or(lhs, rhs);
            return rcast<T>(res);
        }
    }

} // namespace ui::bits

#endif // AMT_UI_BITS_HPP
