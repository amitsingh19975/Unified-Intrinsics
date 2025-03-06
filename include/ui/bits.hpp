#ifndef AMT_UI_BITS_HPP
#define AMT_UI_BITS_HPP

#include "arch/arch.hpp"
#include "ui/base_vec.hpp"
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ui {
    template <unsigned S, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
       return rcast<T>(shift_left<S>(rcast<std::make_unsigned_t<T>>(v)));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
       return rcast<T>(shift_left(rcast<std::make_unsigned_t<T>>(v), s));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
       return rcast<T>(shift_left(rcast<std::make_unsigned_t<T>>(v), s));
    }

    template <unsigned S, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
       return rcast<T>(shift_right<S>(rcast<std::make_unsigned_t<T>>(v)));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
       return rcast<T>(shift_right(rcast<std::make_unsigned_t<T>>(v), s));
    }

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
            auto lhs = logical_shift_left<S>(v);
            auto rhs = logical_shift_right<bits - S>(v);
            auto res = bitwise_or(lhs, rhs);
            return res;
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
            auto lhs = logical_shift_right<S>(v);
            auto rhs = logical_shift_left<bits - S>(v);
            auto res = bitwise_or(lhs, rhs);
            return res;
        }
    }

    template <unsigned S, std::size_t N, std::integral T>
        requires (S <= sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto shift_left_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (S == 0) {
            return v;
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            // [10, 01, 11] << 1 => (1) [00, 11, 10]

            constexpr auto make_mask = []<std::size_t... Is>(std::index_sequence<Is...>) {
                return Vec<N, T>::load(((Is != 0 ? ~T(0) : 0))...);
            };
            constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, Vec<N, T> const& v_) {
                return shuffle<(Is + 1)..., 0>(v_);
            };
            auto mask = make_mask(std::make_index_sequence<N>{});

            if constexpr (S == bits) {
                auto t0 = bitwise_and(v, mask);
                return helper(std::make_index_sequence<N - 1>{}, t0);
            } else {
                auto l0 = logical_shift_left<S>(v); // [(1) 00, (0) 10, (1) 10]
                auto r0 = logical_shift_right<bits - S>(v); // [01, 00, 01]
                r0 = bitwise_and(mask, r0);
                auto r1 = helper(std::make_index_sequence<N - 1>{}, r0);
                return bitwise_or(l0, r1);
            } 
        }
    }

    namespace internal {
        template <unsigned S, typename T>
        constexpr auto shift_across_lane_helper() noexcept {
            if constexpr (S > 32) return std::uint64_t{};
            else if constexpr (S > 16) return std::uint32_t{};
            else if constexpr (S > 8) return std::uint16_t{};
            else return T{};
        };
    } // namespace internal

    template <unsigned S, std::size_t N, std::integral T>
        requires (S > sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto shift_left_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using utype = decltype(::ui::internal::shift_across_lane_helper<S, T>());
        using type = std::conditional_t<std::is_signed_v<T>, std::make_signed_t<utype>, utype>;
        static constexpr auto bits = sizeof(utype) * 8;

        auto t0 = rcast<T>(shift_left_across_lane<bits>(rcast<type>(v)));
        if constexpr (S < bits) {
            return t0;
        } else {
            return shift_left_across_lane<S - bits>(t0);
        }
    }

    template <unsigned S, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto logical_shift_left_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return shift_left_across_lane<S>(v);
    }

    template <unsigned S, std::size_t N, std::integral T>
        requires (S <= sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto shift_right_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (S == 0) {
            return v;
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            // [10, 01, 11] << 1 => (1) [00, 11, 10]

            auto sign_mask = cmp(v, op::less_zero_t{});
            constexpr auto make_mask = []<std::size_t... Is>(std::index_sequence<Is...>) {
                return mask_t<N, T>::load(((Is == 0 ? ~T(0) : 0))...);
            };

            constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, Vec<N, T> const& v_) {
                return shuffle<0, Is...>(v_);
            };
            auto mask = make_mask(std::make_index_sequence<N>{});
            sign_mask = bitwise_and(sign_mask, mask);

            if constexpr (S == bits) {
                auto t0 = helper(std::make_index_sequence<N - 1>{}, v);
                return bitwise_or(rcast<T>(sign_mask), t0);
            } else {
                auto sm = logical_shift_left<S>(sign_mask);
                auto l0 = logical_shift_right<S>(v); // [(1) 00, (0) 10, (1) 10]
                auto r0 =logical_shift_left<bits - S>(v); // [01, 00, 01]
                auto r1 = helper(std::make_index_sequence<N - 1>{}, r0);
                return bitwise_or(
                    bitwise_or(l0, bitwise_and(r1, rcast<T>(mask))),
                    rcast<T>(sm)
                );
            } 
        }
    }

    template <unsigned S, std::size_t N, std::integral T>
        requires (S <= sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto logical_shift_right_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (S == 0) {
            return v;
        } else {
            static constexpr auto bits = sizeof(T) * 8;
            // [10, 01, 11] << 1 => (1) [00, 11, 10]

            constexpr auto make_mask = []<std::size_t... Is>(std::index_sequence<Is...>) {
                return Vec<N, T>::load(((Is != 0 ? ~T(0) : 0))...);
            };

            constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, Vec<N, T> const& v_) {
                return shuffle<0, Is...>(v_);
            };
            auto mask = make_mask(std::make_index_sequence<N>{});

            if constexpr (S == bits) {
                auto t0 = helper(std::make_index_sequence<N - 1>{}, v);
                return bitwise_and(mask, t0);
            } else {
                auto l0 = logical_shift_right<S>(v); // [(1) 00, (0) 10, (1) 10]
                auto r0 =logical_shift_left<bits - S>(v); // [01, 00, 01]
                auto r1 = helper(std::make_index_sequence<N - 1>{}, r0);
                return bitwise_or(l0, bitwise_and(r1, mask));
            } 
        }
    }

    template <unsigned S, std::size_t N, std::integral T>
        requires (S > sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto shift_right_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using utype = decltype(::ui::internal::shift_across_lane_helper<S, T>());
        using type = std::conditional_t<std::is_signed_v<T>, std::make_signed_t<utype>, utype>;
        static constexpr unsigned bits = sizeof(utype) * 8;

        auto t0 = rcast<T>(shift_right_across_lane<bits>(rcast<type>(v)));
        if constexpr (S < bits) {
            return t0;
        } else {
            return shift_right_across_lane<S - bits>(t0);
        }
    }

    template <unsigned S, std::size_t N, std::integral T>
        requires (S > sizeof(T) * 8)
    UI_ALWAYS_INLINE static constexpr auto logical_shift_right_across_lane(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using utype = decltype(::ui::internal::shift_across_lane_helper<S, T>());
        using type = std::conditional_t<std::is_signed_v<T>, std::make_signed_t<utype>, utype>;
        static constexpr unsigned bits = sizeof(utype) * 8;

        auto t0 = rcast<T>(logical_shift_right_across_lane<bits>(rcast<type>(v)));
        if constexpr (S < bits) {
            return t0;
        } else {
            return logical_shift_right_across_lane<S - bits>(t0);
        }
    }
} // namespace ui

#endif // AMT_UI_BITS_HPP
