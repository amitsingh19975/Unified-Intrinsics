#ifndef AMT_UI_ARCH_ARM_ABS_HPP
#define AMT_UI_ARCH_ARM_ABS_HPP

#include "cast.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "basic.hpp"
#include "ui/float.hpp"

namespace ui::arm::neon { 

// MARK: Difference

    namespace internal {

        template <typename To, typename From>
        UI_ALWAYS_INLINE constexpr auto abs_diff_scalar_helper(From lhs, From rhs) noexcept -> To {
            return static_cast<To>(lhs > rhs ? (lhs - rhs) : (rhs - lhs));
        }

        template <std::size_t M0, std::size_t M1, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto abs_diff_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& fn0,
            auto&& fn1
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;
            if constexpr (M0 != 1 && N == 1) {
                #ifdef UI_CPU_ARM64
                    if constexpr (std::floating_point<T>) {
                        if constexpr (std::same_as<T, float>) {
                            return {
                                .val = vabds_f32(lhs.val, rhs.val)
                            };
                        } else if constexpr (std::same_as<T, double>) {
                            return {
                                .val = vabdd_f64(lhs.val, rhs.val)
                            };
                        } 
                    }
                #endif
                return {
                    .val = abs_diff_scalar_helper<T>(lhs.val, rhs.val)
                };
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(lhs, rhs)
                );
            } else if constexpr (N == M1) {
                return std::bit_cast<ret_t>(
                    fn1(lhs, rhs)
                );
            } else {
                return join(
                    abs_diff_helper<M0, M1>(lhs.lo, rhs.lo, fn0, fn1),
                    abs_diff_helper<M0, M1>(lhs.hi, rhs.hi, fn0, fn1)
                );
            }
        }

    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_helper<8, 16>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_s8(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_s8(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_helper<8, 16>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_u8(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_u8(to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_helper<4, 8>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_s16(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_s16(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_helper<4, 8>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_u16(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_u16(to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_helper<2, 4>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_s32(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_s32(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_helper<2, 4>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabd_u32(to_vec(l), to_vec(r)); },
                [](auto const& l, auto const& r) { return vabdq_u32(to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        // make the index very large to use scalar implementation
        return internal::abs_diff_helper<1000000,1000000>(
            lhs, rhs,
            [](auto const&, auto const&){},
            [](auto const&, auto const&){}
        );
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, float> const& lhs,
        Vec<N, float> const& rhs
    ) noexcept -> Vec<N, float> {
        return internal::abs_diff_helper<2, 4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vabd_f32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vabdq_f32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, double> const& lhs,
        Vec<N, double> const& rhs
    ) noexcept -> Vec<N, double> {
        return internal::abs_diff_helper<1, 2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vabd_f64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vabdq_f64(to_vec(l), to_vec(r)); }
        );
    }
// !MARK

// MARK: Widening Difference
    namespace internal {

        template <std::size_t M0, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto widening_abs_diff_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& fn0
        ) noexcept -> Vec<N, widening_result_t<T>> {
            using result_t = widening_result_t<T>;
            using ret_t = Vec<N, result_t>;
            if constexpr (M0 != 1 && N == 1) {
                return {
                    .val = abs_diff_scalar_helper<result_t>(lhs.val, rhs.val)
                };
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(lhs, rhs)
                );
            } else {
                return join(
                    widening_abs_diff_helper<M0>(lhs.lo, rhs.lo, fn0),
                    widening_abs_diff_helper<M0>(lhs.hi, rhs.hi, fn0)
                );
            }
        }

    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        if constexpr (std::is_signed_v<T>) {
            return internal::widening_abs_diff_helper<8>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_s8(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::widening_abs_diff_helper<8>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_u8(to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        if constexpr (std::is_signed_v<T>) {
            return internal::widening_abs_diff_helper<4>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_s16(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::widening_abs_diff_helper<4>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_u16(to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        if constexpr (std::is_signed_v<T>) {
            return internal::widening_abs_diff_helper<2>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_s32(to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::widening_abs_diff_helper<2>(
                lhs, rhs,
                [](auto const& l, auto const& r) { return vabdl_u32(to_vec(l), to_vec(r)); }
            );
        }
    }
// !MARK

// MARK: Absolute difference and Accumulate

    namespace internal {
        template <std::size_t M0, std::size_t M1, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto abs_diff_acc_helper(
            Vec<N, T> const& acc,
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& fn0,
            auto&& fn1
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;
            if constexpr (M0 != 1 && N == 1) {
                return {
                    .val = acc.val + abs_diff_scalar_helper<T>(lhs.val, rhs.val)
                };
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(acc, lhs, rhs)
                );
            } else if constexpr (N == M1) {
                return std::bit_cast<ret_t>(
                    fn1(acc, lhs, rhs)
                );
            } else {
                return join(
                    abs_diff_helper<M0, M1>(acc.lo, lhs.lo, rhs.lo, fn0, fn1),
                    abs_diff_helper<M0, M1>(acc.hi, lhs.hi, rhs.hi, fn0, fn1)
                );
            }
        }

        template <std::size_t M0, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto abs_diff_acc_helper(
            Vec<N, widening_result_t<T>> const& acc,
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& fn0
        ) noexcept -> Vec<N, widening_result_t<T>> {
            using ret_t = Vec<N, widening_result_t<T>>;
            if constexpr (M0 != 1 && N == 1) {
                return {
                    .val = acc.val + abs_diff_scalar_helper<T>(lhs.val, rhs.val)
                };
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(acc, lhs, rhs)
                );
            } else {
                return join(
                    abs_diff_acc_helper<M0>(acc.lo, lhs.lo, rhs.lo, fn0),
                    abs_diff_acc_helper<M0>(acc.hi, lhs.hi, rhs.hi, fn0)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<8, 16>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_s8(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_s8(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<8, 16>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_u8(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_u8(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<4, 8>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_s16(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_s16(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<4, 8>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_u16(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_u16(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<2, 4>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_s32(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_s32(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<2, 4>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vaba_u32(to_vec(a), to_vec(l), to_vec(r)); },
                [](auto const& a, auto const& l, auto const& r) { return vabaq_u32(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::abs_diff_acc_helper<100000, 100000>(
            acc, lhs, rhs,
            [](auto const&, auto const&, auto const&) {  },
            [](auto const&, auto const&, auto const&) {  }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, U> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<8>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_s8(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<8>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_u8(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, U> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<4>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_s16(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<4>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_u16(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, U> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            return internal::abs_diff_acc_helper<2>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_s32(to_vec(a), to_vec(l), to_vec(r)); }
            );
        } else {
            return internal::abs_diff_acc_helper<2>(
                acc, lhs, rhs,
                [](auto const& a, auto const& l, auto const& r) { return  vabal_u32(to_vec(a), to_vec(l), to_vec(r)); }
            );
        }
    }

// !MARK

// MARK: Absolute Value
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
        if constexpr (std::floating_point<T>) {
            if constexpr (N == 1) return { .val = static_cast<T>(std::abs(v.val)) };
        #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vabs_f32(to_vec(v)));
                else if constexpr (N == 4) return std::bit_cast<ret_t>(vabsq_f32(to_vec(v)));
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vabs_f64(to_vec(v)));
            }
        #else
        #endif
            if constexpr (N > 1) {
                return join(
                    abs(v.lo),
                    abs(v.hi)
                );
            }
        } else if constexpr (std::is_signed_v<T>) {
            if constexpr (N == 1) return { .val = static_cast<T>(std::abs(v.val)) };
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 8) {
                    return std::bit_cast<ret_t>(vabs_s8(to_vec(v)));
                } else if constexpr (N == 16) {
                    return std::bit_cast<ret_t>(vabsq_s8(to_vec(v)));
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N == 4) return std::bit_cast<ret_t>(vabs_s16(to_vec(v)));
                else if constexpr (N == 8) return std::bit_cast<ret_t>(vabsq_s16(to_vec(v)));
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vabs_s32(to_vec(v)));
                else if constexpr (N == 4) return std::bit_cast<ret_t>(vabsq_s32(to_vec(v)));
            #ifdef UI_CPU_ARM64
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vabsq_s64(to_vec(v)));
            #endif
            }
          
            if constexpr (N > 1) {
                return join(
                    abs(v.lo),
                    abs(v.hi)
                );
            }
        } else {
            return v;
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
        constexpr auto helper = [](T val) {
            static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<T>::min());
            static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<T>::max());
            return static_cast<T>(std::clamp(std::abs(static_cast<std::int64_t>(val)), min, max));

        };
        if constexpr (std::floating_point<T>) {
            if constexpr (N == 1) return { .val = helper(v.val) };
            else {
                return join(
                    sat_abs(v.lo),
                    sat_abs(v.hi)
                );
            }
        } else if constexpr (std::is_signed_v<T>) {
            if constexpr (N == 1) return { .val = helper(v.val) };
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 8) {
                    return std::bit_cast<ret_t>(vqabs_s8(to_vec(v)));
                } else if constexpr (N == 16) {
                    return std::bit_cast<ret_t>(vqabsq_s8(to_vec(v)));
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N == 4) return std::bit_cast<ret_t>(vqabs_s16(to_vec(v)));
                else if constexpr (N == 8) return std::bit_cast<ret_t>(vqabsq_s16(to_vec(v)));
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vqabs_s32(to_vec(v)));
                else if constexpr (N == 4) return std::bit_cast<ret_t>(vqabsq_s32(to_vec(v)));
            #ifdef UI_CPU_ARM64
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 2) return std::bit_cast<ret_t>(vqabsq_s64(to_vec(v)));
            #endif
            }
           
            if constexpr (N > 1) {
                return join(
                    sat_abs(v.lo),
                    sat_abs(v.hi)
                );
            }
        } else {
            return v;
        }
    }

// !MARK

// MARK: (B)Float16 Absolute value

    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, ui::float16> const& v
    ) noexcept -> Vec<N, float16> {
        if constexpr (N == 1) {
            return {
                .val = v.val.abs()
            };
        } else {
            #ifdef UI_HAS_FLOAT_16 
            if constexpr (N == 4) {
                return from_vec(vabs_f16(to_vec(v)));
            } else if constexpr (N == 8) {
                return from_vec(vabsq_f16(to_vec(v)));
            }
            return join(
                abs(v.lo),
                abs(v.hi)
            );
            #else
            return cast<float16>(abs(cast<float>(v)));
            #endif
        }
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, ui::bfloat16> const& v
    ) noexcept -> Vec<N, bfloat16> {
        if constexpr (N == 1) {
            return {
                .val = v.abs()
            };
        } else {
            return cast<bfloat16>(abs(cast<float>(v)));
        }
    }
// !MARK

// MARK: (B)Float16 Absolute value
    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, ui::float16> const& a,
        Vec<N, ui::float16> const& b
    ) noexcept -> Vec<N, float16> {
        if constexpr (N == 1) {
            auto temp = a.val > b.val ? a.val - b.val : b.val - a.val;
            return {
                .val = temp
            };
        } else {
            #ifdef UI_HAS_FLOAT_16 
            if constexpr (N == 4) {
                return from_vec(vabd_f16(to_vec(a), to_vec(b)));
            } else if constexpr (N == 8) {
                return from_vec(vabdq_f16(to_vec(a), to_vec(b)));
            }
            return join(
                abs_diff(a.lo, b.lo),
                abs_diff(a.hi, b.hi)
            );
            #else
            auto ta = cast<float>(a);
            auto tb = cast<float>(b);
            return cast<float16>(abs_diff(ta, tb));
            #endif
        }
    }
    
    template <std::size_t N>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, ui::bfloat16> const& a,
        Vec<N, ui::bfloat16> const& b
    ) noexcept -> Vec<N, bfloat16> {
        if constexpr (N == 1) {
            auto temp = a.val > b.val ? a.val - b.val : b.val - a.val;
            return {
                .val = temp
            };
        } else {
            auto ta = cast<float>(a);
            auto tb = cast<float>(b);
            return cast<bfloat16>(abs_diff(ta, tb));
        }
    }
// !MARK

} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_ABS_HPP
