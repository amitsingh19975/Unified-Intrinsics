#ifndef AMT_UI_ARCH_ARM_ADD_HPP
#define AMT_UI_ARCH_ARM_ADD_HPP

#include "cast.hpp"
#include <algorithm>
#include <arm_neon.h>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "basic.hpp"
#include "ui/base.hpp"

namespace ui::arm { 

// MARK: Wrapping Addition
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto wrapping_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val + rhs.val) };
        } else if constexpr (N == 8) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vadd_u8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vadd_s8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 16) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vaddq_u8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vaddq_s8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                wrapping_add(lhs.lo, rhs.lo),
                wrapping_add(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 16)
    UI_ALWAYS_INLINE auto wrapping_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val + rhs.val) };
        } else if constexpr (N == 4) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vadd_u16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vadd_s16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 8) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vaddq_u16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vaddq_s16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                wrapping_add(lhs.lo, rhs.lo),
                wrapping_add(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto wrapping_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val + rhs.val) };
        } else if constexpr (N == 2) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vadd_u32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vadd_s32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 4) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vaddq_u32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vaddq_s32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                wrapping_add(lhs.lo, rhs.lo),
                wrapping_add(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto wrapping_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vadd_u64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vadd_s64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 2) {
            if constexpr (!std::is_signed_v<T>) {
                return std::bit_cast<ret_t>(
                    vaddq_u64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return std::bit_cast<ret_t>(
                    vaddq_s64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                wrapping_add(lhs.lo, rhs.lo),
                wrapping_add(lhs.hi, rhs.hi)
            );
        }    
    }

  
// !MARK

// MARK: Floating-Point Addition

  template <std::size_t N>
    UI_ALWAYS_INLINE auto float_add(
        Vec<N, float> const& lhs,
        Vec<N, float> const& rhs
    ) noexcept -> Vec<N, float> {
        using ret_t = Vec<N, float>;

        if constexpr (N == 1) {
            return { .val = lhs.val + rhs.val };
        } else if constexpr (N == 2) {
            return std::bit_cast<ret_t>(
                vadd_f32(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else if constexpr (N == 4) {
            return std::bit_cast<ret_t>(
                vaddq_f32(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else {
            return join(
                float_add(lhs.lo, rhs.lo),
                float_add(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto float_add(
        Vec<N, double> const& lhs,
        Vec<N, double> const& rhs
    ) noexcept -> Vec<N, double> {
        using ret_t = Vec<N, double>;

        if constexpr (N == 1) {
            return std::bit_cast<ret_t>(
                vadd_f64(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else if constexpr (N == 2) {
            return std::bit_cast<ret_t>(
                vaddq_f64(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else {
            return join(
                float_add(lhs.lo, rhs.lo),
                float_add(lhs.hi, rhs.hi)
            );
        }    
    }

// !MARK

// MARK: Widening Addition
//
    namespace internal {
        template <std::size_t M, std::size_t N, std::integral T, std::integral U>
        UI_ALWAYS_INLINE auto widening_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, U> const& rhs,
            auto&& sign_fn,
            auto&& unsigned_fn
        ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
            using result_t = internal::widening_result_t<T, U>;
            using ret_t = Vec<N, result_t>;

            if constexpr (N == 1) {
                return { .val = static_cast<result_t>(lhs.val) + static_cast<result_t>(rhs.val) };
            } else if constexpr (N == M) {
                if constexpr (!std::is_signed_v<T>) {
                    return std::bit_cast<ret_t>(
                        unsigned_fn(lhs, rhs) 
                    );
                } else {
                    return std::bit_cast<ret_t>(
                        sign_fn(lhs, rhs)
                    );
                }
            } else {
                return join(
                    widening_add_helper<M>(lhs.lo, rhs.lo, sign_fn, unsigned_fn),
                    widening_add_helper<M>(lhs.hi, rhs.hi, sign_fn, unsigned_fn)
                );
            }    
        }


    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_add_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddl_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddl_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && sizeof(U) == 1)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        return internal::widening_add_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddw_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddw_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 1 && sizeof(U) == 2)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept {
        return widening_add(rhs, lhs); 
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_add_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddl_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddl_u16(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && sizeof(U) == 2)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        return internal::widening_add_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddw_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddw_u16(to_vec(l), to_vec(r)); }
        );
    }
    
    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && sizeof(U) == 4)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept {
        return widening_add(rhs, lhs); 
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_add_helper<2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddl_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddl_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 8 && sizeof(U) == 4)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        return internal::widening_add_helper<2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddw_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddw_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && sizeof(U) == 8)
    UI_ALWAYS_INLINE auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept {
        return widening_add(rhs, lhs);
    }
// !MARK


// MARK: Narrowing Addition

    namespace internal { 
        template <bool Round, std::size_t M0, std::size_t M1, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto halving_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&&   signed_fn0,
            auto&& unsigned_fn0,
            auto&&   signed_fn0_round,
            auto&& unsigned_fn0_round,
            auto&&   signed_fn1,
            auto&& unsigned_fn1,
            auto&&   signed_fn1_round,
            auto&& unsigned_fn1_round
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;
            using acc_t = widening_result_t<T>;
            if constexpr (N == 1) {
                return { .val = halving_round_helper<Round, acc_t>(lhs.val, rhs.val, op::add_t{})};
            } else if constexpr (N == M0) {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (!Round) {
                        return std::bit_cast<ret_t>(
                            signed_fn0(lhs, rhs)
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            signed_fn0_round(lhs, rhs)
                        );
                    }
                } else {
                    if constexpr (!Round) {
                        return std::bit_cast<ret_t>(
                            unsigned_fn0(lhs, rhs)
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            unsigned_fn0_round(lhs, rhs)
                        );
                    }
                }
            } else if constexpr (N == M1) {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (!Round) {
                        return std::bit_cast<ret_t>(
                            signed_fn1(lhs, rhs)
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            signed_fn1_round(lhs, rhs)
                        );
                    }
                } else {
                    if constexpr (!Round) {
                        return std::bit_cast<ret_t>(
                            unsigned_fn1(lhs, rhs)
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            unsigned_fn1_round(lhs, rhs)
                        );
                    }
                }
            } else {
                return join(
                    halving_add_helper<Round, M0, M1>(
                        lhs.lo,
                        rhs.lo,
                        signed_fn0,
                        unsigned_fn0,
                        signed_fn0_round,
                        unsigned_fn0_round,
                        signed_fn1,
                        unsigned_fn1,
                        signed_fn1_round,
                        unsigned_fn1_round), 
                    halving_add_helper<Round, M0, M1>(
                        lhs.hi,
                        rhs.hi,
                        signed_fn0,
                        unsigned_fn0,
                        signed_fn0_round,
                        unsigned_fn0_round,
                        signed_fn1,
                        unsigned_fn1,
                        signed_fn1_round,
                        unsigned_fn1_round)
                );
            }
        }


    } // namespace internal

    template <bool Round = false, std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::halving_add_helper<Round, 8, 16>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhadd_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhadd_u8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_u8(to_vec(l), to_vec(r)) ; },
            
            [](auto const& l, auto const& r) { return vhaddq_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhaddq_u8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_u8(to_vec(l), to_vec(r)) ; }
        ); 
    }

    template <bool Round = false, std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::halving_add_helper<Round, 4, 8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhadd_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhadd_u16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_u16(to_vec(l), to_vec(r)) ; },
            
            [](auto const& l, auto const& r) { return vhaddq_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhaddq_u16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_u16(to_vec(l), to_vec(r)) ; }
        ); 
    }

    template <bool Round = false, std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::halving_add_helper<Round, 2, 4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhadd_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhadd_u32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhadd_u32(to_vec(l), to_vec(r)) ; },
            
            [](auto const& l, auto const& r) { return vhaddq_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhaddq_u32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vrhaddq_u32(to_vec(l), to_vec(r)) ; }
        ); 
    }

    namespace internal {

        template <std::size_t M, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto narrowing_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& signed_fn,
            auto&& unsigned_fn
        ) noexcept -> Vec<N, narrowing_result_t<T>> {
            using result_t = narrowing_result_t<T>; 
            using ret_t    = Vec<N, result_t>; 

            if constexpr (N == 1) {
                return {
                    .val = static_cast<result_t>((lhs.val + rhs.val) >> (sizeof(result_t) * 8))
                };
            } else if constexpr (N == M) {
                if constexpr (std::is_signed_v<T>) {
                    return std::bit_cast<ret_t>(
                        signed_fn(lhs, rhs)
                    );
                } else {
                    return std::bit_cast<ret_t>(
                        unsigned_fn(lhs, rhs)
                    );
                }
            } else {
                return join(
                    narrowing_add_helper<M>(lhs.lo, rhs.lo, signed_fn, unsigned_fn),
                    narrowing_add_helper<M>(lhs.hi, rhs.hi, signed_fn, unsigned_fn)
                );
            }
        }
    } // namespace internal

    
    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::narrowing_add_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddhn_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddhn_u16(to_vec(l), to_vec(r)); }
        ); 
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::narrowing_add_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddhn_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddhn_u32(to_vec(l), to_vec(r)); }
        ); 
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::narrowing_add_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vaddhn_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vaddhn_u64(to_vec(l), to_vec(r)); }
        ); 
    }
// !MARK

// MARK: Saturating Addition

    namespace internal {

        template <typename To>
        UI_ALWAYS_INLINE constexpr auto sat_add_helper_for_one(auto lhs, auto rhs) noexcept -> To {
            auto sum = static_cast<std::int64_t>(lhs) + static_cast<std::int64_t>(rhs);
            static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<To>::min());
            static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<To>::max());
            return static_cast<To>(
                std::clamp<std::int64_t>(sum, min, max)
            );
        } 
    
        template <std::size_t M0, std::size_t M1, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto sat_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& signed_fn0,
            auto&& unsigned_fn0,
            auto&& signed_fn1,
            auto&& unsigned_fn1
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;

            if constexpr (M0 != 1 && N == 1) {
                #ifdef UI_CPU_ARM64
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return { .val = vqaddb_s8(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 2) {
                        return { .val = vqaddh_s16(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 4) {
                        return { .val = vqadds_s32(lhs.val, rhs.val) };
                    } else {
                        return { .val = vqaddd_s64(lhs.val, rhs.val) };
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return { .val = vqaddb_u8(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 2) {
                        return { .val = vqaddh_u16(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 4) {
                        return { .val = vqadds_u32(lhs.val, rhs.val) };
                    } else {
                        return { .val = vqaddd_u64(lhs.val, rhs.val) };
                    }
                }
                #else
                return {
                    .val = sat_add_helper_for_one<T>(lhs.val, rhs.val)
                };
                #endif
            } else if constexpr (N == M0) {
                if constexpr (std::is_signed_v<T>) {
                    return std::bit_cast<ret_t>(
                        signed_fn0(lhs, rhs)
                    );
                } else {
                    return std::bit_cast<ret_t>(
                        unsigned_fn0(lhs, rhs)
                    );
                } 
            } else if constexpr (N == M1) {
                if constexpr (std::is_signed_v<T>) {
                    return std::bit_cast<ret_t>(
                        signed_fn1(lhs, rhs)
                    );
                } else {
                    return std::bit_cast<ret_t>(
                        unsigned_fn1(lhs, rhs)
                    );
                } 
            } else {
                return join(
                    sat_add_helper<M0, M1>(lhs.lo, lhs.lo, signed_fn0, unsigned_fn0, signed_fn1, unsigned_fn1),
                    sat_add_helper<M0, M1>(lhs.hi, lhs.hi, signed_fn0, unsigned_fn0, signed_fn1, unsigned_fn1)
                );
            }
        }

        template <std::size_t M0, std::size_t M1, std::size_t N, std::integral T, std::integral U>
            requires (std::is_signed_v<T> && !std::is_signed_v<U>)
        UI_ALWAYS_INLINE auto sat_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, U> const& rhs,
            [[maybe_unused]] auto&& fn0,
            [[maybe_unused]] auto&& fn1
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;

            #ifdef UI_CPU_ARM64
            static constexpr auto shoudld_enable = M0 != 1;
            #else
            static constexpr auto shoudld_enable = true;
            #endif

            if constexpr (shoudld_enable && N == 1) {
                #ifdef UI_CPU_ARM64
                if constexpr (sizeof(T) == 1) {
                    return { .val = vuqaddb_s8(lhs.val, rhs.val) };
                } else if constexpr (sizeof(T) == 2) {
                    return { .val = vuqaddh_s16(lhs.val, rhs.val) };
                } else if constexpr (sizeof(T) == 4) {
                    return { .val = vuqadds_s32(lhs.val, rhs.val) };
                } else {
                    return { .val = vuqaddd_s64(lhs.val, rhs.val) };
                }
                #else
                    return ret_t {
                        .val = sat_helper_for_one<T>(lhs.val, rhs.val)
                    }; 
                #endif
            #ifdef UI_CPU_ARM64
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(lhs, rhs)
                );
            } else if constexpr (N == M1) {
                return std::bit_cast<ret_t>(
                    fn1(lhs, rhs)
                );
            #endif
            } else {
                return join(
                    sat_add_helper<M0, M1>(lhs.lo, rhs.lo, fn0, fn1),
                    sat_add_helper<M0, M1>(lhs.hi, rhs.hi, fn0, fn1)
                );
            }
        }

        template <std::size_t M0, std::size_t M1, std::size_t N, std::integral T, std::integral U>
            requires (!std::is_signed_v<T> && std::is_signed_v<U>)
        UI_ALWAYS_INLINE auto sat_add_helper(
            Vec<N, T> const& lhs,
            Vec<N, U> const& rhs,
            [[maybe_unused]] auto&& fn0,
            [[maybe_unused]] auto&& fn1
        ) noexcept -> Vec<N, T> {
            using ret_t = Vec<N, T>;
    
            #ifdef UI_CPU_ARM64
            static constexpr auto shoudld_enable = M0 != 1;
            #else
            static constexpr auto shoudld_enable = true;
            #endif

            if constexpr (shoudld_enable && N == 1) {
                #ifdef UI_CPU_ARM64
                if constexpr (sizeof(T) == 1) {
                    return { .val = vsqaddb_u8(lhs.val, rhs.val) };
                } else if constexpr (sizeof(T) == 2) {
                    return { .val = vsqaddh_u16(lhs.val, rhs.val) };
                } else if constexpr (sizeof(T) == 4) {
                    return { .val = vsqadds_u32(lhs.val, rhs.val) };
                } else {
                    return { .val = vsqaddd_u64(lhs.val, rhs.val) };
                }
                #else
                return ret_t {
                    .val = sat_helper_for_one<T>(lhs.val, rhs.val)
                }; 
                #endif
            #ifdef UI_CPU_ARM64
            } else if constexpr (N == M0) {
                return std::bit_cast<ret_t>(
                    fn0(lhs, rhs)
                );
            } else if constexpr (N == M1) {
                return std::bit_cast<ret_t>(
                    fn1(lhs, rhs)
                );
            #endif
            } else {
                return join(
                    sat_add_helper<M0, M1>(lhs.lo, rhs.lo, fn0, fn1),
                    sat_add_helper<M0, M1>(lhs.hi, rhs.hi, fn0, fn1)
                );
            }
        }

    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<8, 16>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqadd_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqadd_u8(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqaddq_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqaddq_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 1 && std::is_signed_v<T> && !std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<8, 16>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vuqadd_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vuqaddq_s8(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 1 && !std::is_signed_v<T> && std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<8, 16>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vsqadd_u8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsqaddq_u8(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<4, 8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqadd_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqadd_u16(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqaddq_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqaddq_u16(to_vec(l), to_vec(r)); }
        );
    }
    
    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && std::is_signed_v<T> && !std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<4, 8>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vuqadd_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vuqaddq_s16(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && !std::is_signed_v<T> && std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<4, 8>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vsqadd_u16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsqaddq_u16(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<2, 4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqadd_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqadd_u32(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqaddq_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqaddq_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && std::is_signed_v<T> && !std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<2, 4>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vuqadd_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vuqaddq_s32(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }


    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && !std::is_signed_v<T> && std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<2, 4>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vsqadd_u32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsqaddq_u32(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<1, 2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqadd_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqadd_u64(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqaddq_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqaddq_u64(to_vec(l), to_vec(r)); }
        );
    }  

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 8 && std::is_signed_v<T> && !std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_add_helper<1, 2>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vuqadd_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vuqaddq_s64(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 8 && !std::is_signed_v<T> && std::is_signed_v<U>)
    UI_ALWAYS_INLINE auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {    
        return internal::sat_add_helper<1, 2>(
            lhs, rhs,
            #ifdef UI_CPU_ARM64
            [](auto const& l, auto const& r) { return vsqadd_u64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsqaddq_u64(to_vec(l), to_vec(r)); }
            #else
            []{},
            []{}
            #endif
        );
    }
// !MARK

// MARK: Pairwise Addition

    template <std::size_t N, typename T>
        requires (N != 1)
    UI_ALWAYS_INLINE auto padd(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return std::bit_cast<ret_t>(
                    vpaddq_f64(to_vec(lhs), to_vec(rhs))
                );
            } 
            #endif
            if constexpr (std::same_as<T, float>) {
                return std::bit_cast<ret_t>(
                    vpadd_f32(to_vec(lhs), to_vec(rhs))
                );
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 4) {
                    return std::bit_cast<ret_t>(
                        vpadd_s32(to_vec(lhs), to_vec(rhs))
                    );
                #ifdef UI_CPU_ARM64
                }
                else if constexpr (sizeof(T) == 8) {
                    return std::bit_cast<ret_t>(
                        vpaddq_s64(to_vec(lhs), to_vec(rhs))
                    );
                #endif
                } 
            } else {
                if constexpr (sizeof(T) == 4) {
                    return std::bit_cast<ret_t>(
                        vpadd_u32(to_vec(lhs), to_vec(rhs))
                    );
                #ifdef UI_CPU_ARM64
                }
                else if constexpr (sizeof(T) == 8) {
                    return std::bit_cast<ret_t>(
                        vpaddq_u64(to_vec(lhs), to_vec(rhs))
                    );
                #endif
                } 
            }
            return ret_t(lhs[0] + lhs[1], rhs[0] + rhs[1]);
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vpaddq_f32(to_vec(lhs), to_vec(rhs))
                    );
                }
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vpadd_s8(to_vec(lhs), to_vec(rhs))
                        );

                    #ifdef UI_CPU_ARM64
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vpaddq_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vpadd_s16(to_vec(lhs), to_vec(rhs))
                        );
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vpaddq_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #ifdef UI_CPU_ARM64
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vpaddq_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vpadd_u8(to_vec(lhs), to_vec(rhs))
                        );

                    #ifdef UI_CPU_ARM64
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vpaddq_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vpadd_u16(to_vec(lhs), to_vec(rhs))
                        );
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vpaddq_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    #ifdef UI_CPU_ARM64
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vpaddq_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                    #endif
                }
            } 
            return join(
                padd(lhs.lo, rhs.lo),
                padd(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto padd(
        Vec<N, T> const& v
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64 
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<T>(vpadds_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<T>(vpaddd_f64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                if constexpr (N == 2) {
                    return static_cast<T>(vpaddd_u64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, std::int64_t>) {
                if constexpr (N == 2) {
                    return static_cast<T>(vpaddd_s64(to_vec(v)));
                }
            }
            #endif

            return padd(v.lo) + padd(v.hi);
        }
    }
// !Mark

// MARK: Widening Pairwise Addition

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_padd(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1) {
            return Vec<1, result_t>{ .val = static_cast<result_t>(v.val) };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpaddl_s8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpaddl_s16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpaddl_s16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpaddl_s16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpaddl_s32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpaddl_s32(to_vec(v))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpaddl_u8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpaddl_u16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpaddl_u16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpaddl_u16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpaddl_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpaddl_u32(to_vec(v))
                        );
                    }
                }
            }

            return join(
                widening_padd(v.lo),
                widening_padd(v.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_padd(
        Vec<N, internal::widening_result_t<T>> const& x,
        Vec<2 * N, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1)  {
            if constexpr (sizeof(T) == 4) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec(
                        vpadal_s32(to_vec(x), to_vec(v))
                    );
                } else {
                    return from_vec(
                        vpadal_u32(to_vec(x), to_vec(v))
                    );
                }
            }
            return { .val = x.val + static_cast<result_t>(v.lo.val) + static_cast<result_t>(v.hi.val) };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpadal_s8(to_vec(x), to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpadalq_s8(to_vec(x), to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpadal_s16(to_vec(x), to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpadalq_s16(to_vec(x), to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpadalq_s32(to_vec(x), to_vec(v))
                        );
                    } 
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpadal_u8(to_vec(x), to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpadalq_u8(to_vec(x), to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpadal_u16(to_vec(x), to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpadalq_u16(to_vec(x), to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpadalq_u32(to_vec(x), to_vec(v))
                        );
                    } 
                }
            }

            return join(
                widening_padd(x.lo, v.lo),
                widening_padd(x.hi, v.hi)
            );
        }
    }
// !MARK

// MARK: Addition across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::add_t op
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<T>(
                        vaddv_f32(to_vec(v))
                    );
                } else if constexpr (N == 4) {
                    return static_cast<T>(
                        vaddvq_f32(to_vec(v))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<T>(
                        vaddvq_f64(to_vec(v))
                    );
                } 
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<T>(
                            vaddv_s8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return static_cast<T>(
                            vaddvq_s8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<T>(
                            vaddv_s16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return static_cast<T>(
                            vaddvq_s16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<T>(
                            vaddv_s32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return static_cast<T>(
                            vaddvq_s32(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<T>(
                            vaddvq_s64(to_vec(v))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<T>(
                            vaddv_u8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return static_cast<T>(
                            vaddvq_u8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<T>(
                            vaddv_u16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return static_cast<T>(
                            vaddvq_u16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<T>(
                            vaddv_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return static_cast<T>(
                            vaddvq_u32(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<T>(
                            vaddvq_u64(to_vec(v))
                        );
                    }
                }
            }
            #endif
            return fold(v.lo, op) + fold(v.hi, op);
        }
    }

// !MARK

// MARK: Widening Addition across vector
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_fold(
        Vec<N, T> const& v,
        op::add_t op
    ) noexcept -> internal::widening_result_t<T> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return static_cast<result_t>(v.val);
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(
                            vaddlv_s8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(
                            vaddlvq_s8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(
                            vaddlv_s16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(
                            vaddlvq_s16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(
                            vaddlv_s32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(
                            vaddlvq_s32(to_vec(v))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(
                            vaddlv_u8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(
                            vaddlvq_u8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(
                            vaddlv_u16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(
                            vaddlvq_u16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(
                            vaddlv_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(
                            vaddlvq_u32(to_vec(v))
                        );
                    }
                }
            }
            #endif
            return fold(v.lo, op) + fold(v.hi, op);
        }
    }

// !MARK

} // namespace ui::arm;

#endif // AMT_UI_ARCH_ARM_ADD_HPP
