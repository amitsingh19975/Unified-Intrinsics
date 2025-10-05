#ifndef AMT_UI_ARCH_ARM_SUB_HPP
#define AMT_UI_ARCH_ARM_SUB_HPP

#include "cast.hpp"
#include "add.hpp"
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include "../basic.hpp"
#include "../emul/sub.hpp"
#include "logical.hpp"

namespace ui::arm::neon { 

// MARK: Wrapping Subtraction
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val - rhs.val) };
        } else if constexpr (N == 8) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsub_u8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsub_s8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 16) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsubq_u8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsubq_s8(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val - rhs.val) };
        } else if constexpr (N == 4) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsub_u16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsub_s16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 8) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsubq_u16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsubq_s16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(lhs.val - rhs.val) };
        } else if constexpr (N == 2) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsub_u32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsub_s32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 4) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsubq_u32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsubq_s32(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsub_u64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsub_s64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else if constexpr (N == 2) {
            if constexpr (!std::is_signed_v<T>) {
                return from_vec<T>(
                    vsubq_u64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else {
                return from_vec<T>(
                    vsubq_s64(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Floating-Point Subtraction
    template <std::size_t N>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, float> const& lhs,
        Vec<N, float> const& rhs
    ) noexcept -> Vec<N, float> {
        if constexpr (N == 1) {
            return { .val = lhs.val - rhs.val };
        } else if constexpr (N == 2) {
            return from_vec(
                vsub_f32(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else if constexpr (N == 4) {
            return from_vec(
                vsubq_f32(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, double> const& lhs,
        Vec<N, double> const& rhs
    ) noexcept -> Vec<N, double> {
        if constexpr (N == 1) {
            return from_vec(
                vsub_f64(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else if constexpr (N == 2) {
            return from_vec(
                vsubq_f64(
                    to_vec(lhs), to_vec(rhs)
                )
            );
        } else {
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
        }    
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, float16> const& lhs,
        Vec<N, float16> const& rhs
    ) noexcept -> Vec<N, float16> {
        if constexpr (N == 1) {
            return { .val = lhs.val - rhs.val };
        } else {
            #ifdef UI_HAS_FLOAT_16
            if constexpr (N == 4) {
                return from_vec(
                    vsub_f16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            } else if constexpr (N == 8) {
                return from_vec(
                    vsubq_f16(
                        to_vec(lhs), to_vec(rhs)
                    )
                );
            }
            return join(
                sub(lhs.lo, rhs.lo),
                sub(lhs.hi, rhs.hi)
            );
            #else
            return cast<float16>(sub(cast<float>(lhs), cast<float>(rhs)));
            #endif
        }
    }

    template <std::size_t N>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, bfloat16> const& lhs,
        Vec<N, bfloat16> const& rhs
    ) noexcept -> Vec<N, bfloat16> {
        if constexpr (N == 1) {
            return { .val = lhs.val - rhs.val };
        } else {
            return cast<bfloat16>(sub(cast<float>(lhs), cast<float>(rhs)));
        }
    }
// !MARK

// MARK: Widening Subtraction
    namespace internal {
        using namespace ::ui::internal;
        template <std::size_t M, std::size_t N, std::integral T, std::integral U>
        UI_ALWAYS_INLINE auto widening_sub_helper(
            Vec<N, T> const& lhs,
            Vec<N, U> const& rhs,
            auto&& sign_fn,
            auto&& unsigned_fn
        ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
            using result_t = internal::widening_result_t<T, U>;
            if constexpr (N == 1) {
                return { .val = static_cast<result_t>(lhs.val) - static_cast<result_t>(rhs.val) };
            } else if constexpr (N == M) {
                if constexpr (!std::is_signed_v<T>) {
                    return from_vec<result_t>(
                        unsigned_fn(lhs, rhs) 
                    );
                } else {
                    return from_vec<result_t>(
                        sign_fn(lhs, rhs)
                    );
                }
            } else {
                return join(
                    widening_sub_helper<M>(lhs.lo, rhs.lo, sign_fn, unsigned_fn),
                    widening_sub_helper<M>(lhs.hi, rhs.hi, sign_fn, unsigned_fn)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>>;

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_sub_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubl_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubl_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && sizeof(U) == 1 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::widening_sub_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubw_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubw_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 1 && sizeof(U) == 2 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, U>{
        // FIXME: is it better to negate or cast?
        // Evaluate the performace and fix it.
        if constexpr (std::is_signed_v<T>) {
            return negate(widening_sub(rhs, lhs));
        } else {
            return sub(cast<U>(lhs), rhs);
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_sub_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubl_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubl_u16(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && sizeof(U) == 2 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::widening_sub_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubw_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubw_u16(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 2 && sizeof(U) == 4 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, U> {
        // FIXME: is it better to negate or cast?
        // Evaluate the performace and fix it.
        if constexpr (std::is_signed_v<T>) {
            return negate(widening_sub(rhs, lhs));
        } else {
            using wtype = internal::widening_result_t<T>;
            return sub(cast<wtype>(lhs), rhs); 
        }
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        return internal::widening_sub_helper<2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubl_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubl_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 8 && sizeof(U) == 4 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::widening_sub_helper<2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubw_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubw_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T, std::integral U>
        requires (sizeof(T) == 4 && sizeof(U) == 8 && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, U> {
        // FIXME: is it better to negate or cast?
        // Evaluate the performace and fix it.
        if constexpr (std::is_signed_v<T>) {
            return negate(widening_sub(rhs, lhs));
        } else {
            using wtype = internal::widening_result_t<T>;
            return sub(cast<wtype>(lhs), rhs); 
        }
    }
// !MARK


// MARK: Narrowing Subtraction
    namespace internal {
        template <std::size_t M0, std::size_t M1, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto halving_sub_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&&   signed_fn0,
            auto&& unsigned_fn0,
            auto&&   signed_fn1,
            auto&& unsigned_fn1
        ) noexcept -> Vec<N, T> {
            if constexpr (N == 1) {
                return emul::halving_sub(lhs, rhs);
            } else if constexpr (N == M0) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec<T>(
                        signed_fn0(lhs, rhs)
                    );
                } else {
                    return from_vec<T>(
                        unsigned_fn0(lhs, rhs)
                    );
                }
            } else if constexpr (N == M1) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec<T>(
                        signed_fn1(lhs, rhs)
                    );
                } else {
                    return from_vec<T>(
                        unsigned_fn1(lhs, rhs)
                    );
                }
            } else {
                return join(
                    halving_sub_helper<M0, M1>(
                        lhs.lo,
                        rhs.lo,
                        signed_fn0,
                        unsigned_fn0,
                        signed_fn1,
                        unsigned_fn1),
                    halving_sub_helper<M0, M1>(
                        lhs.hi,
                        rhs.hi,
                        signed_fn0,
                        unsigned_fn0,
                        signed_fn1,
                        unsigned_fn1)
                );
            }
        }


    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::halving_sub_helper<8, 16>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhsub_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsub_u8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_s8(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_u8(to_vec(l), to_vec(r)) ; }
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::halving_sub_helper<4, 8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhsub_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsub_u16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_s16(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_u16(to_vec(l), to_vec(r)) ; }
        ); 
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::halving_sub_helper<2, 4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vhsub_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsub_u32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_s32(to_vec(l), to_vec(r)) ; },
            [](auto const& l, auto const& r) { return vhsubq_u32(to_vec(l), to_vec(r)) ; }
        ); 
    }

    namespace internal {
        template <std::size_t M, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto high_narrowing_sub_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& signed_fn,
            auto&& unsigned_fn
        ) noexcept -> Vec<N, narrowing_result_t<T>> {
            using result_t = narrowing_result_t<T>; 

            if constexpr (N == 1) {
                return {
                    .val = static_cast<result_t>((lhs.val - rhs.val) >> (sizeof(result_t) * 8))
                };
            } else if constexpr (N == M) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec<result_t>(
                        signed_fn(lhs, rhs)
                    );
                } else {
                    return from_vec<result_t>(
                        unsigned_fn(lhs, rhs)
                    );
                }
            } else {
                return join(
                    high_narrowing_sub_helper<M>(lhs.lo, rhs.lo, signed_fn, unsigned_fn),
                    high_narrowing_sub_helper<M>(lhs.hi, rhs.hi, signed_fn, unsigned_fn)
                );
            }
        }
    } // namespace internal

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto high_narrowing_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::high_narrowing_sub_helper<8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubhn_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubhn_u16(to_vec(l), to_vec(r)); }
        ); 
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto high_narrowing_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::high_narrowing_sub_helper<4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubhn_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubhn_u32(to_vec(l), to_vec(r)); }
        ); 
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto high_narrowing_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        return internal::high_narrowing_sub_helper<2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vsubhn_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vsubhn_u64(to_vec(l), to_vec(r)); }
        );
    }
// !MARK

// MARK: Saturating Subtraction
    namespace internal {

        template <std::size_t M0, std::size_t M1, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto sat_sub_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& signed_fn0,
            auto&& unsigned_fn0,
            auto&& signed_fn1,
            auto&& unsigned_fn1
        ) noexcept -> Vec<N, T> {
            if constexpr (M0 != 1 && N == 1) {
                #ifdef UI_CPU_ARM64
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return { .val = vqsubb_s8(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 2) {
                        return { .val = vqsubh_s16(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 4) {
                        return { .val = vqsubs_s32(lhs.val, rhs.val) };
                    } else {
                        return { .val = vqsubd_s64(lhs.val, rhs.val) };
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return { .val = vqsubb_u8(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 2) {
                        return { .val = vqsubh_u16(lhs.val, rhs.val) };
                    } else if constexpr (sizeof(T) == 4) {
                        return { .val = vqsubs_u32(lhs.val, rhs.val) };
                    } else {
                        return { .val = vqsubd_u64(lhs.val, rhs.val) };
                    }
                }
                #endif
                return emul::sat_sub(lhs, rhs);
            } else if constexpr (N == M0) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec<T>(
                        signed_fn0(lhs, rhs)
                    );
                } else {
                    return from_vec<T>(
                        unsigned_fn0(lhs, rhs)
                    );
                } 
            } else if constexpr (N == M1) {
                if constexpr (std::is_signed_v<T>) {
                    return from_vec<T>(
                        signed_fn1(lhs, rhs)
                    );
                } else {
                    return from_vec<T>(
                        unsigned_fn1(lhs, rhs)
                    );
                } 
            } else {
                return join(
                    sat_sub_helper<M0, M1>(lhs.lo, rhs.lo, signed_fn0, unsigned_fn0, signed_fn1, unsigned_fn1),
                    sat_sub_helper<M0, M1>(lhs.hi, rhs.hi, signed_fn0, unsigned_fn0, signed_fn1, unsigned_fn1)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_sub_helper<8, 16>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqsub_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsub_u8(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqsubq_s8(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsubq_u8(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_sub_helper<4, 8>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqsub_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsub_u16(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqsubq_s16(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsubq_u16(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_sub_helper<2, 4>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqsub_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsub_u32(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqsubq_s32(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsubq_u32(to_vec(l), to_vec(r)); }
        );
    }

    template <std::size_t N, std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::sat_sub_helper<1, 2>(
            lhs, rhs,
            [](auto const& l, auto const& r) { return vqsub_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsub_u64(to_vec(l), to_vec(r)); },

            [](auto const& l, auto const& r) { return vqsubq_s64(to_vec(l), to_vec(r)); },
            [](auto const& l, auto const& r) { return vqsubq_u64(to_vec(l), to_vec(r)); }
        );
    }

// !MARK

// MARK: Subtraction with carry
    template <std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        T a,
        T b
    ) noexcept -> std::pair<T /*result*/, T /*carry*/> {
        if constexpr (sizeof(T) == 8) {
            auto res = T{};
            auto c = T{};
            asm volatile(
                "subs   %0, %2, %3\n\t"
                "cset   %w1, cc\n\t"
                : "=&r"(res), "=r"(c)
                : "r"(a), "r"(b)
                : "cc"
            );
            return { res, c };
        } else if constexpr (sizeof(T) == 4) {
            auto res = T{};
            auto c = T{};
            asm volatile(
                "subs   %w0, %w2, %w3\n\t"
                "cset   %w1, cc\n\t"
                : "=&r"(res), "=r"(c)
                : "r"(a), "r"(b)
                : "cc"
            );
            return { res, c };
        } else {
            return emul::subc(a, b);
        }
    }

    template <std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        T a,
        T b,
        T carry
    ) noexcept -> std::pair<T /*result*/, T /*carry*/> {
        if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
            auto [ta, c0] = subc(a, carry);
            auto [res, c1] = subc(ta, b);
            return { res, c0 | c1 };
        } else {
            return emul::subc(a, b);
        }
    }

    template <std::size_t N, std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> std::pair<Vec<N, T> /*result*/, T /*carry*/> {
        if constexpr (N == 1) {
            auto [s, c] = subc(a.val, b.val);
            return { Vec<N, T>(s), c };
        } else {
            if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
                auto res = Vec<N, T>{};
                auto c = T{};

                if constexpr (sizeof(T) == 8) {
                    asm volatile(
                        "subs %0, %1, %2\n\t"    // sum0 = a0 + b0
                        : "=&r"(res[0])
                        : "r"(a[0]), "r"(b[0])
                        : "cc"
                    );
                } else {
                    asm volatile(
                        "subs %w0, %w1, %w2\n\t"    // sum0 = a0 + b0
                        : "=&r"(res[0])
                        : "r"(a[0]), "r"(b[0])
                        : "cc"
                    );
                }

                auto helper = [&a, &b, &res]<std::size_t... Is>(auto&& fn, std::index_sequence<Is...>) {
                    ((res[Is + 1] = fn(a[Is + 1], b[Is + 1])),...);
                };

                helper([](T l, T r){
                    auto dst = T{};
                    if constexpr (sizeof(T) == 8) {
                        asm volatile(
                            "sbcs %0, %1, %2\n\t"
                            : "=&r"(dst), "+r"(l)
                            : "r"(r)
                            : "cc"
                        );
                    } else {
                        asm volatile(
                            "sbcs %w0, %w1, %w2\n\t"
                            : "=&r"(dst), "+r"(l)
                            : "r"(r)
                            : "cc"
                        );
                    }
                    return dst;
                }, std::make_index_sequence<N - 1>{});

                if constexpr (sizeof(T) == 8) {
                    asm volatile("cset %0, cc" : "=r"(c) :: "cc");
                } else {
                    asm volatile("cset %w0, cc" : "=r"(c) :: "cc");
                }
                return { res, c };
            } else {
                return emul::subc(a, b);
            }
        }
    }

    namespace detail {
        template <std::size_t N, std::integral T>
            requires std::is_unsigned_v<T>
        UI_ALWAYS_INLINE auto subc_helper(
            Vec<N, T> const& a,
            Vec<N, T> const& b,
            T carry = {}
        ) noexcept -> std::pair<Vec<N, T> /*result*/, T /*carry*/> {
            auto i = std::size_t{};
            auto tb = Vec<N, T>{};
            while (i < N && carry) {
                auto [r, c] = addc(b[i], carry);
                tb[i++] = r;
                carry = c;
            }
            return subc(a, tb);
        }
    } // namespace detail

    template <std::size_t N, std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T carry
    ) noexcept -> std::pair<Vec<N, T> /*result*/, T /*carry*/> {
        if (carry > 2) {
            return detail::subc_helper(a, b, carry);
        }

        if constexpr (N == 1) {
            auto [s, c] = subc(a.val, b.val);
            return { Vec<N, T>(s), c };
        } else {
            if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
                auto res = Vec<N, T>{};
                auto c = T{};

                if constexpr (sizeof(T) == 8) {
                    asm volatile(
                        "subs xzr, %3, #1\n\t"
                        "sbcs %0, %1, %2\n\t"    // sum0 = a0 + b0
                        : "=&r"(res[0])
                        : "r"(a[0]), "r"(b[0]), "r"(carry)
                        : "cc"
                    );
                } else {
                    asm volatile(
                        "subs wzr, %w3, #1\n\t"
                        "sbcs %w0, %w1, %w2\n\t"    // sum0 = a0 + b0
                        : "=&r"(res[0])
                        : "r"(a[0]), "r"(b[0]), "r"(carry)
                        : "cc"
                    );
                }

                auto helper = [&a, &b, &res]<std::size_t... Is>(auto&& fn, std::index_sequence<Is...>) {
                    ((res[Is + 1] = fn(a[Is + 1], b[Is + 1])),...);
                };

                helper([](T l, T r){
                    auto dst = T{};
                    if constexpr (sizeof(T) == 8) {
                        asm volatile(
                            "sbcs %0, %1, %2\n\t"
                            : "=&r"(dst), "+r"(l)
                            : "r"(r)
                            : "cc"
                        );
                    } else {
                        asm volatile(
                            "sbcs %w0, %w1, %w2\n\t"
                            : "=&r"(dst), "+r"(l)
                            : "r"(r)
                            : "cc"
                        );
                    }
                    return dst;
                }, std::make_index_sequence<N - 1>{});

                if constexpr (sizeof(T) == 8) {
                    asm volatile("cset %0, cc" : "=r"(c) :: "cc");
                } else {
                    asm volatile("cset %w0, cc" : "=r"(c) :: "cc");
                }
                return { res, c };
            } else {
                return emul::subc(a, b);
            }
        }
    }
// !MARK
} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_SUB_HPP
