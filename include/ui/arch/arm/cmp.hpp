#ifndef AMT_UI_ARCH_ARM_CMP_HPP
#define AMT_UI_ARCH_ARM_CMP_HPP

#include "cast.hpp"
#include "ui/float.hpp"
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace ui::arm::neon { 

// MARK: Bitwise equal and 'and' test
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vceqs_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vceqd_f64(lhs.val, rhs.val)) };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            return { .val = (lhs.val == rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vceq_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vceqq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vceqq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vceq_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vceqq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vceq_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vceqq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vceq_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vceqq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vceq_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vceqq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vceq_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vceq_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vceqq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vceq_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vceqq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vceq_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vceqq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vceq_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }
    
    /**
     * @return (lhs & rhs) == 0
     */
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::and_test_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::integral<T>) {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vtstd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vtstd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            static constexpr auto true_ = std::numeric_limits<result_t>::max();
            static constexpr auto false_ = result_t{};

            return { .val = ((lhs.val & rhs.val)) ? true_ : false_ };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vtst_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vtstq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vtst_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vtstq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vtst_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vtstq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vtst_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vtst_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vtstq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vtst_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vtstq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vtst_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vtstq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vtst_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }
// !MARK

// MARK:  Greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcges_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcged_f64(lhs.val, rhs.val)) };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            return { .val = (lhs.val >= rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcge_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcgeq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgeq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcge_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcgeq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcge_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgeq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcge_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgeq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcge_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgeq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcge_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcge_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgeq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcge_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgeq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcge_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgeq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcge_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        op::greater_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = static_cast<result_t>(vcgezs_f32(v.val))
                }; 
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = static_cast<result_t>(vcgezd_f64(v.val))
                }; 
            } else {
                return {
                    .val = static_cast<result_t>(vcgezd_s64(static_cast<std::int64_t>(v.val)))
                };
            }
            #endif
            return { .val = (v.val == T{0}) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgez_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcgezq_f32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgezq_f64(to_vec(v)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcgez_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcgezq_f16(to_vec(v)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgez_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgezq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgez_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgezq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgez_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgezq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgez_s64(to_vec(v)));
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgez_s8(std::bit_cast<int8x8_t>(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgezq_s8(std::bit_cast<int8x16_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgez_s16(std::bit_cast<int16x4_t>(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgezq_s16(std::bit_cast<int16x8_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgez_s32(std::bit_cast<int32x2_t>(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgezq_s32(std::bit_cast<int32x4_t>(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgez_s64(std::bit_cast<int64x2_t>(v)));
                    }
                #endif
                }
            }

            return join(
                cmp(v.lo, op),
                cmp(v.hi, op)
            );
        }
    }

// !MARK

// MARK: Less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::less_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcles_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcled_f64(lhs.val, rhs.val)) };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            return { .val = (lhs.val <= rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcle_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcleq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcleq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcle_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcleq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcle_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcleq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcle_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcleq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcle_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcleq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcle_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcle_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcleq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcle_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcleq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcle_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcleq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcle_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        op::less_equal_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = static_cast<result_t>(vclezs_f32(v.val))
                }; 
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = static_cast<result_t>(vclezd_f64(v.val))
                }; 
            } else {
                return {
                    .val = static_cast<result_t>(vclezd_s64(static_cast<std::int64_t>(v.val)))
                };
            }
            #endif
            return { .val = (v.val == T{0}) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vclez_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vclezq_f32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vclezq_f64(to_vec(v)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vclez_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vclezq_f16(to_vec(v)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclez_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vclezq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclez_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclezq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclez_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclezq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclez_s64(to_vec(v)));
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclez_s8(std::bit_cast<int8x8_t>(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vclezq_s8(std::bit_cast<int8x16_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclez_s16(std::bit_cast<int16x4_t>(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclezq_s16(std::bit_cast<int16x8_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclez_s32(std::bit_cast<int32x2_t>(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclezq_s32(std::bit_cast<int32x4_t>(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclez_s64(std::bit_cast<int64x2_t>(v)));
                    }
                #endif
                }
            }

            return join(
                cmp(v.lo, op),
                cmp(v.hi, op)
            );
        }
    }
// !MARK

// MARK: Greater Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcgts_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcgtd_f64(lhs.val, rhs.val)) };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            return { .val = (lhs.val > rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgt_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcgtq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgtq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcgt_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcgtq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgt_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgtq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgt_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgt_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgt_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgt_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgtq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgt_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgt_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgt_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        op::greater_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = static_cast<result_t>(vcgtzs_f32(v.val))
                }; 
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = static_cast<result_t>(vcgtzd_f64(v.val))
                }; 
            } else {
                return {
                    .val = static_cast<result_t>(vcgtzd_s64(static_cast<std::int64_t>(v.val)))
                };
            }
            #endif
            return { .val = (v.val > T{0}) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgtz_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcgtzq_f32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcgtzq_f64(to_vec(v)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcgtz_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcgtzq_f16(to_vec(v)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtz_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgtzq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtz_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtzq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgtz_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtzq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgtz_s64(to_vec(v)));
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtz_s8(std::bit_cast<int8x8_t>(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcgtzq_s8(std::bit_cast<int8x16_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtz_s16(std::bit_cast<int16x4_t>(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcgtzq_s16(std::bit_cast<int16x8_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgtz_s32(std::bit_cast<int32x2_t>(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcgtzq_s32(std::bit_cast<int32x4_t>(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcgtz_s64(std::bit_cast<int64x2_t>(v)));
                    }
                #endif
                }
            }

            return join(
                cmp(v.lo, op),
                cmp(v.hi, op)
            );
        }
    }

// !MARK

// MARK: Less Than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::less_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vclts_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcltd_f64(lhs.val, rhs.val)) };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_s64(
                                static_cast<std::int64_t>(lhs.val),
                                static_cast<std::int64_t>(rhs.val)
                            )
                        )
                    };
                } else {
                    return { 
                        .val = static_cast<result_t>(
                            vceqd_u64(
                                static_cast<std::uint64_t>(lhs.val),
                                static_cast<std::uint64_t>(rhs.val)
                            )
                        )
                    };
                }
            }
            #endif
            return { .val = (lhs.val < rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vclt_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcltq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcltq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vclt_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcltq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclt_s8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcltq_s8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclt_s16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltq_s16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclt_s32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltq_s32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclt_s64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vclt_u8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcltq_u8(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vclt_u16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltq_u16(to_vec(lhs), to_vec(rhs)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclt_u32(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltq_u32(to_vec(lhs), to_vec(rhs)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vclt_u64(to_vec(lhs), to_vec(rhs)));
                    }
                #endif
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& v,
        op::less_zero_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = static_cast<result_t>(vcltzs_f32(v.val))
                }; 
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = static_cast<result_t>(vcltzd_f64(v.val))
                }; 
            } else {
                return {
                    .val = static_cast<result_t>(vcltzd_s64(static_cast<std::int64_t>(v.val)))
                };
            }
            #endif
            return { .val = (v.val < T{0}) ? std::numeric_limits<result_t>::max() : result_t{} };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcltz_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcltzq_f32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcltzq_f64(to_vec(v)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcltz_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcltzq_f16(to_vec(v)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltz_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcltzq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltz_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltzq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcltz_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltzq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcltz_s64(to_vec(v)));
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) { 
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltz_s8(std::bit_cast<int8x8_t>(v)));
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(vcltzq_s8(std::bit_cast<int8x16_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltz_s16(std::bit_cast<int16x4_t>(v)));
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcltzq_s16(std::bit_cast<int16x8_t>(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcltz_s32(std::bit_cast<int32x2_t>(v)));
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcltzq_s32(std::bit_cast<int32x4_t>(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcltz_s64(std::bit_cast<int64x2_t>(v)));
                    }
                #endif
                }
            }

            return join(
                cmp(v.lo, op),
                cmp(v.hi, op)
            );
        }
    }

// !MARK

// MARK: Absolute greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::abs_greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcages_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcaged_f64(lhs.val, rhs.val)) };
            }
            #endif
            if constexpr (std::is_signed_v<T>) {
                return { .val = (std::abs(lhs.val) >= std::abs(rhs.val)) ? std::numeric_limits<result_t>::max() : result_t{} };
            } else {
                return { .val = (lhs.val >= rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcage_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcageq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcageq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcage_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcageq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else {
                if constexpr (sizeof(T) == 8) {
                    return cmp(cast<double>(lhs), cast<double>(rhs), op);
                } else {
                    return cmp(cast<float>(lhs), cast<float>(rhs), op);
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

// !MARK

// MARK: Absolute less than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::abs_less_equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcales_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcaled_f64(lhs.val, rhs.val)) };
            }
            #endif
            if constexpr (std::is_signed_v<T>) {
                return { .val = (std::abs(lhs.val) <= std::abs(rhs.val)) ? std::numeric_limits<result_t>::max() : result_t{} };
            } else {
                return { .val = (lhs.val <= rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcale_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcaleq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcaleq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcale_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcaleq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else {
                if constexpr (sizeof(T) == 8) {
                    return cmp(cast<double>(lhs), cast<double>(rhs), op);
                } else {
                    return cmp(cast<float>(lhs), cast<float>(rhs), op);
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

// !MARK

// MARK: Absolute greater than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::abs_greater_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcagts_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcagtd_f64(lhs.val, rhs.val)) };
            }
            #endif
            if constexpr (std::is_signed_v<T>) {
                return { .val = (std::abs(lhs.val) > std::abs(rhs.val)) ? std::numeric_limits<result_t>::max() : result_t{} };
            } else {
                return { .val = (lhs.val > rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcagt_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcagtq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcagtq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcagt_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcagtq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else {
                if constexpr (sizeof(T) == 8) {
                    return cmp(cast<double>(lhs), cast<double>(rhs), op);
                } else {
                    return cmp(cast<float>(lhs), cast<float>(rhs), op);
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

// !MARK

// MARK: Absolute less than
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::abs_less_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return { .val = static_cast<result_t>(vcalts_f32(lhs.val, rhs.val)) };
            } else if constexpr (std::same_as<T, double>) {
                return { .val = static_cast<result_t>(vcaltd_f64(lhs.val, rhs.val)) };
            }
            #endif
            if constexpr (std::is_signed_v<T>) {
                return { .val = (std::abs(lhs.val) > std::abs(rhs.val)) ? std::numeric_limits<result_t>::max() : result_t{} };
            } else {
                return { .val = (lhs.val > rhs.val) ? std::numeric_limits<result_t>::max() : result_t{} };
            }
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcalt_f32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcaltq_f32(to_vec(lhs), to_vec(rhs)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcaltq_f64(to_vec(lhs), to_vec(rhs)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vcalt_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vcaltq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
            } else {
                if constexpr (sizeof(T) == 8) {
                    return cmp(cast<double>(lhs), cast<double>(rhs), op);
                } else {
                    return cmp(cast<float>(lhs), cast<float>(rhs), op);
                }
            } 
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }
// !MARK



} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_CMP_HPP
