#ifndef AMT_UI_ARCH_ARM_BIT_HPP
#define AMT_UI_ARCH_ARM_BIT_HPP

#include "cast.hpp"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace ui::arm::neon {
// MARK: Count leading sign bits
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_sign_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            static constexpr auto bits = sizeof(T) * 8 - 1;
            auto const sign_bit = (v.val >> bits) & 1;
            auto val = v.val;
            auto count = static_cast<T>(val & sign_bit);
            for (auto pos = bits - 1; pos > 0; --pos) {
                if (((val >> pos) & 1) == sign_bit) {
                    ++count;
                    continue;
                }
                break;
            }
            return {
                .val = count
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vcls_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vclsq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vcls_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vclsq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vcls_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vclsq_s32(to_vec(v)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vcls_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vclsq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vcls_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vclsq_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vcls_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vclsq_u32(to_vec(v)));
                    }
                }
            }
            return join(
                count_leading_sign_bits(v.lo),
                count_leading_sign_bits(v.hi)
            );
        }
    }
// !MARK

// MARK: Count leading zeros
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_zeros(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(std::countl_zero(static_cast<std::make_unsigned_t<T>>(v.val)))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vclz_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vclzq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vclz_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vclzq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vclz_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vclzq_s32(to_vec(v)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vclz_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vclzq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vclz_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vclzq_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vclz_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vclzq_u32(to_vec(v)));
                    }
                }
            }

            return join(
                count_leading_zeros(v.lo),
                count_leading_zeros(v.hi)
            );
        }
    }
// !MARK

// MARK: Population Count
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto popcount(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(std::popcount(static_cast<std::make_unsigned_t<T>>(v.val)))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vclt_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vcltq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vclt_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vcltq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vclt_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vcltq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vcltq_s64(to_vec(v)));
                    }
                #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vclt_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vcltq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vclt_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vcltq_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vclt_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vcltq_u32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vcltq_u64(to_vec(v)));
                    }
                #endif
                }
            }

            return join(
                count_leading_zeros(v.lo),
                count_leading_zeros(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise clear
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_clear(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val & (~b.val))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbic_s8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbicq_s8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbic_s16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbicq_s16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbic_s32(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbicq_s32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbicq_s64(to_vec(a), to_vec(b)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbic_u8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbicq_u8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbic_u16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbicq_u16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbic_u32(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbicq_u32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbicq_u64(to_vec(a), to_vec(b)));
                    }
                }
            }
            return join(
                bitwise_clear(a.lo, b.lo),
                bitwise_clear(a.hi, b.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise select
    template <std::size_t N, typename T>
        requires (std::is_arithmetic_v<T>)
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& a,
        Vec<N, T> const& b,
        Vec<N, T> const& c
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            static constexpr auto max = std::numeric_limits<mask_inner_t<T>>::max();
            return {
                .val = static_cast<T>(a.val == max ? b.val : c.val)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vbsl_f32(to_vec(a), to_vec(b), to_vec(c)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vbslq_f32(to_vec(a), to_vec(b), to_vec(c)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vbslq_f64(to_vec(a), to_vec(b), to_vec(c)));
                }
            #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbsl_s8(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbslq_s8(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbsl_s16(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbslq_s16(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbsl_s32(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbslq_s32(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbslq_s64(to_vec(a), to_vec(b), to_vec(c)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbsl_u8(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbslq_u8(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbsl_u16(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbslq_u16(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbsl_u32(to_vec(a), to_vec(b), to_vec(c)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbslq_u32(to_vec(a), to_vec(b), to_vec(c)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbslq_u64(to_vec(a), to_vec(b), to_vec(c)));
                    }
                }
            }
            return join(
                bitwise_select(a.lo, b.lo, c.lo),
                bitwise_select(a.hi, b.hi, c.hi)
            );
        }
    }
// !MARK

}

#endif // AMT_UI_ARCH_ARM_BIT_HPP
