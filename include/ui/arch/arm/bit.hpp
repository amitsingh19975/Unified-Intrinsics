#ifndef AMT_UI_ARCH_ARM_BIT_HPP
#define AMT_UI_ARCH_ARM_BIT_HPP

#include "cast.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include "../emul/bit.hpp"
#include "add.hpp"

namespace ui::arm::neon {
// MARK: Count leading sign bits
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto count_leading_sign_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::count_leading_sign_bits(v);
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
            return emul::count_leading_zeros(v);
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
        static constexpr auto cast8 = [](auto const& v_) {
            using type = std::conditional_t<std::is_signed_v<T>, std::int8_t, std::uint8_t>;
            return std::bit_cast<Vec<N * sizeof(T), type>>(v_);
        };
        static constexpr auto from8 = [](auto const& v_) {
            using type = std::conditional_t<std::is_signed_v<T>, std::int8_t, std::uint8_t>;
            if constexpr (sizeof(T) == 2) {
                return cast<T>(from_vec<type>(v_).lo);
            } else if constexpr (sizeof(T) == 4) {
                return cast<T>(from_vec<type>(v_).lo.lo);
            } else if constexpr (sizeof(T) == 8) {
                return cast<T>(from_vec<type>(v_).lo.lo.lo);
            }
        };

        if constexpr (N == 1) {
            return emul::popcount(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vcnt_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vcntq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        return cast<T>(padd(c, c).lo);
                    } else if constexpr (N == 8) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        return cast<T>(padd(c, c).lo);
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        return cast<T>(r1.lo.lo);
                    } else if constexpr (N == 4) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        return cast<T>(r1.lo.lo);
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        auto r2 = padd(r1, r1);
                        return cast<T>(r2.lo.lo.lo);
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vcnt_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vcntq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        return cast<T>(padd(c, c).lo);
                    } else if constexpr (N == 8) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        return cast<T>(padd(c, c).lo);
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        return cast<T>(r1.lo.lo);
                    } else if constexpr (N == 4) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        return cast<T>(r1.lo.lo);
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        auto t = cast8(v);
                        auto c = popcount(t);
                        auto r0 = padd(c, c);
                        auto r1 = padd(r0, r0);
                        auto r2 = padd(r1, r1);
                        return cast<T>(r2.lo.lo.lo);
                    }
                }
            }

            return join(
                popcount(v.lo),
                popcount(v.hi)
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
            return emul::bitwise_clear(a, b);
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
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_select(cond, true_, false_);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vbsl_f32(to_vec(cond), to_vec(true_), to_vec(false_)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vbslq_f32(to_vec(cond), to_vec(true_), to_vec(false_)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vbslq_f64(to_vec(cond), to_vec(true_), to_vec(false_)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 2) {
                    return from_vec<T>(vbsl_f16(to_vec(cond), to_vec(true_), to_vec(false_)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vbslq_f16(to_vec(cond), to_vec(true_), to_vec(false_)));
                }
                #else
                return cast<T>(bitwise_select(a, cast<std::uint16_t>(b), cast<std::uint16_t>(c))); 
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(bitwise_select(cond, cast<std::uint16_t>(true_), cast<std::uint16_t>(false_))); 
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbsl_s8(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbslq_s8(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbsl_s16(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbslq_s16(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbsl_s32(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbslq_s32(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbslq_s64(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vbsl_u8(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vbslq_u8(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vbsl_u16(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vbslq_u16(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbsl_u32(to_vec(cond), to_vec(true_), to_vec(false_)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vbslq_u32(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vbslq_u64(to_vec(cond), to_vec(true_), to_vec(false_)));
                    }
                }
            }
            return join(
                bitwise_select(cond.lo, true_.lo, false_.lo),
                bitwise_select(cond.hi, true_.hi, false_.hi)
            );
        }
    }
// !MARK

}

#endif // AMT_UI_ARCH_ARM_BIT_HPP
