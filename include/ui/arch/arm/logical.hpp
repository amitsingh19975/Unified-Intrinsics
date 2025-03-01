#ifndef AMT_UI_ARCH_ARM_LOGICAL_HPP
#define AMT_UI_ARCH_ARM_LOGICAL_HPP

#include "cast.hpp"
#include "../emul/logical.hpp"
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ui::arm::neon {
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

// MARK: Negation
    template <std::size_t N, typename T>
        requires (std::is_floating_point_v<T> || std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::negate(v);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vneg_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vnegq_f32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vnegq_f64(to_vec(v)));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vneg_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vnegq_f16(to_vec(v)));
                }
                #else
                auto tmp = rcast<std::uint16_t>(v);
                auto mask = Vec<N, std::uint16_t>::load(0x8000);
                return rcast<T>(bitwise_xor(tmp, mask)); 
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                auto tmp = rcast<std::uint16_t>(v);
                auto mask = Vec<N, std::uint16_t>::load(0x8000);
                return rcast<T>(bitwise_xor(tmp, mask)); 
            } else if constexpr (std::integral<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vneg_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vnegq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vneg_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vnegq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vneg_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vnegq_s32(to_vec(v)));
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vnegq_s64(to_vec(v)));
                    }
                #endif
                }
            } 
            return join(
                negate(v.lo),
                negate(v.hi)
            );
        }
    }

    template <std::size_t N, typename T>
        requires (std::is_arithmetic_v<T> && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::sat_negate(v);
        } else {
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 8) {
                    return from_vec<T>(vqneg_s8(to_vec(v)));
                } else if constexpr (N == 16) {
                    return from_vec<T>(vqnegq_s8(to_vec(v)));
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N == 4) {
                    return from_vec<T>(vqneg_s16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vqnegq_s16(to_vec(v)));
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 2) {
                    return from_vec<T>(vqneg_s32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vqnegq_s32(to_vec(v)));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 2) {
                    return from_vec<T>(vqnegq_s64(to_vec(v)));
                }
            #endif
            }
            return join(
                sat_negate(v.lo),
                sat_negate(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Not
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_not(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_not(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmvn_s8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmvnq_s8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmvn_s16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmvnq_s16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmvn_s32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmvnq_s32(to_vec(v))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmvn_u8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmvnq_u8(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmvn_u16(to_vec(v))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmvnq_u16(to_vec(v))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmvn_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmvnq_u32(to_vec(v))
                        );
                    }
                }
            }
            return join(
                bitwise_not(v.lo),
                bitwise_not(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise And
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_and(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_and(lhs, rhs);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vand_s8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vandq_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vand_s16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vandq_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vand_s32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vandq_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vand_u8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vandq_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vand_u16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vandq_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vand_u32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vandq_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }
            return join(
                bitwise_and(lhs.lo, rhs.lo),
                bitwise_and(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise OR
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_or(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_or(lhs, rhs);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vorr_s8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vorrq_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vorr_s16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vorrq_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vorr_s32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vorrq_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vorr_u8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vorrq_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vorr_u16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vorrq_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vorr_u32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vorrq_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }
            return join(
                bitwise_or(lhs.lo, rhs.lo),
                bitwise_or(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise XOR
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_xor(lhs, rhs);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            veor_s8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            veorq_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            veor_s16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            veorq_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            veor_s32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            veorq_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            veor_u8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            veorq_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            veor_u16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            veorq_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            veor_u32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            veorq_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }
            return join(
                bitwise_xor(lhs.lo, rhs.lo),
                bitwise_xor(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Or-Not lhs | ~rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_ornot(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_ornot(lhs, rhs);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vorn_s8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vornq_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vorn_s16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vornq_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vorn_s32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vornq_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vorn_u8(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vornq_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vorn_u16(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vornq_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vorn_u32(to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vornq_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }
            return join(
                bitwise_ornot(lhs.lo, rhs.lo),
                bitwise_ornot(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Not-And ~lhs & rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_notand(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return bitwise_not(bitwise_ornot(lhs, rhs));
    }
// !MARK
}

#endif // AMT_UI_ARCH_ARM_LOGICAL_HPP
