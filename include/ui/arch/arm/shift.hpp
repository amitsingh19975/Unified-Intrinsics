
#ifndef AMT_UI_ARCH_ARM_SHIFT_HPP
#define AMT_UI_ARCH_ARM_SHIFT_HPP

#include "cast.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include "../basic.hpp"
#include "../emul/shift.hpp"

namespace ui::arm::neon { 
    namespace internal {
        template <std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto shift_left_right_helper(
            Vec<N, T> const& v,
            Vec<N, std::make_signed_t<T>> const& s
        ) noexcept -> Vec<N, T> {
            if constexpr (N == 1) {
                if constexpr (std::same_as<T, std::int64_t>) {
                    return from_vec<T>(vshl_s64(to_vec(v), to_vec(s)));
                } else if constexpr (std::same_as<T, std::uint64_t>) {
                    return from_vec<T>(vshl_u64(to_vec(v), to_vec(s)));
                }
                return {
                    .val = static_cast<T>(v.val << s.val)
                };
            } else {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vshl_s8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vshlq_s8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vshl_s16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vshlq_s16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vshl_s32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vshlq_s32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vshlq_s64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vshl_u8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vshlq_u8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vshl_u16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vshlq_u16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vshl_u32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vshlq_u32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vshlq_u64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                }

                return join(
                    shift_left_right_helper(v.lo, s.lo),
                    shift_left_right_helper(v.hi, s.hi)
                );
            }
        }
    } // namespace internal

// MARK: Left Shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::shift_left_right_helper(v, rcast<std::make_signed_t<T>>(s));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::shift_left_right_helper(v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vshl_n_s64(to_vec(v), Shift));
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vshl_n_u64(to_vec(v), Shift));
            }
            return {
                .val = static_cast<T>(v.val << Shift)
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vshl_n_s8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vshlq_n_s8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vshl_n_s16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vshlq_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshl_n_s32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vshlq_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshlq_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vshl_n_u8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vshlq_n_u8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vshl_n_u16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vshlq_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshl_n_u32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vshlq_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshlq_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                shift_left<Shift>(v.lo),
                shift_left<Shift>(v.hi)
            );
        }
    }
// !MARK


// MARK: Saturating Left Shift
    namespace internal {
        template <std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto sat_shift_left_right_helper(
            Vec<N, T> const& v,
            Vec<N, std::make_signed_t<T>> const& s
        ) noexcept -> Vec<N, T> {
            if constexpr (N == 1) {
                if constexpr (std::same_as<T, std::int64_t>) {
                    return from_vec<T>(vqshl_s64(to_vec(v), to_vec(s)));
                } else if constexpr (std::same_as<T, std::uint64_t>) {
                    return from_vec<T>(vqshl_u64(to_vec(v), to_vec(s)));
                }
                #ifdef UI_CPU_ARM64
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return {
                            .val = static_cast<T>(vqshlb_s8(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 2) {
                        return {
                            .val = static_cast<T>(vqshlh_s16(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 4) {
                        return {
                            .val = static_cast<T>(vqshls_s32(v.val, s.val))
                        };
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return {
                            .val = static_cast<T>(vqshlb_u8(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 2) {
                        return {
                            .val = static_cast<T>(vqshlh_u16(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 4) {
                        return {
                            .val = static_cast<T>(vqshls_u32(v.val, s.val))
                        };
                    }
                }
                #endif
                if (s[0] < 0) return emul::sat_shift_right(v, { .val = static_cast<std::make_unsigned_t<T>>(s[0]) });
                return ui::emul::sat_shift_left(v, rcast<std::make_unsigned_t<T>>(s));
            } else {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vqshl_s8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vqshlq_s8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vqshl_s16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vqshlq_s16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqshl_s32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vqshlq_s32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqshlq_s64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vqshl_u8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vqshlq_u8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vqshl_u16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vqshlq_u16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqshl_u32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vqshlq_u32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqshlq_u64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                }

                return join(
                    sat_shift_left_right_helper(v.lo, s.lo),
                    sat_shift_left_right_helper(v.hi, s.hi)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::sat_shift_left_right_helper(
            v,
            rcast<std::make_signed_t<T>>(s)
        );
    }


    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift < (sizeof(T) * 8))
    UI_ALWAYS_INLINE auto sat_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vqshl_n_s64(to_vec(v), Shift));
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vqshl_n_u64(to_vec(v), Shift));
            }
            #ifdef UI_CPU_ARM64
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    return {
                        .val = static_cast<T>(vqshlb_n_s8(v.val, Shift))
                    };
                } else if constexpr (sizeof(T) == 2) {
                    return {
                        .val = static_cast<T>(vqshlh_n_s16(v.val, Shift))
                    };
                } else if constexpr (sizeof(T) == 4) {
                    return {
                        .val = static_cast<T>(vqshls_n_s32(v.val, Shift))
                    };
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    return {
                        .val = static_cast<T>(vqshlb_n_u8(v.val, Shift))
                    };
                } else if constexpr (sizeof(T) == 2) {
                    return {
                        .val = static_cast<T>(vqshlh_n_u16(v.val, Shift))
                    };
                } else if constexpr (sizeof(T) == 4) {
                    return {
                        .val = static_cast<T>(vqshls_n_u32(v.val, Shift))
                    };
                }
            }
            #endif

            return emul::sat_shift_left<Shift>(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vqshl_n_s8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vqshlq_n_s8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vqshl_n_s16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vqshlq_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vqshl_n_s32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vqshlq_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vqshlq_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vqshl_n_u8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vqshlq_n_u8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vqshl_n_u16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vqshlq_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vqshl_n_u32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vqshlq_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vqshlq_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                sat_shift_left<Shift>(v.lo),
                sat_shift_left<Shift>(v.hi)
            );
        }
    }

// !MARK

// MARK: Vector rounding shift left
    namespace internal {
        template <std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto rounding_shift_left_right_helper(
            Vec<N, T> const& v,
            Vec<N, std::make_signed_t<T>> const& s
        ) noexcept -> Vec<N, T> {
            if constexpr (N == 1) {
                if constexpr (std::same_as<T, std::int64_t>) {
                    return from_vec<T>(vrshl_s64(to_vec(v), to_vec(s)));
                } else if constexpr (std::same_as<T, std::uint64_t>) {
                    return from_vec<T>(vrshl_u64(to_vec(v), to_vec(s)));
                }
                if (s[0] < 0) return emul::rounding_shift_right(v, { .val = static_cast<std::make_unsigned_t<T>>(s[0]) });
                return emul::rounding_shift_left(v, rcast<std::make_unsigned_t<T>>(s));
            } else {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vrshl_s8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vrshlq_s8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vrshl_s16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vrshlq_s16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vrshl_s32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vrshlq_s32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vrshlq_s64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vrshl_u8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vrshlq_u8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vrshl_u16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vrshlq_u16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vrshl_u32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vrshlq_u32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vrshlq_u64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                }

                return join(
                    rounding_shift_left_right_helper(v.lo, s.lo),
                    rounding_shift_left_right_helper(v.hi, s.hi)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::rounding_shift_left_right_helper(v, rcast<std::make_signed_t<T>>(s));
    }
// !MARK

// MARK: Vector saturating rounding shift left
    namespace internal {
        template <std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto sat_rounding_shift_helper(
            Vec<N, T> const& v,
            Vec<N, std::make_signed_t<T>> const& s
        ) noexcept -> Vec<N, T> {
            if constexpr (N == 1) {
                if constexpr (std::same_as<T, std::int64_t>) {
                    return from_vec<T>(vqrshl_s64(to_vec(v), to_vec(s)));
                } else if constexpr (std::same_as<T, std::uint64_t>) {
                    return from_vec<T>(vqrshl_u64(to_vec(v), to_vec(s)));
                }
                #ifdef UI_CPU_ARM64
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return {
                            .val = static_cast<T>(vqrshlb_s8(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 2) {
                        return {
                            .val = static_cast<T>(vqrshlh_s16(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 4) {
                        return {
                            .val = static_cast<T>(vqrshls_s32(v.val, s.val))
                        };
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return {
                            .val = static_cast<T>(vqrshlb_u8(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 2) {
                        return {
                            .val = static_cast<T>(vqrshlh_u16(v.val, s.val))
                        };
                    } else if constexpr (sizeof(T) == 4) {
                        return {
                            .val = static_cast<T>(vqrshls_u32(v.val, s.val))
                        };
                    }
                }
                #endif
                if (s[0] < 0) return emul::sat_rounding_shift_right(v, { .val = static_cast<std::make_unsigned_t<T>>(s[0]) });
                return emul::sat_rounding_shift_left(v, rcast<std::make_unsigned_t<T>>(s));
            } else {
                if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vqrshl_s8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vqrshlq_s8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vqrshl_s16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vqrshlq_s16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqrshl_s32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vqrshlq_s32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqrshlq_s64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vqrshl_u8(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vqrshlq_u8(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vqrshl_u16(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vqrshlq_u16(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqrshl_u32(to_vec(v), to_vec(s))
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vqrshlq_u32(to_vec(v), to_vec(s))
                            );
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vqrshlq_u64(to_vec(v), to_vec(s))
                            );
                        }
                    }
                }

                return join(
                    sat_rounding_shift_helper(v.lo, s.lo),
                    sat_rounding_shift_helper(v.hi, s.hi)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_left(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::sat_rounding_shift_helper(v, rcast<std::make_signed_t<T>>(s));
    }
// !MARK

// MARK: Vector shift left and widen
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift < (sizeof(T) * 8)) && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_shift_left(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, ::ui::internal::widening_result_t<T>> {
        using result_t = ::ui::internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_shift_left<Shift>(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vshll_n_s8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vshll_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vshll_n_s32(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vshll_n_u8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vshll_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vshll_n_u32(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                widening_shift_left<Shift>(v.lo),
                widening_shift_left<Shift>(v.hi)
            );
        }
    }
// !MARK

// MARK: Vector shift left and insert
    /**
     * @code
     * mask = (1 << Shift) - 1
     * (a & mask) | (b << Shift) & ~mask
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
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vsli_n_s64(to_vec(a), to_vec(b), Shift));  
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vsli_n_u64(to_vec(a), to_vec(b), Shift));  
            } else {
                return emul::insert_shift_left<Shift>(a, b);
            }
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vsli_n_s8(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vsliq_n_s8(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vsli_n_s16(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vsliq_n_s16(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsli_n_s32(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vsliq_n_s32(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsliq_n_s64(to_vec(a), to_vec(b), Shift));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vsli_n_u8(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vsliq_n_u8(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vsli_n_u16(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vsliq_n_u16(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsli_n_u32(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vsliq_n_u32(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsliq_n_u64(to_vec(a), to_vec(b), Shift));
                    }
                }
            }

            return join(
                insert_shift_left<Shift>(a.lo, b.lo),
                insert_shift_left<Shift>(a.hi, b.hi)
            );
        }
    }
// !MARK

// MARK: Right shift
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::shift_left_right_helper(v, negate(rcast<std::make_signed_t<T>>(s)));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_signed_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::shift_left_right_helper(v, s);
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vshr_n_s64(to_vec(v), Shift));
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vshr_n_u64(to_vec(v), Shift));
            }
            return {
                .val = static_cast<T>(v.val >> Shift)
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vshr_n_s8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vshrq_n_s8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vshr_n_s16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vshrq_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshr_n_s32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vshrq_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshrq_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vshr_n_u8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vshrq_n_u8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vshr_n_u16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vshrq_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshr_n_u32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vshrq_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vshrq_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                shift_right<Shift>(v.lo),
                shift_right<Shift>(v.hi)
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
        return internal::sat_shift_left_right_helper(
            v,
            negate(rcast<std::make_signed_t<T>>(s))
        );
    }
// !MARK

// MARK: Vector saturating rounding shift left
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::sat_rounding_shift_helper(v, negate(rcast<std::make_signed_t<T>>(s)));
    }
// !MARK

// MARK:  Vector rounding shift right
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vrshr_n_s64(to_vec(v), Shift));
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vrshr_n_u64(to_vec(v), Shift));
            }
            return emul::rounding_shift_right<Shift>(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrshr_n_s8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrshrq_n_s8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vrshr_n_s16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vrshrq_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrshr_n_s32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vrshrq_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrshrq_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrshr_n_u8(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrshrq_n_u8(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vrshr_n_u16(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vrshrq_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrshr_n_u32(to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vrshrq_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrshrq_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                rounding_shift_right<Shift>(v.lo),
                rounding_shift_right<Shift>(v.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rounding_shift_right(
        Vec<N, T> const& v,
        Vec<N, std::make_unsigned_t<T>> const& s
    ) noexcept -> Vec<N, T> {
        return internal::rounding_shift_left_right_helper(
            v,
            negate(rcast<std::make_signed_t<T>>(s))
        );
    }
// !MARK

// MARK: Vector rounding shift right and accumulate
    template <unsigned Shift, std::size_t N, std::integral T>
        requires (Shift > 0 && Shift < sizeof(T) * 8)
    UI_ALWAYS_INLINE auto rounding_shift_right_accumulate(
        Vec<N, T> const& a,
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vrsra_n_s64(to_vec(a), to_vec(v), Shift));
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vrsra_n_u64(to_vec(a), to_vec(v), Shift));
            }
            return emul::rounding_shift_right_accumulate<Shift>(a, v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrsra_n_s8(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrsraq_n_s8(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vrsra_n_s16(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vrsraq_n_s16(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrsra_n_s32(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vrsraq_n_s32(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrsraq_n_s64(to_vec(a), to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrsra_n_u8(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrsraq_n_u8(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vrsra_n_u16(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vrsraq_n_u16(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrsra_n_u32(to_vec(a), to_vec(v), Shift)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vrsraq_n_u32(to_vec(a), to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrsraq_n_u64(to_vec(a), to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                rounding_shift_right_accumulate<Shift>(a.lo, v.lo),
                rounding_shift_right_accumulate<Shift>(a.lo, v.hi)
            );
        }
    }
// !MARK

// MARK: Vector shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = ::ui::internal::narrowing_result_t<T>;
        if constexpr (N == 1) {
            return Vec<1, result_t> {
                .val = static_cast<result_t>(v.val >> Shift)
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vshrn_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vshrn_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vshrn_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vshrn_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vshrn_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vshrn_n_u64(to_vec(v), Shift)
                        );
                    }
                }

            }

            return join(
                narrowing_shift_right<Shift>(v.lo),
                narrowing_shift_right<Shift>(v.hi)
            );
        }
    }
// !MARK

// MARK: Vector saturating shift right and narrow   
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = ::ui::internal::narrowing_result_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    return ret_t { .val = static_cast<result_t>(vqshrnh_n_s16(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 4) {
                    return ret_t { .val = static_cast<result_t>(vqshrns_n_s32(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 8) {
                    return ret_t { .val = static_cast<result_t>(vqshrnd_n_s64(v.val, Shift)) };
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    return ret_t { .val = static_cast<result_t>(vqshrnh_n_u16(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 4) {
                    return ret_t { .val = static_cast<result_t>(vqshrns_n_u32(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 8) {
                    return ret_t { .val = static_cast<result_t>(vqshrnd_n_u64(v.val, Shift)) };
                }
            }
            #endif
            return emul::sat_narrowing_shift_right<Shift>(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vqshrn_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vqshrn_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vqshrn_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vqshrn_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vqshrn_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vqshrn_n_u64(to_vec(v), Shift)
                        );
                    }
                }

            }

            return join(
                sat_narrowing_shift_right<Shift>(v.lo),
                sat_narrowing_shift_right<Shift>(v.hi)
            );
        }
    }

    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = std::make_unsigned_t<::ui::internal::narrowing_result_t<T>>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (sizeof(T) == 2) {
                return ret_t { .val = static_cast<result_t>(vqshrunh_n_s16(v.val, Shift)) };
            } else if constexpr (sizeof(T) == 4) {
                return ret_t { .val = static_cast<result_t>(vqshruns_n_s32(v.val, Shift)) };
            } else if constexpr (sizeof(T) == 8) {
                return ret_t { .val = static_cast<result_t>(vqshrund_n_s64(v.val, Shift)) };
            }
            #endif

            return Vec<1, result_t> {
                .val = static_cast<result_t>(v.val >> Shift)
            };
        } else {
            if constexpr (sizeof(T) == 2) {
                if constexpr (N == 8) {
                    return from_vec<result_t>(
                        vqshrun_n_s16(to_vec(v), Shift)
                    );
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 4) {
                    return from_vec<result_t>(
                        vqshrun_n_s32(to_vec(v), Shift)
                    );
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 2) {
                    return from_vec<result_t>(
                        vqshrun_n_s64(to_vec(v), Shift)
                    );
                }
            }
            
            return join(
                sat_unsigned_narrowing_shift_right<Shift>(v.lo),
                sat_unsigned_narrowing_shift_right<Shift>(v.hi)
            );
        }
    }
// !MARK

// MARK: Vector saturating rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto sat_rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = ::ui::internal::narrowing_result_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    return ret_t { .val = static_cast<result_t>(vqrshrnh_n_s16(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 4) {
                    return ret_t { .val = static_cast<result_t>(vqrshrns_n_s32(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 8) {
                    return ret_t { .val = static_cast<result_t>(vqrshrnd_n_s64(v.val, Shift)) };
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    return ret_t { .val = static_cast<result_t>(vqrshrnh_n_u16(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 4) {
                    return ret_t { .val = static_cast<result_t>(vqrshrns_n_u32(v.val, Shift)) };
                } else if constexpr (sizeof(T) == 8) {
                    return ret_t { .val = static_cast<result_t>(vqrshrnd_n_u64(v.val, Shift)) };
                }
            }
            #endif

            return emul::sat_rounding_narrowing_shift_right<Shift>(v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vqrshrn_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vqrshrn_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vqrshrn_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vqrshrn_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vqrshrn_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vqrshrn_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                sat_rounding_narrowing_shift_right<Shift>(v.lo),
                sat_rounding_narrowing_shift_right<Shift>(v.hi)
            );
        }
    }
    
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1 && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_rounding_unsigned_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = std::make_unsigned_t<::ui::internal::narrowing_result_t<T>>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (sizeof(T) == 2) {
                return ret_t { .val = static_cast<result_t>(vqrshrunh_n_s16(v.val, Shift)) };
            } else if constexpr (sizeof(T) == 4) {
                return ret_t { .val = static_cast<result_t>(vqrshruns_n_s32(v.val, Shift)) };
            } else if constexpr (sizeof(T) == 8) {
                return ret_t { .val = static_cast<result_t>(vqrshrund_n_s64(v.val, Shift)) };
            }
            #endif

            return emul::sat_rounding_unsigned_narrowing_shift_right<Shift>(v);
        } else {
            if constexpr (sizeof(T) == 2) {
                if constexpr (N == 8) {
                    return from_vec<result_t>(
                        vqrshrun_n_s16(to_vec(v), Shift)
                    );
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 4) {
                    return from_vec<result_t>(
                        vqrshrun_n_s32(to_vec(v), Shift)
                    );
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 2) {
                    return from_vec<result_t>(
                        vqrshrun_n_s64(to_vec(v), Shift)
                    );
                }
            }

            return join(
                sat_rounding_unsigned_narrowing_shift_right<Shift>(v.lo),
                sat_rounding_unsigned_narrowing_shift_right<Shift>(v.hi)
            );
        }
    }
// !MARK

// MARK: Vector rounding shift right and narrow
    template <unsigned Shift, std::size_t N, std::integral T>
        requires ((Shift > 0 && Shift < sizeof(T) * 8) && sizeof(T) > 1)
    UI_ALWAYS_INLINE auto rounding_narrowing_shift_right(
        Vec<N, T> const& v
    ) noexcept {
        using result_t = ::ui::internal::narrowing_result_t<T>;
        using ret_t = Vec<N, result_t>;
        if constexpr (N == 1) {
            auto temp = static_cast<std::int64_t>(v.val);
            if constexpr (Shift > 1) {
                temp += (1ll << (Shift - 1));
            }
            return ret_t {
                .val = static_cast<result_t>(temp >> Shift)
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vrshrn_n_s16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vrshrn_n_s32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vrshrn_n_s64(to_vec(v), Shift)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vrshrn_n_u16(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vrshrn_n_u32(to_vec(v), Shift)
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vrshrn_n_u64(to_vec(v), Shift)
                        );
                    }
                }
            }

            return join(
                rounding_narrowing_shift_right<Shift>(v.lo),
                rounding_narrowing_shift_right<Shift>(v.hi)
            );
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
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, std::int64_t>) {
                return from_vec<T>(vsri_n_s64(to_vec(a), to_vec(b), Shift));  
            } else if constexpr (std::same_as<T, std::uint64_t>) {
                return from_vec<T>(vsri_n_u64(to_vec(a), to_vec(b), Shift));  
            }
            return emul::insert_shift_right<Shift>(a, b);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vsri_n_s8(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vsriq_n_s8(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vsri_n_s16(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vsriq_n_s16(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsri_n_s32(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vsriq_n_s32(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsriq_n_s64(to_vec(a), to_vec(b), Shift));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vsri_n_u8(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vsriq_n_u8(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vsri_n_u16(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vsriq_n_u16(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsri_n_u32(to_vec(a), to_vec(b), Shift));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vsriq_n_u32(to_vec(a), to_vec(b), Shift));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsriq_n_u64(to_vec(a), to_vec(b), Shift));
                    }
                }
            }
        
            return join(
                insert_shift_right<Shift>(a.lo, b.lo),
                insert_shift_right<Shift>(a.hi, b.hi)
            );
        }
    }
// !MARK

} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_SHIFT_HPP
