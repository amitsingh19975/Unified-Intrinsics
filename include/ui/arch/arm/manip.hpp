#ifndef AMT_UI_ARCH_ARM_MANIPULATION_HPP
#define AMT_UI_ARCH_ARM_MANIPULATION_HPP

#include "cast.hpp"
#include "ui/float.hpp"
#include <arm_neon.h>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace ui::arm::neon {

// MARK: Copy vector lane
    template <unsigned ToLane, unsigned FromLane, std::size_t N, std::size_t M, typename T>
        requires (ToLane < N && FromLane < M && std::is_arithmetic_v<T>)
    UI_ALWAYS_INLINE auto copy(
        Vec<N, T> const& to,
        Vec<M, T> const& from
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = from[FromLane]
            };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vcopy_lane_f32(
                                to_vec(to),
                                ToLane,
                                to_vec(from),
                                FromLane
                            )
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vcopyq_lane_f32(
                            to_vec(to),
                            ToLane,
                            to_vec(from),
                            FromLane
                        ));
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vcopy_laneq_f32(
                                to_vec(to),
                                ToLane,
                                to_vec(from),
                                FromLane
                            )
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vcopyq_laneq_f32(
                            to_vec(to),
                            ToLane,
                            to_vec(from),
                            FromLane
                        ));
                    }
                } else if constexpr (M > 4) {
                    if constexpr (FromLane < M / 2) {
                        return copy<ToLane, FromLane>(to, from.lo);
                    } else {
                        return copy<ToLane, FromLane - M / 2>(to, from.hi);
                    }
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vcopyq_lane_f64(
                                to_vec(to),
                                ToLane,
                                to_vec(from),
                                FromLane
                            )
                        );
                    }
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vcopyq_laneq_f64(
                                to_vec(to),
                                ToLane,
                                to_vec(from),
                                FromLane
                            )
                        );
                    }
                } else if constexpr (M > 2) {
                    if constexpr (FromLane < M / 2) {
                        return copy<ToLane, FromLane>(to, from.lo);
                    } else {
                        return copy<ToLane, FromLane - M / 2>(to, from.hi);
                    }
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopy_lane_s8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vcopyq_lane_s8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopy_laneq_s8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vcopyq_laneq_s8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopy_lane_s16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopyq_lane_s16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopy_laneq_s16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopyq_laneq_s16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopy_lane_s32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopyq_lane_s32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopy_laneq_s32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopyq_laneq_s32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopyq_lane_s64(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopyq_laneq_s64(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 2) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopy_lane_u8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vcopyq_lane_u8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopy_laneq_u8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 16) {
                            return from_vec<T>(
                                vcopyq_laneq_u8(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopy_lane_u16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopyq_lane_u16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopy_laneq_u16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vcopyq_laneq_u16(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopy_lane_u32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopyq_lane_u32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopy_laneq_u32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vcopyq_laneq_u32(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopyq_lane_u64(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vcopyq_laneq_u64(
                                    to_vec(to),
                                    ToLane,
                                    to_vec(from),
                                    FromLane
                                )
                            );
                        }
                    } else if constexpr (M > 2) {
                        if constexpr (FromLane < M / 2) {
                            return copy<ToLane, FromLane>(to, from.lo);
                        } else {
                            return copy<ToLane, FromLane - M / 2>(to, from.hi);
                        }
                    }
                }
            }
            #endif

            if constexpr (ToLane < N / 2) {
                return join(
                    copy<ToLane, FromLane>(to.lo, from),
                    to.hi
                );
            } else {
                return join(
                    to.lo,
                    copy<ToLane - N / 2, FromLane>(to.hi, from)
                );
            }
        }
    }

// !MARK

// MARK: Reverse bits within elements
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto reverse_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = maths::bit_reverse(v.val)
            };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrbit_s8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrbitq_s8(to_vec(v))
                        );
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vrbit_u8(to_vec(v))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vrbitq_u8(to_vec(v))
                        );
                    }
                } 
            }
            #endif
            return join(
                reverse_bits(v.lo),
                reverse_bits(v.hi)
            );
        }
    }
// !MARK


// MARK: Reverse elements
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto reverse(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = v.val };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vrev64_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vrev64q_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vrev64_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vrev64q_f16(to_vec(v)));
                }
            } else if constexpr (std::floating_point<T>) {
                // DO nothing
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vrev64_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vrev64q_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vrev64_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vrev64q_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vrev64_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vrev64q_s32(to_vec(v)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vrev64_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vrev64q_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vrev64_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vrev64q_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vrev64_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vrev64q_u32(to_vec(v)));
                    }
                }
            }
            return join(
                reverse(v.hi),
                reverse(v.lo)
            );
        }
    }
// !MARK

// MARK: Zip
    namespace internal {
        template <std::size_t N, typename T>
        UI_ALWAYS_INLINE auto zip_low_helper(
            Vec<N, T> const& a,
            Vec<N, T> const& b
        ) noexcept -> Vec<2 * N, T> {
            if constexpr (N == 1) {
                return { a.val, b.val };
            } else {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vzip1_f32(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vzip1q_f32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vzip1q_f64(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, float16>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip1_f16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vzip1q_f16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(vzip1_s8(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 16) {
                            return from_vec<T>(vzip1q_s8(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip1_s16(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 8) {
                            return from_vec<T>(vzip1q_s16(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip1_s32(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 4) {
                            return from_vec<T>(vzip1q_s32(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip1q_s64(to_vec(a), to_vec(b)));
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(vzip1_u8(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 16) {
                            return from_vec<T>(vzip1q_u8(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip1_u16(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 8) {
                            return from_vec<T>(vzip1q_u16(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip1_u32(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 4) {
                            return from_vec<T>(vzip1q_u32(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip1q_u64(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif
                return join(
                    zip_low_helper(a.lo, b.lo),
                    zip_low_helper(a.hi, b.hi)
                );
            }
        }
        
        template <std::size_t N, typename T>
        UI_ALWAYS_INLINE auto zip_high_helper(
            Vec<N, T> const& a,
            Vec<N, T> const& b
        ) noexcept -> Vec<2 * N, T> {
            if constexpr (N == 1) {
                return { a.val, b.val };
            } else {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vzip2_f32(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vzip2q_f32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vzip2q_f64(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, float16>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip2_f16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vzip2q_f16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(vzip2_s8(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 16) {
                            return from_vec<T>(vzip2q_s8(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip2_s16(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 8) {
                            return from_vec<T>(vzip2q_s16(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip2_s32(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 4) {
                            return from_vec<T>(vzip2q_s32(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip2q_s64(to_vec(a), to_vec(b)));
                        }
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        if constexpr (N == 8) {
                            return from_vec<T>(vzip2_u8(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 16) {
                            return from_vec<T>(vzip2q_u8(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip2_u16(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 8) {
                            return from_vec<T>(vzip2q_u16(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip2_u32(to_vec(a), to_vec(b)));
                        } else if constexpr (N == 4) {
                            return from_vec<T>(vzip2q_u32(to_vec(a), to_vec(b)));
                        }
                    } else if constexpr (sizeof(T) == 8) {
                        if constexpr (N == 2) {
                            return from_vec<T>(vzip2q_u64(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif
                return join(
                    zip_high_helper(a.lo, b.lo),
                    zip_high_helper(a.hi, b.hi)
                );
            }
        }
    } // namespace internal

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return internal::zip_low_helper(a.lo, b.lo);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        return internal::zip_high_helper(a.hi, b.hi);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto unzip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(vuzp1_f32(to_vec(a), to_vec(b)));
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(vuzp1q_f64(to_vec(a), to_vec(b)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vuzp1_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vuzp1q_s64(to_vec(a), to_vec(b)));
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vuzp1_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vuzp1q_s64(to_vec(a), to_vec(b)));
                }
            }
            #endif
            return { a.lo.val, b.lo.val };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp1q_f32(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp1_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vuzp1q_f16(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vuzp1_s8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vuzp1q_s8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp1_s16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vuzp1q_s16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp1q_s32(to_vec(a), to_vec(b)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vuzp1_u8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vuzp1q_u8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp1_u16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vuzp1q_u16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp1q_u32(to_vec(a), to_vec(b)));
                    }
                }
            }
            #endif
            return join(unzip_low(a.lo, a.hi), unzip_low(b.lo, b.hi));
        }
    }
    
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto unzip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(vuzp2_f32(to_vec(a), to_vec(b)));
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(vuzp2q_f64(to_vec(a), to_vec(b)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vuzp2_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vuzp2q_s64(to_vec(a), to_vec(b)));
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vuzp2_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vuzp2q_s64(to_vec(a), to_vec(b)));
                }
            }
            #endif
            return { a.hi.val, b.hi.val };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp2q_f32(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp2_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vuzp2q_f16(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vuzp2_s8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vuzp2q_s8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp2_s16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vuzp2q_s16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp2q_s32(to_vec(a), to_vec(b)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vuzp2_u8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vuzp2q_u8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp2_u16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vuzp2q_u16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vuzp2q_u32(to_vec(a), to_vec(b)));
                    }
                }
            }
            #endif
            return join(unzip_high(a.lo, a.hi), unzip_high(b.lo, b.hi));
        }
    }
// !MARK

// MARK: Transpose elements
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto transpose_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(vtrn1_f32(to_vec(a), to_vec(b)));
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(vtrn1q_f64(to_vec(a), to_vec(b)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vtrn1_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vtrn1q_s64(to_vec(a), to_vec(b)));
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vtrn1_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vtrn1q_s64(to_vec(a), to_vec(b)));
                }
            }
            #endif
            return { a.lo.val, b.lo.val };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn1q_f32(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn1_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vtrn1q_f16(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vtrn1_s8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vtrn1q_s8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn1_s16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vtrn1q_s16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn1q_s32(to_vec(a), to_vec(b)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vtrn1_u8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vtrn1q_u8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn1_u16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vtrn1q_u16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn1q_u32(to_vec(a), to_vec(b)));
                    }
                }
            }
            #endif
            return join(transpose_low(a.lo, b.lo), transpose_low(a.hi, b.hi));
        }
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto transpose_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(vtrn2_f32(to_vec(a), to_vec(b)));
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(vtrn2q_f64(to_vec(a), to_vec(b)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vtrn2_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vtrn2q_s64(to_vec(a), to_vec(b)));
                }
            } else {
                if constexpr (sizeof(T) == 4) {
                    return from_vec<T>(vtrn2_s32(to_vec(a), to_vec(b)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<T>(vtrn2q_s64(to_vec(a), to_vec(b)));
                }
            }
            #endif
            return { a.hi.val, b.hi.val };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn2q_f32(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn2_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vtrn2q_f16(to_vec(a), to_vec(b)));
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vtrn2_s8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vtrn2q_s8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn2_s16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vtrn2q_s16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn2q_s32(to_vec(a), to_vec(b)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vtrn2_u8(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 16) {
                        return from_vec<T>(vtrn2q_u8(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn2_u16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vtrn2q_u16(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vtrn2q_u32(to_vec(a), to_vec(b)));
                    }
                }
            }
            #endif
            return join(transpose_high(a.lo, b.lo), transpose_high(a.hi, b.hi));
        }
    }

// !MARK
}

#endif // AMT_UI_ARCH_ARM_MANIPULATION_HPP
