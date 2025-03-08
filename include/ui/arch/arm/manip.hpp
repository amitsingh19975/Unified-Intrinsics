#ifndef AMT_UI_ARCH_ARM_MANIPULATION_HPP
#define AMT_UI_ARCH_ARM_MANIPULATION_HPP

#include "cast.hpp"
#include "shift.hpp"
#include "logical.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace ui::arm::neon {

// MARK: Copy vector lane
    template <unsigned ToLane, unsigned FromLane, std::size_t N, std::size_t M, typename T>
        requires (ToLane < N && FromLane < M)
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
            } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return rcast<T>(copy<ToLane, FromLane>(
                    rcast<std::uint16_t>(to),
                    rcast<std::uint16_t>(from)
                ));
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
                    auto [lo, hi] = from_vec<T>(vrev64q_f32(to_vec(v)));
                    return join(hi, lo);
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vrev64_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    auto [lo, hi] = from_vec<T>(vrev64q_f16(to_vec(v)));
                    return join(hi, lo);
                }
                #else
                return rcast<T>(reverse(rcast<std::uint16_t>(v))); 
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return rcast<T>(reverse(rcast<std::uint16_t>(v))); 
            } else if constexpr (std::floating_point<T>) {
                if constexpr (sizeof(T) == 8) {
                    return rcast<T>(reverse(rcast<std::int64_t>(v)));
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vrev64_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        auto [lo, hi] = from_vec<T>(vrev64q_s8(to_vec(v)));
                        return join(hi, lo);
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vrev64_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        auto [lo, hi] = from_vec<T>(vrev64q_s16(to_vec(v)));
                        return join(hi, lo);
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vrev64_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        auto [lo, hi] = from_vec<T>(vrev64q_s32(to_vec(v)));
                        return join(hi, lo);
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(vrev64_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        auto [lo, hi] = from_vec<T>(vrev64q_u8(to_vec(v)));
                        return join(hi, lo);
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vrev64_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        auto [lo, hi] = from_vec<T>(vrev64q_u16(to_vec(v)));
                        return join(hi, lo);
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vrev64_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        auto [lo, hi] = from_vec<T>(vrev64q_u32(to_vec(v)));
                        return join(hi, lo);
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
        struct zip_helper {
            template <std::size_t N, typename T>
                requires (N == 2)
            UI_ALWAYS_INLINE auto low(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept -> Vec<N, T> {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(vzip1_f32(to_vec(a), to_vec(b)));
                } else if constexpr (std::same_as<T, double>) {
                    return rcast<T>(low(rcast<std::uint64_t>(a), rcast<std::uint64_t>(b)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(vzip1q_f64(to_vec(a), to_vec(b)));
                } else if constexpr (std::integral<T>) {
                    if constexpr (std::is_signed_v<T>) {
                        if constexpr (sizeof(T) == 4) {
                            return from_vec<T>(vzip1_s32(to_vec(a), to_vec(b)));
                        } else if constexpr (sizeof(T) == 8) {
                            return from_vec<T>(vzip1q_s64(to_vec(a), to_vec(b)));
                        }
                    } else {
                        if constexpr (sizeof(T) == 4) {
                            return from_vec<T>(vzip1_u32(to_vec(a), to_vec(b)));
                        } else if constexpr (sizeof(T) == 8) {
                            return from_vec<T>(vzip1q_u64(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif

                return { a[0], b[0] };
            }

            template <std::size_t N, typename T>
                requires (N > 2)
            UI_ALWAYS_INLINE auto low(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip1q_f32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, float16>) {
                    #ifdef UI_HAS_FLOAT_16
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip1_f16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vzip1q_f16(to_vec(a), to_vec(b)));
                    }
                    #else
                    auto l = rcast<std::uint16_t>(a);
                    auto r = rcast<std::uint16_t>(b);
                    using ret_t = decltype(low(l, r));
                    if constexpr (!std::is_void_v<ret_t>) {
                        return rcast<T>(low(l, r));
                    }
                    #endif
                } else if constexpr (std::same_as<T, bfloat16>) {
                    auto l = rcast<std::uint16_t>(a);
                    auto r = rcast<std::uint16_t>(b);
                    using ret_t = decltype(low(l, r));
                    if constexpr (!std::is_void_v<ret_t>) {
                        return rcast<T>(low(l, r));
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
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip1q_s32(to_vec(a), to_vec(b)));
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
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip1q_u32(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif
            }

            template <std::size_t N, typename T>
                requires (N == 2)
            UI_ALWAYS_INLINE auto high(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept -> Vec<N, T> {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(vzip2_f32(to_vec(a), to_vec(b)));
                } else if constexpr (std::same_as<T, double>) {
                    return rcast<T>(high(rcast<std::uint64_t>(a), rcast<std::uint64_t>(b)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(vzip2q_f64(to_vec(a), to_vec(b)));
                } else if constexpr (std::integral<T>) {
                    if constexpr (std::is_signed_v<T>) {
                        if constexpr (sizeof(T) == 4) {
                            return from_vec<T>(vzip2_s32(to_vec(a), to_vec(b)));
                        } else if constexpr (sizeof(T) == 8) {
                            return from_vec<T>(vzip2q_s64(to_vec(a), to_vec(b)));
                        }
                    } else {
                        if constexpr (sizeof(T) == 4) {
                            return from_vec<T>(vzip2_u32(to_vec(a), to_vec(b)));
                        } else if constexpr (sizeof(T) == 8) {
                            return from_vec<T>(vzip2q_u64(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif

                return { a[1], b[1] };
            }

            template <std::size_t N, typename T>
                requires (N > 2)
            UI_ALWAYS_INLINE auto high(
                Vec<N, T> const& a,
                Vec<N, T> const& b
            ) const noexcept {
                #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip2q_f32(to_vec(a), to_vec(b)));
                    }
                } else if constexpr (std::same_as<T, float16>) {
                    #ifdef UI_HAS_FLOAT_16
                    if constexpr (N == 4) {
                        return from_vec<T>(vzip2_f16(to_vec(a), to_vec(b)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vzip2q_f16(to_vec(a), to_vec(b)));
                    }
                    #else
                    auto l = rcast<std::uint16_t>(a);
                    auto r = rcast<std::uint16_t>(b);
                    using ret_t = decltype(high(l, r));
                    if constexpr (!std::is_void_v<ret_t>) {
                        return rcast<T>(high(l, r));
                    }
                    #endif
                } else if constexpr (std::same_as<T, bfloat16>) {
                    auto l = rcast<std::uint16_t>(a);
                    auto r = rcast<std::uint16_t>(b);
                    using ret_t = decltype(high(l, r));
                    if constexpr (!std::is_void_v<ret_t>) {
                        return rcast<T>(high(l, r));
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
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip2q_s32(to_vec(a), to_vec(b)));
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
                        if constexpr (N == 4) {
                            return from_vec<T>(vzip2q_u32(to_vec(a), to_vec(b)));
                        }
                    }
                }
                #endif
            }
        };

        template <std::size_t N, typename T>
        UI_ALWAYS_INLINE auto zipping_helper(
            Vec<N, T> const& a,
            Vec<N, T> const& b
        ) noexcept -> Vec<2 * N, T> {
            auto const zip = internal::zip_helper{};
            if constexpr (!(
                std::is_void_v<decltype(zip.low(a,b))> ||
                std::is_void_v<decltype(zip.high(a,b))>
            )) {
                return join(zip.low(a, b), zip.high(a, b));
            } else {
                using ret_low_t = decltype(zip.low(a.lo, b.lo));
                if constexpr (std::is_void_v<ret_low_t>) {
                    return join(zipping_helper(a.lo, b.lo), zipping_helper(a.hi, b.hi));
                } else {
                    using ret_high_t = decltype(zip.high(a.hi, b.hi));
                    if constexpr (std::is_void_v<ret_high_t>) {
                        return join(zip.low(a.lo, b.lo), zipping_helper(a.hi, b.hi));
                    } else {
                        return join(
                            join(zip.low(a.lo, b.lo), zip.high(a.lo, b.lo)),
                            join(zip.low(a.hi, b.hi), zip.high(a.hi, b.hi))
                        );
                    }
                }
            }
        }
    } // namespace internal

    /*
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_low(a, b) == Vec<4, int>::load(0, 4, 1, 5))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        auto const zip = internal::zip_helper{};
        using ret_low_t = decltype(zip.low(a, b));
        if constexpr (!std::is_void_v<ret_low_t>) {
            return internal::zip_helper{}.low(a, b);
        } else {
            return internal::zipping_helper(a.lo, b.lo);
        }
    }

    /**
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_low(a, b) == Vec<4, int>::load(2, 6, 3, 7))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto zip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        auto const zip = internal::zip_helper{};
        using ret_low_t = decltype(zip.high(a, b));
        if constexpr (!std::is_void_v<ret_low_t>) {
            return internal::zip_helper{}.high(a, b);
        } else {
            return internal::zipping_helper(a.hi, b.hi);
        }
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
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp1_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vuzp1q_f16(to_vec(a), to_vec(b)));
                }
                #else
                return rcast<T>(unzip_low(rcast<std::uint16_t>(a), rcast<std::uint16_t>(b)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return rcast<T>(unzip_low(rcast<std::uint16_t>(a), rcast<std::uint16_t>(b)));
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
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vuzp2_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vuzp2q_f16(to_vec(a), to_vec(b)));
                }
                #else
                return rcast<T>(unzip_high(rcast<std::uint16_t>(a), rcast<std::uint16_t>(b)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return rcast<T>(unzip_high(rcast<std::uint16_t>(a), rcast<std::uint16_t>(b)));
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
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn1_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vtrn1q_f16(to_vec(a), to_vec(b)));
                }
                #else
                return rcast<T>(transpose_low(
                    rcast<std::uint16_t>(a),
                    rcast<std::uint16_t>(b)
                ));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return rcast<T>(transpose_low(
                    rcast<std::uint16_t>(a),
                    rcast<std::uint16_t>(b)
                ));
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
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vtrn2_f16(to_vec(a), to_vec(b)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vtrn2q_f16(to_vec(a), to_vec(b)));
                }
                #else
                return rcast<T>(transpose_high(
                    rcast<std::uint16_t>(a),
                    rcast<std::uint16_t>(b)
                ));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return rcast<T>(transpose_high(
                    rcast<std::uint16_t>(a),
                    rcast<std::uint16_t>(b)
                ));
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

} // namespace ui::arm::neon

namespace ui {
// MARK: IntMask
    template <std::size_t N, typename T>
    inline constexpr IntMask<N, T>::IntMask(mask_t<N, T> const& m) noexcept {
        using namespace arm::neon;

        using mtype = mask_inner_t<T>;
        if constexpr (is_packed) {
            if constexpr (sizeof(T) == 1) {
                static const constexpr std::uint8_t md[16] = {
                      1 << 0, 1 << 1, 1 << 2, 1 << 3,
                      1 << 4, 1 << 5, 1 << 6, 1 << 7,
                      1 << 0, 1 << 1, 1 << 2, 1 << 3,
                      1 << 4, 1 << 5, 1 << 6, 1 << 7,
                };

                auto ext = rcast<mtype>(shift_right<7>(rcast<std::make_signed_t<T>>(m)));
                auto masked = bitwise_and(Vec<N, mtype>::load(md, N), ext); 

                if constexpr (N == 16) {
                    auto t0 = vzip_u8(to_vec(masked.lo), to_vec(masked.hi));
                    auto t1 = rcast<base_type>(join(from_vec<T>(t0.val[0]), from_vec<T>(t0.val[1])));
                    mask = static_cast<base_type>(vaddvq_u16(to_vec(t1)));
                } else if constexpr (N == 8) {
                    mask = static_cast<base_type>(vaddv_u8(to_vec(masked)));
                }
            } else {
                auto ext = rcast<mtype>(shift_right<7>(rcast<std::make_signed_t<T>>(m)));
                auto helper = [&ext]<std::size_t... Is>(std::index_sequence<Is...>) -> base_type {
                    auto res = base_type{};
                    ((res |= (base_type((ext[Is] & 1) << Is))),...);
                    return res;
                };
                mask = helper(std::make_index_sequence<N>{});
            }
        } else {
            auto tmp = rcast<std::uint16_t>(m);
            auto s = narrowing_shift_right<4>(tmp);
            mask = std::bit_cast<base_type>(rcast<std::uint64_t>(s));
        }
    }

// !MARK
} // namespace ui

#endif // AMT_UI_ARCH_ARM_MANIPULATION_HPP
