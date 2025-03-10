#ifndef AMT_UI_ARCH_WASM_CAST_HPP
#define AMT_UI_ARCH_WASM_CAST_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../basic.hpp"
#include "basic.hpp"
#include "../../vec_headers.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include "../emul/cast.hpp"
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <wasm_simd128.h>

namespace ui::wasm {
    using emul::rcast;

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto to_vec(Vec<N, T> const& v) noexcept {
        return std::bit_cast<v128_t>(v);
    }

    template <typename T, std::size_t N = sizeof(v128_t) / sizeof(T)>
    UI_ALWAYS_INLINE constexpr auto from_vec(v128_t v) noexcept -> Vec<N, T> {
        return std::bit_cast<Vec<N, T>>(v); 
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto fit_to_vec(Vec<N, T> const& v) noexcept -> v128_t {
        static constexpr auto bits = sizeof(T) * N;
        if constexpr (bits == sizeof(v128_t)) {
            return v;
        } else if constexpr (bits == sizeof(std::int64_t)) {
            return wasm_i64x2_make(std::bit_cast<std::int64_t>(v), 0);
        } else if constexpr (bits == sizeof(std::int32_t)) {
            return wasm_i32x4_make(std::bit_cast<std::int32_t>(v), 0, 0, 0);
        } else if constexpr (bits == sizeof(std::int16_t)) {
            return wasm_i16x8_make(std::bit_cast<std::int16_t>(v), 0, 0, 0, 0, 0, 0, 0);
        } else if constexpr (bits == sizeof(std::int8_t)) {
            return wasm_i8x16_make(
                std::bit_cast<std::int8_t>(v), 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0
            );
        }
    }

    namespace internal {
        using namespace ::ui::internal;

        template <typename To, bool S, typename M, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto iter(Vec<N, T> const& v, M const& m) noexcept -> Vec<N, To> {
            if constexpr (N == 1) {
                if constexpr (S) return emul::sat_cast<To>(v);
                else return emul::cast<To>(v);
            } else {
                if constexpr (is_case_invocable<N, M, decltype(v)>) {
                    auto temp = m.template match<N>(v);
                    if constexpr (is_vec<decltype(temp)>) return temp;
                    else return from_vec<To>(temp);
                } else {
                    return join(
                        iter<To, S>(v.lo, m),
                        iter<To, S>(v.hi, m)
                    );
                }
            }
        }

        template <typename To, bool Saturating = false, bool ClampFp = true>
        struct CastImpl {
            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int8_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        return v;
                    } else if constexpr (sizeof(To) == 2) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<8> = [](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                return from_vec<To>(wasm_i16x8_extend_low_i8x16(m)).lo;
                            },
                            case_maker<16> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto lo = wasm_i16x8_extend_low_i8x16(m);
                                auto hi = wasm_i16x8_extend_high_i8x16(m);
                                return join(
                                    from_vec<To>(lo),
                                    from_vec<To>(hi)
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 4) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = fit_to_vec(v_);
                                    auto z = wasm_i8x16_const_splat(0);
                                    m = wasm_i8x16_max(z, m);
                                    return from_vec<To>(m).lo;
                                },
                                case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_i8x16_const_splat(0);
                                    m = wasm_i8x16_max(z, m);
                                    return from_vec<To>(m);
                                }
                            });
                        }
                        return rcast<To>(v);
                    } else if constexpr (sizeof(To) == 2) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<8> = [](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                if constexpr (Saturating) {
                                    auto z = wasm_i8x16_const_splat(0);
                                    m = wasm_i8x16_max(z, m);
                                }
                                return from_vec<To>(wasm_i16x8_extend_low_i8x16(m)).lo;
                            },
                            case_maker<16> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                if constexpr (Saturating) {
                                    auto z = wasm_i8x16_const_splat(0);
                                    m = wasm_i8x16_max(z, m);
                                }
                                return join(
                                    from_vec<To>(wasm_i16x8_extend_low_i8x16(m)),
                                    from_vec<To>(wasm_i16x8_extend_high_i8x16(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 4) {
                        auto temp = CastImpl<std::int16_t>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint8_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To, Saturating>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                            return iter<To, Saturating>(temp, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    m = wasm_u8x16_min(m, wasm_u8x16_const_splat(std::numeric_limits<To>::max()));
                                    return from_vec<To>(
                                        wasm_u8x16_narrow_i16x8(m, m)
                                    ).lo;
                                },
                                case_maker<16> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    lo = wasm_u8x16_min(lo, wasm_u8x16_const_splat(std::numeric_limits<To>::max()));
                                    hi = wasm_u8x16_min(hi, wasm_u8x16_const_splat(std::numeric_limits<To>::max()));
                                    return from_vec<To>(
                                        wasm_u8x16_narrow_i16x8(lo, hi)
                                    );
                                }
                            });
                        }
                        return rcast<std::int8_t>(v);
                    } else if constexpr (sizeof(To) == 2) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<8> = [](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                return from_vec<To>(wasm_u16x8_extend_low_u8x16(m));
                            },
                            case_maker<16> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return join(
                                    from_vec<To>(wasm_u16x8_extend_low_u8x16(m)),
                                    from_vec<To>(wasm_u16x8_extend_high_u8x16(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 4) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To>{}(temp);
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                        return CastImpl<To>{}(temp);
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        return v;
                    } else if constexpr (sizeof(To) == 2) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<8> = [](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                return from_vec<To>(wasm_u16x8_extend_low_u8x16(m));
                            },
                            case_maker<16> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return join(
                                    from_vec<To>(wasm_u16x8_extend_low_u8x16(m)),
                                    from_vec<To>(wasm_u16x8_extend_high_u8x16(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 4) {
                        auto temp = CastImpl<std::uint16_t>{}(v);
                        return CastImpl<To>{}(temp);
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::uint32_t>{}(v);
                        return CastImpl<To>{}(temp);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int16_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    return from_vec<To>(wasm_i8x16_narrow_i16x8(m, m)).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    return from_vec<To>(
                                        wasm_i8x16_narrow_i16x8(lo, hi)
                                    );
                                }
                            });
                        }

                        constexpr auto fn1 = [](v128_t m) {
                            auto mask = wasm_i16x8_const_splat(0xFF);
                            m = wasm_v128_and(m, mask);
                            auto res = wasm_i8x16_shuffle(m, m,
                                      0, 2, 4, 6, 8, 10, 12, 14,
                                      16, 18, 20, 22, 24, 26, 28, 30);
                            return from_vec<To>(res).lo;
                        };
                        constexpr auto fn2 = [](v128_t lo, v128_t hi) {
                            auto mask = wasm_i16x8_const_splat(0xFF);
                            lo = wasm_v128_and(lo, mask);
                            hi = wasm_v128_and(hi, mask);
                            auto res = wasm_i8x16_shuffle(lo, hi,
                                      0, 2, 4, 6, 8, 10, 12, 14,
                                      16, 18, 20, 22, 24, 26, 28, 30);
                            return from_vec<To>(res);
                        };

                        return iter<To, false>(v, Matcher {
                            case_maker<8> = [fn=fn1](auto const& v_) {
                                auto m = to_vec(v_);
                                return fn(m);
                            },
                            case_maker<16> = [fn=fn2](auto const& v_) {
                                auto lo = to_vec(v_.lo);
                                auto hi = to_vec(v_.hi);
                                return fn(lo, hi);
                            }
                        });
                    } else if constexpr (sizeof(To) == 2)  {
                        return v;
                    } else if constexpr (sizeof(To) == 4) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<4> = [](auto const& v_) {
                                auto m = wasm_i32x4_extend_low_i16x8(fit_to_vec(v_)); 
                                return m;
                            },
                            case_maker<8> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return join(
                                    from_vec<To>(wasm_i32x4_extend_low_i16x8(m)),
                                    from_vec<To>(wasm_i32x4_extend_high_i16x8(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                        return CastImpl<To>{}(temp);
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    return from_vec<To>(wasm_u8x16_narrow_i16x8(m, m));
                                },
                                case_maker<16> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    return from_vec<To>(
                                        wasm_u8x16_narrow_i16x8(lo, hi)
                                    );
                                }
                            });
                        }
                        auto temp = CastImpl<std::int8_t>{}(v);
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 2)  {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto m = fit_to_vec(v_);
                                    auto z = wasm_i16x8_const_splat(0);
                                    return from_vec<To>(wasm_i16x8_max(m, z)).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_i16x8_const_splat(0);
                                    return from_vec<To>(wasm_i16x8_max(m, z));
                                }
                            });
                        }
                        return rcast<To>(v);
                    } else if constexpr (sizeof(To) == 4) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<4> = [](auto const& v_) {
                                auto m = fit_to_vec(v_); 
                                if constexpr (Saturating) {
                                    auto z = wasm_i16x8_const_splat(0);
                                    m = wasm_i16x8_max(m, z);
                                }
                                return wasm_i32x4_extend_low_i16x8(m);
                            },
                            case_maker<8> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                if constexpr (Saturating) {
                                    auto z = wasm_i16x8_const_splat(0);
                                    m = wasm_i16x8_max(m, z);
                                }
                                return join(
                                    from_vec<To>(wasm_i32x4_extend_low_i16x8(m)),
                                    from_vec<To>(wasm_i32x4_extend_high_i16x8(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint16_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::int8_t>::max());
                                    m = wasm_u16x8_min(m, z);
                                    return from_vec<To>(wasm_i8x16_narrow_i16x8(m, m)).lo;
                                },
                                case_maker<16> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::int8_t>::max());
                                    lo = wasm_u16x8_min(z, lo);
                                    hi = wasm_u16x8_min(z, hi);
                                    return from_vec<To>(
                                        wasm_i8x16_narrow_i16x8(lo, hi)
                                    );
                                }
                            });
                        }
                        auto temp = CastImpl<std::int8_t>{}(rcast<std::int16_t>(v));
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto m = fit_to_vec(v_);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::int16_t>::max());
                                    m = wasm_u16x8_min(m, z);
                                    return from_vec<To>(m).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::int16_t>::max());
                                    m = wasm_u16x8_min(z, m);
                                    return from_vec<To>(m);
                                }
                            });
                        }
                        return rcast<To>(v);
                    } else if constexpr (sizeof(To) == 4) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<4> = [](auto const& v_) {
                                auto m = wasm_u32x4_extend_low_u16x8(fit_to_vec(v_)); 
                                return m;
                            },
                            case_maker<8> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return join(
                                    from_vec<To>(wasm_u32x4_extend_low_u16x8(m)),
                                    from_vec<To>(wasm_u32x4_extend_high_u16x8(m))
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::int32_t>{}(v);
                        return CastImpl<To>{}(temp);
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    auto m = fit_to_vec(v_);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::uint8_t>::max());
                                    m = wasm_u16x8_min(m, z);
                                    return from_vec<To>(wasm_u8x16_narrow_i16x8(m, m)).lo;
                                },
                                case_maker<16> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    auto z = wasm_u16x8_const_splat(std::numeric_limits<std::uint8_t>::max());
                                    lo = wasm_u16x8_min(z, lo);
                                    hi = wasm_u16x8_min(z, hi);
                                    return from_vec<To>(
                                        wasm_u8x16_narrow_i16x8(lo, hi)
                                    );
                                }
                            });
                        }
                        auto temp = CastImpl<std::int8_t>{}(v);
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        return v;
                    } else if constexpr (sizeof(To) == 4) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<4> = [](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                return from_vec<To>(wasm_u32x4_extend_low_u16x8(m));
                            },
                            case_maker<8> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto lo = wasm_u32x4_extend_low_u16x8(m);
                                auto hi = wasm_u32x4_extend_high_u16x8(m);
                                return join(
                                    from_vec<To>(lo),
                                    from_vec<To>(hi)
                                );
                            }
                        });
                    } else if constexpr (sizeof(To) == 8) {
                        auto temp = CastImpl<std::uint32_t>{}(v);
                        return CastImpl<To>{}(temp);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int32_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return iter<To, Saturating>(v, Matcher {
                        case_maker<2> = [](auto const& v_) {
                            auto m = fit_to_vec(v_);
                            return from_vec<To>(wasm_f32x4_convert_i32x4(m)).lo;
                        },
                        case_maker<4> = [](auto const& v_) {
                            return wasm_f32x4_convert_i32x4(to_vec(v_));
                        }
                    });
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    return from_vec<To>(wasm_i16x8_narrow_i32x4(m, m)).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    return from_vec<To>(
                                        wasm_i16x8_narrow_i32x4(lo, hi)
                                    );
                                }
                            });
                        }

                        constexpr auto fn = [](v128_t m) {
                            auto mask = wasm_i32x4_const_splat(0xFFFF);
                            m = wasm_v128_and(m, mask);
                            auto res = wasm_i16x8_shuffle(m, m,
                                0,  2,  4,  6,
                                8, 10, 12, 14
                            );
                            return from_vec<To>(res).lo;
                        };

                        return iter<To, false>(v, Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                auto m = fit_to_vec(v_);
                                return fn(m);
                            },
                            case_maker<4> = [fn](auto const& v_) {
                                auto m = to_vec(v_);
                                return fn(m);
                            }
                        });
                    } else if constexpr (sizeof(To) == 4) {
                        return v;
                    } else if constexpr (sizeof(To) == 8) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<2> = [](auto const& v_) {
                                auto m = wasm_i64x2_extend_low_i32x4(fit_to_vec(v_)); 
                                return m;
                            },
                            case_maker<4> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return join(
                                    from_vec<To>(wasm_i64x2_extend_low_i32x4(m)),
                                    from_vec<To>(wasm_i64x2_extend_high_i32x4(m))
                                );
                            }
                        });
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_u32x4_const_splat(std::numeric_limits<std::uint16_t>::max());
                                    m = wasm_i32x4_min(m, z);
                                    return from_vec<To>(wasm_u16x8_narrow_i32x4(m, m)).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    auto z = wasm_u32x4_const_splat(std::numeric_limits<std::uint16_t>::max());
                                    lo = wasm_i32x4_min(z, lo);
                                    hi = wasm_i32x4_min(z, hi);
                                    return from_vec<To>(
                                        wasm_u16x8_narrow_i32x4(lo, hi)
                                    );
                                }
                            });
                        }
                        auto temp = CastImpl<std::make_signed_t<To>>{}(v);
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto z = wasm_i32x4_const_splat(0);
                                    return from_vec<To>(wasm_i32x4_max(m, z));
                                }
                            });
                        }
                        return rcast<To>(v);
                    } else if constexpr (sizeof(To) == 8) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<4> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                if constexpr (Saturating) {
                                    auto z = wasm_i32x4_const_splat(0);
                                    m = wasm_i32x4_max(m, z);
                                }
                                auto lo = wasm_i64x2_extend_low_i32x4(m);
                                auto hi = wasm_i64x2_extend_high_i32x4(m);
                                return join(
                                    from_vec<To>(lo),
                                    from_vec<To>(hi)
                                );
                            }
                        });
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint32_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return iter<To, Saturating>(v, Matcher {
                        case_maker<2> = [](auto const& v_) {
                            auto m = fit_to_vec(v_);
                            return from_vec<To>(wasm_f32x4_convert_u32x4(m)).lo;
                        },
                        case_maker<4> = [](auto const& v_) {
                            return wasm_f32x4_convert_u32x4(to_vec(v_));
                        }
                    });
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto mx = wasm_u32x4_const_splat(std::numeric_limits<std::int16_t>::max());
                                    auto m = wasm_u32x4_min(to_vec(v_), mx);
                                    return from_vec<To>(
                                        wasm_i16x8_narrow_i32x4(m, m)
                                    ).lo;
                                },
                                case_maker<8> = [](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    auto mx = wasm_u32x4_const_splat(std::numeric_limits<std::int16_t>::max());
                                    lo = wasm_u32x4_min(lo, mx);
                                    hi = wasm_u32x4_min(hi, mx);
                                    return from_vec<To>(
                                        wasm_i16x8_narrow_i32x4(lo, hi)
                                    );
                                }
                            });
                        }
                        auto temp = CastImpl<std::make_unsigned_t<To>>{}(v);
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [](auto const& v_) {
                                    auto mx = wasm_u32x4_const_splat(std::numeric_limits<std::int32_t>::max());
                                    auto m = wasm_u32x4_min(to_vec(v_), mx);
                                    return m;
                                }
                            });
                        }
                        return rcast<To>(v);
                    } else if constexpr (sizeof(To) == 8) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<2> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return from_vec<To>(wasm_u64x2_extend_low_u32x4(m, m));
                            },
                            case_maker<4> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto lo = wasm_u64x2_extend_low_u32x4(m);
                                auto hi = wasm_u64x2_extend_high_u32x4(m);
                                return join(
                                    from_vec<To>(lo),
                                    from_vec<To>(hi)
                                );
                            }
                        });
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        if constexpr (Saturating) {
                            constexpr auto convert = [](auto const& lo, auto const& hi) {
                                return from_vec<To>(wasm_i16x8_shuffle(lo, hi,
                                  0, 2, 4, 6, 8, 10, 12, 14));
                            };
                            return iter<To, Saturating>(v, Matcher {
                                case_maker<4> = [convert](auto const& v_) {
                                    auto m = to_vec(v_);
                                    auto mx = wasm_u32x4_const_splat(std::numeric_limits<std::uint16_t>::max());
                                    m = wasm_u32x4_min(m, mx);
                                    return convert(m, m).lo;
                                },
                                case_maker<8> = [convert](auto const& v_) {
                                    auto lo = to_vec(v_.lo);
                                    auto hi = to_vec(v_.hi);
                                    auto mx = wasm_u32x4_const_splat(std::numeric_limits<std::uint16_t>::max());
                                    lo = wasm_u32x4_min(lo, mx);
                                    hi = wasm_u32x4_min(hi, mx);
                                    return convert(lo, hi);
                                }
                            });
                        }

                        return CastImpl<To>{}(rcast<std::int32_t>(v));
                    } else if constexpr (sizeof(To) == 4) {
                        return v;
                    } else if constexpr (sizeof(To) == 8) {
                        return iter<To, Saturating>(v, Matcher {
                            case_maker<2> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return from_vec<To>(wasm_u64x2_extend_low_u32x4(m, m));
                            },
                            case_maker<4> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto lo = wasm_u64x2_extend_low_u32x4(m);
                                auto hi = wasm_u64x2_extend_high_u32x4(m);
                                return join(
                                    from_vec<To>(lo),
                                    from_vec<To>(hi)
                                );
                            }
                        });
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int64_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return emul::cast<float>(v);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return emul::sat_cast<std::int32_t>(v);
                        }
                        return iter<To, false>(v, Matcher {
                            case_maker<2> = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto mask = wasm_u64x2_const_splat(0xFFFF'FFFFul);
                                m = wasm_v128_and(m, mask);
                                auto res = wasm_i32x4_shuffle(m, m, 0, 2, 4, 6);
                                return from_vec<To>(res).lo;
                            }
                        });
                    } else if constexpr (sizeof(To) == 8) {
                        return v;
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return emul::sat_cast<std::uint32_t>(v);
                        }
                        auto temp = CastImpl<std::make_signed_t<To>>{}(v);
                        return rcast<To>(temp);
                    } else if constexpr (sizeof(To) == 8) {
                        if constexpr (Saturating) return emul::sat_cast<To>(v);
                        return emul::cast<To>(v);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint64_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return emul::cast<float>(v);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return emul::sat_cast<std::int32_t>(v);
                        }
                        return CastImpl<To>{}(rcast<std::int64_t>(v));
                    } else {
                        if constexpr (Saturating) return emul::sat_cast<To>(v);
                        return emul::cast<To>(v);;
                    }
                } else {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                        return CastImpl<To, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 4) {
                        if constexpr (Saturating) {
                            return emul::sat_cast<std::uint32_t>(v);
                        }
                        return CastImpl<To>{}(rcast<std::int64_t>(v));
                    } else {
                        return v;
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, float> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    return cast_float32_to_float16(v);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    return cast_float32_to_bfloat16(v);
                } else if constexpr (std::same_as<To, float>) {
                    return v;
                } else if constexpr (std::same_as<To, double>) {
                    return iter<To, Saturating>(v, Matcher {
                        case_maker<2> = [](auto const& v_) {
                            auto m = fit_to_vec(v_);
                            return wasm_f64x2_promote_low_f32x4(m);
                        },
                        case_maker<4> = [](auto const& v_) {
                            auto lo = from_vec<To>(wasm_f64x2_promote_low_f32x4(to_vec(v_)));
                            auto hi = from_vec<To>(wasm_f64x2_promote_low_f32x4(to_vec(join(v_.hi, v_.hi))));
                            return join(lo, hi); 
                        }
                    });
                } else if constexpr (std::is_signed_v<To>) {
                    auto temp = iter<std::int32_t, Saturating>(v, Matcher {
                        case_maker<4> = [](auto const& v_) {
                            return wasm_i32x4_trunc_sat_f32x4(to_vec(v_));
                        }
                    });

                    if constexpr (sizeof(To) == 4) return temp;
                    else {
                        auto t0 = CastImpl<To>{}(temp);
                        if constexpr (ClampFp) {
                            return map([](auto n, auto v_) {
                                if (v_ == INFINITY) return std::numeric_limits<To>::max();
                                if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                                return n;
                            }, t0, v);
                        }
                        return t0;
                    }
                } else {
                    auto temp = iter<std::uint32_t, Saturating>(v, Matcher {
                        case_maker<4> = [](auto const& v_) {
                            return wasm_u32x4_trunc_sat_f32x4(to_vec(v_));
                        }
                    });
                    if constexpr (sizeof(To) == 4) return temp;
                    else {
                        auto t0 = CastImpl<To>{}(temp);
                        if constexpr (ClampFp) {
                            return map([](auto n, auto v_) {
                                if (v_ == INFINITY) return std::numeric_limits<To>::max();
                                if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                                return n;
                            }, t0, v);
                        }
                        return t0;
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, double> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return iter<To, Saturating>(v, [](auto const& v_) {
                        return from_vec<To>(wasm_f32x4_demote_f64x2_zero(v_)).lo;
                    });
                } else if constexpr (std::same_as<To, double>) {
                    return v;
                } else if constexpr (std::is_signed_v<To>) {
                    auto temp = iter<std::int32_t, Saturating>(v, Matcher {
                        case_maker<2> = [](auto const& v_) {
                            return from_vec<std::int32_t>(wasm_i32x4_trunc_sat_f64x2_zero(to_vec(v_))).lo;
                        }
                    });
                    if constexpr (sizeof(To) == 4) return rcast<To>(temp);
                    else {
                        auto t0 = CastImpl<To>{}(temp);
                        if constexpr (ClampFp) {
                            return map([](auto n, auto v_) {
                                if (v_ == INFINITY) return std::numeric_limits<To>::max();
                                if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                                return n;
                            }, t0, v);
                        }
                        return t0;
                    }
                } else {
                    auto temp = iter<std::uint32_t, Saturating>(v, Matcher {
                        case_maker<2> = [](auto const& v_) {
                            return from_vec<std::uint32_t>(wasm_u32x4_trunc_sat_f64x2_zero(to_vec(v_))).lo;
                        }
                    });
                    if constexpr (sizeof(To) == 4) return temp;
                    else {
                        auto t0 = CastImpl<To>{}(temp);
                        if constexpr (ClampFp) {
                            return map([](auto n, auto v_) {
                                if (v_ == INFINITY) return std::numeric_limits<To>::max();
                                if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                                return n;
                            }, t0, v);
                        }
                        return t0;
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, float16> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    return v;
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return cast_float16_to_float32(v);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else {
                    auto temp = CastImpl<float, Saturating, false>{}(v);
                    auto t0 = CastImpl<To>{}(temp);
                    return map([](auto n, auto v_) {
                        if (v_ == INFINITY) return std::numeric_limits<To>::max();
                        if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                        return n;
                    }, t0, temp);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, bfloat16> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    return cast_bfloat16_to_float32(v);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<float, Saturating>{}(v);
                    return CastImpl<To>{}(temp);
                } else {
                    auto temp = CastImpl<float, Saturating, false>{}(v);
                    auto t0 = CastImpl<To, false>{}(temp);
                    return map([](auto n, auto v_) {
                        if (v_ == INFINITY) return std::numeric_limits<To>::max();
                        if (v_ == -INFINITY) return std::numeric_limits<To>::min();
                        return n;
                    }, t0, temp);
                }
            }
        };
    } // namespace internal

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return internal::CastImpl<To, false, true>{}(v);
    }

    template <typename To, std::size_t N, std::integral From>
    UI_ALWAYS_INLINE auto sat_cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return internal::CastImpl<To, true, true>{}(v);
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_CAST_HPP
