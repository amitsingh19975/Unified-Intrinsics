#ifndef AMT_UI_FLOAT_HPP
#define AMT_UI_FLOAT_HPP

#include "base.hpp"
#include "vec_headers.hpp"
#include "maths.hpp"
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <limits>

#ifdef UI_ARM_HAS_NEON
    #include <arm_neon.h>
#endif

namespace ui {

    namespace internal {
        #ifdef UI_HAS_FLOAT_16
            #if !defined(UI_ARM_HAS_NEON) || !defined(UI_CPU_ARM64)
                #ifdef UI_HAS_STD_FLOAT_HEADER
                    using float16_t = std::float16_t;
                #else
                    using float16_t = _Float16;
                #endif
            #else
                using float16_t = ::float16_t;
            #endif
        #else
            #define UI_HAS_CUSTOM_FLOAT16_IMPL
            using float16_t = std::uint16_t;
        #endif
    }

    struct alignas(sizeof(internal::float16_t)) float16;

    namespace fp {
        template <typename T>
        struct FloatingPointRep;

        template <>
        struct FloatingPointRep<float> {
            bool sign;
            std::int32_t exponent;
            std::uint32_t mantissa;
        };

        template <>
        struct FloatingPointRep<double> {
            bool sign;
            std::int32_t exponent;
            std::uint64_t mantissa;
        };

        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float n) noexcept -> FloatingPointRep<float> {
            auto bits = std::bit_cast<std::uint32_t>(n);
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }

            return {
                .sign = static_cast<bool>(bits >> 31),
                .exponent = static_cast<int>((bits >> 23) & 0xFF) - 127,
                .mantissa = bits & 0x7FFFFF,
            };
        }

        UI_ALWAYS_INLINE static constexpr auto decompose_fp(double n) noexcept -> FloatingPointRep<double> {
            auto bits = std::bit_cast<std::uint64_t>(n);
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }

            return {
                .sign = static_cast<bool>(bits >> 63),
                .exponent = static_cast<std::int32_t>((bits >> 52) & 0x7FF) - 1023,
                .mantissa = bits & 0xFFFFFFFFFFFFF,
            };
        }
        
        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float> fp) noexcept -> float {
            int biased_exp = fp.exponent + 127;
            
            std::uint32_t exp_field = static_cast<std::uint32_t>(biased_exp) & 0xFF;
            
            auto bits = (static_cast<std::uint32_t>(fp.sign) << 31)
                      | (exp_field << 23)
                      | fp.mantissa;
            
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }
            return std::bit_cast<float>(bits);
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<double> fp) noexcept -> double {
            int biased_exp = fp.exponent + 1023;
            
            std::uint64_t exp_field = static_cast<std::uint32_t>(biased_exp) & 0x7FF;
            
            auto bits = (static_cast<std::uint64_t>(fp.sign) << 63)
                      | (exp_field << 52)
                      | fp.mantissa;
            
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }
            return std::bit_cast<double>(bits);
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float16> fp) noexcept -> float16;
    }


    namespace fp {
        template <>
        struct FloatingPointRep<float16> {
            bool sign;
            std::int16_t exponent;
            std::uint16_t mantissa;
        };

        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float16 fp) noexcept -> FloatingPointRep<float16>;
        UI_ALWAYS_INLINE static constexpr auto compose_fp_helper(FloatingPointRep<float16> fp) noexcept -> internal::float16_t {
            int biased_exp = fp.exponent + 15;
            auto exp_field = static_cast<std::uint16_t>(biased_exp) & 0x1F;
            
            auto bits = static_cast<std::uint16_t>((static_cast<std::uint16_t>(fp.sign) << 15)
                      | (exp_field << 10)
                      | (fp.mantissa & 0x3FF));
            
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }
            return std::bit_cast<internal::float16_t>(bits);
        }
    }

    struct alignas(sizeof(internal::float16_t)) float16 {
        using base_type = internal::float16_t;
        using fp_rep = fp::FloatingPointRep<float16>;

        static constexpr auto min_rep() noexcept -> base_type {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
                return fp::compose_fp_helper(fp_rep{ .sign = true, .exponent = 0, .mantissa = 0xFFFF });
            #else
                return static_cast<base_type>(FLT16_MIN);
            #endif
        }

        static constexpr auto max_rep() noexcept -> base_type {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
                return fp::compose_fp_helper(fp_rep{ .sign = true, .exponent = 15, .mantissa = 0xFFFF });
            #else
                return static_cast<base_type>(FLT16_MAX);
            #endif
        }

        static constexpr auto inf_rep(bool sign) noexcept -> base_type {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
                return fp::compose_fp_helper(fp_rep{ .sign = sign, .exponent = 16, .mantissa = 0 });
            #else
                return static_cast<base_type>(INFINITY * (sign ? -1 : 1));
            #endif
        }

        static constexpr auto quiet_nan_rep() noexcept -> base_type {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
                return fp::compose_fp_helper(fp_rep{ .sign = false, .exponent = 16, .mantissa = 0x1 });
            #else
                return static_cast<base_type>(NAN);
            #endif
        }

        struct internal_t {};

        base_type data;

        constexpr float16() noexcept = default;
        constexpr float16(float16 const&) noexcept = default;
        constexpr float16(float16 &&) noexcept = default;
        constexpr float16& operator=(float16 const&) noexcept = default;
        constexpr float16& operator=(float16 &&) noexcept = default;
        constexpr ~float16() noexcept = default;

        constexpr float16(base_type data, internal_t) noexcept
            : data(data)
        {}

        #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
        constexpr float16_t(float f) noexcept {
            static constexpr auto pinf = inf_rep(true);
            static constexpr auto ninf = inf_rep(false);
            static constexpr auto quiet_nan = quiet_nan_rep();

            auto [sign, exp, man] = fp::decompose_fp(f);
            if (std::isinf(f)) {
                data = sign ? ninf : pinf; 
                return;
            }

            if (std::isnan(f)) {
                data = quiet_nan;
                return;
            } else if (f == 0.f) {
                data = 0;
                return;
            }

            base_type hman{};
            std::int16_t hexp{};
            if (exp != 0) {
                auto nexp = static_cast<int>(exp);
                if (nexp >= 31) {
                    data = sign ? ninf : pinf;
                    return;
                }
                if (nexp <= 0) {
                    int shift = 13 - nexp;
                    hman = static_cast<base_type>((man | (1u << 23)) >> shift);
                    if (shift > 24) {
                        hman = 0;
                    }
                    hexp = 0;
                } else {
                    hexp = static_cast<std::int16_t>(nexp << 10);
                    hman = static_cast<base_type>(man >> 13);
                }

                base_type remainder = man & 0x1FFF;
                if (remainder > 0x1000 || (remainder == 0x1000 && (hman & 1))) {
                    hman++;
                    if (hman >= 0x400) {  // Mantissa overflow
                        hman = 0;
                        hexp += (1 << 10);
                        if (hexp >= (31 << 10)) {  // Exponent overflow
                            data = sign ? ninf : pinf;
                            return;
                        }
                    }
                }
            }
            set_data(sign, hexp >> 10, hman);
        }
        #else
        constexpr float16(float f) noexcept
            : data(static_cast<base_type>(f))
        {}
        #endif

        constexpr float16(double f) noexcept
            : float16(static_cast<float>(f))
        {}

        constexpr float16(std::convertible_to<float> auto v) noexcept
            : float16(static_cast<float>(v))
        {}

        #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
        constexpr operator float() const noexcept {
            auto [s, e, m] = fp::decompose_fp(*this); 
            fp::FloatingPointRep<float> temp { .sign = s, .exponent = 0, .mantissa = 0};
    
            if (e == 16) {
                if (m == 0) return s ? -INFINITY : INFINITY;
                else return NAN;
            } else if (e == -15) {
                if (m == 0) {
                    return 0;
                } else {
                    int tmp_e = -14;
                    std::uint32_t tmp_m = m;
                    while ((tmp_m & 0x400) == 0) {
                        tmp_m <<= 1;
                        tmp_e--;
                    }
                    tmp_m &= 0x3FF;
                    temp.exponent = tmp_e;
                    temp.mantissa = tmp_m << 13;
                }
            } else {
                temp.exponent = e;
                temp.mantissa = static_cast<uint32_t>(m << 13);
            }

            return fp::compose_fp(temp);
        }
        #else
        explicit constexpr operator float() const noexcept {
            return static_cast<float>(data);
        }
        #endif

        explicit constexpr operator double() const noexcept {
            float temp = static_cast<float>(*this);
            return static_cast<double>(temp);
        }

        explicit constexpr operator base_type() const noexcept {
            return data;
        }
        
        template <typename T>
            requires (std::convertible_to<float, T>)
        explicit constexpr operator T() const noexcept {
            return static_cast<T>(static_cast<float>(*this));
        }

        friend constexpr auto operator-(float16 val) noexcept -> float16 {
            auto [s, e, m] = fp::decompose_fp(val);
            val.set_data(!s, e, m);
            return val;
        }

        constexpr auto is_nan() const noexcept -> bool {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
            auto [s, e, m] = fp::decompose_fp(*this);
            (void)s;
            return (e == 16) && m != 0;
            #else
            return std::isnan(float(data));
            #endif
        }

        constexpr auto is_zero() const noexcept -> bool {
            auto f = fp::decompose_fp(*this);
            return f.exponent == -15 && f.mantissa == 0;
        }

        constexpr auto is_neg() const noexcept -> bool {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
            auto f = fp::decompose_fp(*this);
            return f.sign;
            #else
            auto temp = float(data);
            return temp < 0;
            #endif
        }

        constexpr auto is_inf(bool sign) const noexcept -> bool {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
            static constexpr auto pinf = inf_rep(true);
            static constexpr auto ninf = inf_rep(false);
            return sign ? data == ninf : data == pinf;
            #else
            auto temp = float(data);
            return std::isinf(temp) && (sign ? temp < 0 : temp >= 0);
            #endif
        }

        constexpr auto is_inf() const noexcept -> bool {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
            auto f = fp::decompose_fp(*this);
            return f.exponent == 16 && f.mantissa == 0;
            #else
            auto temp = float(data);
            return std::isinf(temp);
            #endif
        }

    private:
        constexpr auto set_data(bool sign, std::int16_t exp, std::uint16_t man) noexcept -> void {
            data = fp::compose_fp(fp::FloatingPointRep<float16>{ .sign = sign, .exponent = exp, .mantissa = man }).data;
        }
    };

    namespace fp {
        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float16 fp) noexcept -> FloatingPointRep<float16> {
            auto bits = std::bit_cast<std::uint16_t>(fp);
            if constexpr (std::endian::native == std::endian::big) {
                bits = maths::byteswap(bits);
            }
            return {
                .sign = static_cast<bool>(bits >> 15),
                .exponent = static_cast<std::int16_t>(((bits >> 10) & 0x1F) - 15),
                .mantissa = static_cast<std::uint16_t>(bits & 0x3FF),
            };
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float16> fp) noexcept -> float16 {
            return { compose_fp_helper(fp), float16::internal_t{} };
        }
    }

    constexpr bool isnan(float16 val) noexcept {
        return val.is_nan();
    }

    constexpr bool isinf(float16 val) noexcept {
        return val.is_inf();
    }

    constexpr int fpclassify(float16 val) noexcept {
        auto [s, e, m] = fp::decompose_fp(val);
        if(e == -15) {
            if (m == 0) return FP_ZERO;
            else return FP_SUBNORMAL;
        }

        if (e == 16) {
            if (m != 0) return FP_NAN;
            else return FP_INFINITE;
        }
        return FP_NORMAL;
    }

    constexpr bool signbit(float16 val) noexcept {
        return val.is_neg();
    }

    #if defined(UI_HAS_BFLOAT_16) && defined(UI_HAS_STD_FLOAT_HEADER)
        using float16_t = std::bfloat16_t;
    #else
        enum class bfloat16_t: std::uint16_t {};
    #endif
}

#if !defined(UI_HAS_CUSTOM_FLOAT16_IMPL) && defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
namespace ui::internal {
    UI_ALWAYS_INLINE auto load_f16(ui::float16 v) noexcept -> float16x4_t {
        auto temp = vdup_n_u16(std::bit_cast<std::uint16_t>(v));
        return std::bit_cast<float16x4_t>(temp);
    }
    UI_ALWAYS_INLINE auto get_first(float16x4_t m) noexcept -> float16_t {
        struct Wrapper {
            float16_t data[4];
        };
        auto temp = std::bit_cast<Wrapper>(m);
        return temp.data[0];
    }

    UI_ALWAYS_INLINE auto mask_get_first(uint16x4_t m) noexcept -> std::uint16_t {
        struct Wrapper {
            std::uint16_t data[4];
        };
        auto temp = std::bit_cast<Wrapper>(m);
        return temp.data[0];
    }

} // ui::internal
#endif


#include <format>

namespace std {
    #ifndef __STDCPP_FLOAT16_T__ 
    template <>
    struct is_floating_point<ui::float16>: std::true_type{}; 

    template <>
    struct is_signed<ui::float16>: std::true_type{}; 

    template <>
    class numeric_limits<ui::float16> {
        using type = ui::float16;
    public:
        static constexpr const bool is_specialized = true;
        static constexpr const bool is_signed   = true;
        static constexpr const int digits       = 11;
        static constexpr const int digits10     = 3;
        static constexpr const int max_digits10 = 2 + (digits * 30103l) / 100000l;
        #ifndef UI_HAS_CUSTOM_FLOAT16_IMPL
        static constexpr type min() _NOEXCEPT { return FLT16_MIN; }
        static constexpr type max() _NOEXCEPT { return FLT16_MAX; }
        #else
        static constexpr type min() noexcept { return type(type::min_rep(), type::internal_t{}); }
        static constexpr type max() noexcept { return type(type::max_rep(), type::internal_t{}); }
        #endif
        static constexpr type lowest() noexcept { return -max(); }

        static constexpr const bool is_integer = false;
        static constexpr const bool is_exact   = false;
        static constexpr const int radix       = FLT_RADIX;
        #ifndef UI_HAS_CUSTOM_FLOAT16_IMPL
        static constexpr type epsilon() noexcept { return FLT16_EPSILON; }
        static constexpr const int min_exponent   = FLT16_MIN_EXP;
        static constexpr const int min_exponent10 = FLT16_MIN_10_EXP;
        static constexpr const int max_exponent   = FLT16_MAX_EXP;
        static constexpr const int max_exponent10 = FLT16_MAX_10_EXP;
        #else
        static constexpr const int min_exponent   = -13;
        static constexpr const int min_exponent10 = -4;
        static constexpr const int max_exponent   = 16;
        static constexpr const int max_exponent10 = 4;
        static constexpr type epsilon() noexcept { return 0.0009765625f; }
        #endif
        static constexpr type round_error() noexcept { return 0.5F; }


        static constexpr const bool has_infinity                                         = true;
        static constexpr const bool has_quiet_NaN                                        = true;
        static constexpr const bool has_signaling_NaN                                    = false;
        static constexpr type infinity() noexcept { return type(type::inf_rep(false), type::internal_t{}); }
        static constexpr type quiet_NaN() noexcept { return type(type::quiet_nan_rep(), type::internal_t{} ); }

        static constexpr const bool is_iec559  = true;
        static constexpr const bool is_bounded = true;
        static constexpr const bool is_modulo  = false;

        static constexpr const bool traps = false;
#if (defined(__arm__) || defined(__aarch64__))
        static constexpr const bool tinyness_before = true;
#else
        static constexpr const bool tinyness_before = false;
#endif
        static constexpr const float_round_style round_style = std::numeric_limits<float>::round_style;
    };
    #endif

    template <>
    struct formatter<ui::float16>: formatter<float> {
        auto format(ui::float16 val, auto& ctx) const {
            return formatter<float>::format(static_cast<float>(val), ctx);
        }
    };

}


static constexpr auto operator==(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    if (lhs.is_nan() || rhs.is_nan()) return false;
    if (lhs.is_zero() == rhs.is_zero()) return true;
    return lhs.data == rhs.data;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return mask_get_first(vceq_f16(load_f16(lhs), load_f16(rhs))) == 0xffff;
        #else
        return lhs.data == rhs.data;
        #endif
    #endif
}

static constexpr auto operator!=(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    return !(lhs == rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator==(ui::float16 lhs, T rhs) noexcept -> bool {
    return lhs == ui::float16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator!=(ui::float16 lhs, T rhs) noexcept -> bool {
    return !(lhs == rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator==(T lhs, ui::float16 rhs) noexcept -> bool {
    return ui::float16(lhs) == rhs;
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator!=(T lhs, ui::float16 rhs) noexcept -> bool {
    return !(lhs == rhs);
}

static constexpr auto operator<(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    // 1. NaN != any number or NaN
    if (lhs.is_nan() || rhs.is_nan()) return false;
    // 2. lhs is +ve infinity is always bigger than any number
    if (lhs.is_inf(true)) {
        return false;
    }
    // 3. lhs is -ve inf < rhs is +ve inf
    if (lhs.is_inf(false)) {
        if (rhs.is_inf(false)) return false;
        return true;
    }
    // 4. +0 == -0
    if (lhs.is_zero() == rhs.is_zero()) return false;

    auto [ls, le, lm] = ui::fp::decompose_fp(lhs);
    auto [rs, re, rm] = ui::fp::decompose_fp(rhs);

    // 5. lhs sign -ve then rhs must've +ve sign. Otherwise, same sign
    if (ls != rs) return rs == false;

    // 6. lhs exponent is smaller than rhs exponent then rhs is bigger
    if (le != re) return le < re;

    // 7. right mantissa must be bigger
    return lm < rm;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return mask_get_first(vclt_f16(load_f16(lhs), load_f16(rhs))) == 0xffff;
        #else
        return lhs.data < rhs.data;
        #endif
    #endif
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator<(ui::float16 lhs, T rhs) noexcept -> bool {
    return lhs < ui::float16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator<(T lhs, ui::float16 rhs) noexcept -> bool {
    return ui::float16(lhs) < rhs;
}

static constexpr auto operator>(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    // 1. NaN != any number or NaN
    if (lhs.is_nan() || rhs.is_nan()) return false;

    // 2. lhs is +ve infinity is always bigger than any number
    if (lhs.is_inf(true)) {
        if (rhs.is_inf(true)) return false;
        return true;
    }

    // 3. lhs is -ve inf < rhs any number
    if (lhs.is_inf(false)) {
        return false;
    }

    // 4. +0 == -0
    if (lhs.is_zero() == rhs.is_zero()) return false;

    auto [ls, le, lm] = ui::fp::decompose_fp(lhs);
    auto [rs, re, rm] = ui::fp::decompose_fp(rhs);

    // 5. lhs sign -ve then rhs must've +ve sign. Otherwise, same sign
    if (ls != rs) return ls == false;

    // 6. lhs exponent is smaller than rhs exponent then rhs is bigger
    if (le != re) return le > re;

    // 7. right mantissa must be bigger
    return lm > rm;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return mask_get_first(vcgt_f16(load_f16(lhs), load_f16(rhs))) == 0xffff;
        #else
        return lhs.data > rhs.data;
        #endif
    #endif
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator>(ui::float16 lhs, T rhs) noexcept -> bool {
    return lhs > ui::float16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator>(T lhs, ui::float16 rhs) noexcept -> bool {
    return ui::float16(lhs) > rhs;
}

static constexpr auto operator<=(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    if (lhs == rhs) return true;
    return lhs < rhs;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return mask_get_first(vcle_f16(load_f16(lhs), load_f16(rhs))) == 0xffff;
        #else
        return lhs.data <= rhs.data;
        #endif
    #endif
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator<=(ui::float16 lhs, T rhs) noexcept -> bool {
    return lhs <= ui::float16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator<=(T lhs, ui::float16 rhs) noexcept -> bool {
    return ui::float16(lhs) <= rhs;
}

static constexpr auto operator>=(ui::float16 lhs, ui::float16 rhs) noexcept -> bool {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    if (lhs == rhs) return true;
    return lhs > rhs;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return mask_get_first(vcge_f16(load_f16(lhs), load_f16(rhs))) == 0xffff;
        #else
        return lhs.data >= rhs.data;
        #endif
    #endif
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator>=(ui::float16 lhs, T rhs) noexcept -> bool {
    return lhs >= ui::float16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::float16, T>
static constexpr auto operator>=(T lhs, ui::float16 rhs) noexcept -> bool {
    return ui::float16(lhs) >= rhs;
}

static constexpr auto operator+(ui::float16 lhs, ui::float16 rhs) noexcept -> ui::float16 {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    auto l = float(lhs);
    auto r = float(rhs);
    return l + r;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return { get_first(vadd_f16(load_f16(lhs), load_f16(rhs))), ui::float16::internal_t{} };
        #else
        return { static_cast<ui::float16_t::base_type>(lhs.data + rhs.data), ui::float16_t::internal_t{} };
        #endif
    #endif
}

static constexpr auto operator-(ui::float16 lhs, ui::float16 rhs) noexcept -> ui::float16 {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    auto l = float(lhs);
    auto r = float(rhs);
    return l - r;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return { get_first(vsub_f16(load_f16(lhs), load_f16(rhs))), ui::float16::internal_t{} };
        #else
        return { static_cast<ui::float16_t::base_type>(lhs.data - rhs.data), ui::float16_t::internal_t{} };
        #endif
    #endif
}

static constexpr auto operator*(ui::float16 lhs, ui::float16 rhs) noexcept -> ui::float16 {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    auto l = float(lhs);
    auto r = float(rhs);
    return l * r;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return { get_first(vmul_f16(load_f16(lhs), load_f16(rhs))), ui::float16::internal_t{} };
        #else
        return { static_cast<ui::float16_t::base_type>(lhs.data * rhs.data), ui::float16_t::internal_t{} };
        #endif
    #endif
}

static constexpr auto operator/(ui::float16 lhs, ui::float16 rhs) noexcept -> ui::float16 {
    #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
    auto l = float(lhs);
    auto r = float(rhs);
    return l / r;
    #else
        #if defined(UI_ARM_HAS_NEON) && defined(UI_CPU_ARM64)
        using namespace ui::internal;
        return { get_first(vdiv_f16(load_f16(lhs), load_f16(rhs))), ui::float16::internal_t{} };
        #else
        return { static_cast<ui::float16_t::base_type>(lhs.data / rhs.data), ui::float16_t::internal_t{} };
        #endif
    #endif
}

#endif // AMT_UI_FLOAT_HPP
