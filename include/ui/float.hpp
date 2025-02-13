#ifndef AMT_UI_FLOAT_HPP
#define AMT_UI_FLOAT_HPP

#include "base.hpp"
#include "vec_headers.hpp"
#include "forward.hpp"
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

        #ifdef UI_HAS_BFLOAT_16
            #if !defined(UI_ARM_HAS_NEON) || !defined(UI_CPU_ARM64)
                #ifdef UI_HAS_STD_FLOAT_HEADER
                    using bfloat16_t = std::float16_t;
                #else
                    using bfloat16_t = _Float16;
                #endif
            #else
                using bfloat16_t = ::bfloat16_t;
            #endif
        #else
            #define UI_HAS_CUSTOM_BFLOAT16_IMPL
            using bfloat16_t = std::uint16_t;
        #endif
    }

    namespace fp {
        template <typename T>
        struct FloatingPointRep;

        template <>
        struct FloatingPointRep<float> {
            static constexpr unsigned sign_bits = 1;
            static constexpr unsigned exponent_bits = 8;
            static constexpr unsigned mantissa_bits = 23;
            static constexpr unsigned bias = (1 << exponent_bits) / 2 - 1;
            bool sign;
            std::int32_t exponent;
            std::uint32_t mantissa;
        };

        template <>
        struct FloatingPointRep<double> {
            static constexpr unsigned sign_bits = 1;
            static constexpr unsigned exponent_bits = 11;
            static constexpr unsigned mantissa_bits = 52;
            static constexpr unsigned bias = (1 << exponent_bits) / 2 - 1;
            bool sign;
            std::int32_t exponent;
            std::uint64_t mantissa;
        };

        template <>
        struct FloatingPointRep<float16> {
            static constexpr unsigned sign_bits = 1;
            static constexpr unsigned exponent_bits = 5;
            static constexpr unsigned mantissa_bits = 10;
            static constexpr unsigned bias = (1 << exponent_bits) / 2 - 1;
            bool sign;
            std::int16_t exponent;
            std::uint16_t mantissa;
        };

        template <>
        struct FloatingPointRep<bfloat16> {
            static constexpr unsigned sign_bits = 1;
            static constexpr unsigned exponent_bits = 8;
            static constexpr unsigned mantissa_bits = 7;
            static constexpr unsigned bias = (1 << exponent_bits) / 2 - 1;
            bool sign;
            std::int16_t exponent;
            std::uint8_t mantissa;
        };

        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float n) noexcept -> FloatingPointRep<float> {
            auto bits = std::bit_cast<std::uint32_t>(n);
            return {
                .sign = static_cast<bool>(bits >> 31),
                .exponent = static_cast<std::int32_t>(((bits >> 23) & 0xFF) - FloatingPointRep<float>::bias),
                .mantissa = bits & 0x7FFFFF,
            };
        }

        UI_ALWAYS_INLINE static constexpr auto decompose_fp(double n) noexcept -> FloatingPointRep<double> {
            auto bits = std::bit_cast<std::uint64_t>(n);
            return {
                .sign = static_cast<bool>(bits >> 63),
                .exponent = static_cast<std::int32_t>(((bits >> 52) & 0x7FF) - FloatingPointRep<double>::bias),
                .mantissa = bits & 0xFFFFFFFFFFFFF,
            };
        }
        
        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float> fp) noexcept -> float {
            int biased_exp = fp.exponent + 127;
            
            std::uint32_t exp_field = static_cast<std::uint32_t>(biased_exp) & 0xFF;
            
            auto bits = (static_cast<std::uint32_t>(fp.sign) << 31)
                      | (exp_field << 23)
                      | fp.mantissa;
            
            return std::bit_cast<float>(bits);
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<double> fp) noexcept -> double {
            int biased_exp = fp.exponent + 1023;
            
            std::uint64_t exp_field = static_cast<std::uint32_t>(biased_exp) & 0x7FF;
            
            auto bits = (static_cast<std::uint64_t>(fp.sign) << 63)
                      | (exp_field << 52)
                      | fp.mantissa;
            
            return std::bit_cast<double>(bits);
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float16> fp) noexcept -> float16;
    }


    namespace fp {
        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float16 fp) noexcept -> FloatingPointRep<float16>;
        UI_ALWAYS_INLINE static constexpr auto compose_fp_helper(FloatingPointRep<float16> fp) noexcept -> internal::float16_t {
            int biased_exp = fp.exponent + 15;
            auto exp_field = static_cast<std::uint16_t>(biased_exp) & 0x1F;
            auto bits = static_cast<std::uint16_t>((static_cast<std::uint16_t>(fp.sign) << 15)
                      | (exp_field << 10)
                      | (fp.mantissa & 0x3FF));
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
                return static_cast<base_type>(__FLT16_MIN__);
            #endif
        }

        static constexpr auto max_rep() noexcept -> base_type {
            #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
                return fp::compose_fp_helper(fp_rep{ .sign = false, .exponent = 15, .mantissa = 0xFFFF });
            #else
                return static_cast<base_type>(__FLT16_MAX__);
            #endif
        }

        base_type data;

        constexpr float16() noexcept = default;
        constexpr float16(float16 const&) noexcept = default;
        constexpr float16(float16 &&) noexcept = default;
        constexpr float16& operator=(float16 const&) noexcept = default;
        constexpr float16& operator=(float16 &&) noexcept = default;
        constexpr ~float16() noexcept = default;

        #ifdef UI_HAS_CUSTOM_FLOAT16_IMPL
        constexpr float16(float f) noexcept {
        #define I(V) std::bit_cast<std::uint32_t>(V)
        #define F(V) std::bit_cast<float>(V)
            // 1. Get the underlying bits of float
            auto bits = I(f);
            // 2. Get the sign bit by masking the exponent and mantissa
            auto sign = bits & 0x8000'0000;
            // 3. Convert the float to postive rational number and clamp the max range to "65536."
            // Max value of float16 is 65504, but the range is clamped to 65536 which is infinity in float16
            auto abs = std::min<std::uint32_t>(bits ^ sign, 0x4780'0000);
            // 4. Multiplying the value with 2^13 and if the range is smaller, then
            // we are working with denormal value. So we clamp it with 0.5 to avoid denormal case
            // and allow us to work with subnormal values. 0.5 will shift the exponent to 2^-14 which
            // is the expected subnormal.
            // On top of that, we mask the exponent field. So magic is simply exponent.
            // float 32 rep: |(sign) 0 | (exponent) magic | (mantiss) 0 |
            auto magic = I(std::max<float>(F(abs) * float(1 << 13), 0.5f)) & (0xff << 23);

            // 5. This will cause the exponent to shift mantissa toward right and will round-to-even.
            auto rounded = I(F(abs) + F(magic));
            // 6. Adjusting exponent. This will convert the exponent with bias 127 to 15 for float16.
            auto exp = ((magic >> 13) - ((127 /*float32 bias*/ - 15 /*float16 bias*/ + 13 /*undo the multiplication of 2^13*/ + 1 /*remove the implicit leading 1*/) << 10));
            // 7. Combine the exponent and rounded mantissa. + is used to allow rounded bit to rollover into exponent bit
            auto f16 = rounded + exp;
            // 8. if the number could be inf or nan based on mantissa. To preserve the state, we
            // bitwise-or with the 1 or 0 if it's nan or not. Inf has all the mantissa bits set to 0.
            data = static_cast<base_type>((sign >> 16) | f16 | (std::isnan(f) ? 1 : 0));
        #undef I
        #undef F
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
        #define I(V) std::bit_cast<std::uint32_t>(V)
        #define F(V) std::bit_cast<float>(V)
            auto wide = static_cast<std::uint32_t>(data);
            // 1. Get the sign
            auto sign = wide & 0x8000;
            // 2. Remove the sign
            auto abs = wide ^ sign;
            // 3. Check if the number is NaN or Infinity by checking the exponent. If the exponenet
            // has all the bits set then it's one of them.
            auto inf_or_nan = (abs >= (31 << 10)) ? (0xff << 23) : 0u;
            // 4. Check if the number is normal or subnormal by check if any of the bits inside the
            // exponent is set or not. If there is not bit set inside exponent, then max value will
            // be the same as the max value when all the bits are set inside the mantissa.
            auto is_norm = (abs > 0x3ff);

            // 5. Subnormal f16's are 2^-14*0.[m0:9] == 2^-24*[m0:9].0
            auto sub = I(static_cast<float>(abs) * (1.f / (1 << 24)));
            // 6. Calculate the normal by fixing the exponent. Convert the float16 bias to float32
            // by subtracting 15 bias (float16) and adding 127 bias (float32).
            auto norm = ((abs << 13) + ((127 - 15) << 23));
            // 7. Choose the correct normal
            auto finite = is_norm ? norm : sub;
            return F((sign << 16) | finite | inf_or_nan); 
        #undef I
        #undef F
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
            if (val.is_nan()) return val;
            std::uint16_t temp = std::bit_cast<std::uint16_t>(val.data) ^ 0x8000u;
            return std::bit_cast<float16>(temp);
        }

        constexpr auto is_nan() const noexcept -> bool {
            auto val = std::bit_cast<std::uint16_t>(data);
            return val & 0x7c01;
        }

        constexpr auto abs() const noexcept -> float16 {
            auto val = std::bit_cast<std::uint16_t>(data);
            auto t = static_cast<std::uint16_t>(val & (~std::uint16_t(0) >> 1));
            return std::bit_cast<float16>(t);
        }

        constexpr auto is_zero() const noexcept -> bool {
            auto val = std::bit_cast<std::uint16_t>(data);
            return (val & 0) | (val & 0x8000u);
        }

        constexpr auto is_neg() const noexcept -> bool {
            return std::bit_cast<std::uint16_t>(data) & 0x8000u;
        }

        constexpr auto is_inf(bool sign) const noexcept -> bool {
            auto val = std::bit_cast<std::uint16_t>(data);
            auto const mask = std::bit_cast<std::uint16_t>(fp::compose_fp_helper({ .sign = sign, .exponent = 0x1f, .mantissa = 0 }));
            return val & mask; 
        }

        constexpr auto is_inf() const noexcept -> bool {
            auto temp = std::bit_cast<std::uint16_t>(data);
            return (temp & 0x7c00) || (temp & 0xfc00);
        }

    private:
        constexpr auto set_data(bool sign, std::int16_t exp, std::uint16_t man) noexcept -> void {
            data = fp::compose_fp(fp::FloatingPointRep<float16>{ .sign = sign, .exponent = exp, .mantissa = man }).data;
        }
    };

    namespace fp {
        UI_ALWAYS_INLINE static constexpr auto decompose_fp(float16 fp) noexcept -> FloatingPointRep<float16> {
            auto bits = std::bit_cast<std::uint16_t>(fp);
            return {
                .sign = static_cast<bool>(bits >> 15),
                .exponent = static_cast<std::int16_t>((((bits >> 10) & 0x1F) - FloatingPointRep<float16>::bias)),
                .mantissa = static_cast<std::uint16_t>(bits & 0x3FF),
            };
        }

        UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float16> fp) noexcept -> float16 {
            return std::bit_cast<float16>(compose_fp_helper(fp));
        }
    }

    struct alignas(sizeof(internal::bfloat16_t)) bfloat16 {
        using base_type = internal::bfloat16_t;
        using fp_rep = fp::FloatingPointRep<bfloat16>;

        struct internal_t {};

        base_type data;

        constexpr bfloat16() noexcept = default;
        constexpr bfloat16(bfloat16 const&) noexcept = default;
        constexpr bfloat16(bfloat16 &&) noexcept = default;
        constexpr bfloat16& operator=(bfloat16 const&) noexcept = default;
        constexpr bfloat16& operator=(bfloat16 &&) noexcept = default;
        constexpr ~bfloat16() noexcept = default;

        constexpr bfloat16(float f) noexcept {
            auto temp = std::bit_cast<std::uint32_t>(f);
            // Rounding-to-even
            auto rounding_bias = 0x7FFF + ((temp >> 16) & 1);
            auto mantissa = ((temp + rounding_bias) & 0x7FFFFF) >> 16;
            temp = (temp >> 16) | mantissa;
            data = std::bit_cast<base_type>(static_cast<std::uint16_t>(temp));
        }

        constexpr bfloat16(double f) noexcept
            : bfloat16(static_cast<float>(f))
        {}

        constexpr bfloat16(std::convertible_to<float> auto v) noexcept
            : bfloat16(static_cast<float>(v))
        {}

        explicit constexpr operator float() const noexcept {
            auto temp = static_cast<std::uint32_t>(std::bit_cast<std::uint16_t>(data));
            return std::bit_cast<float>(temp << 16);
        }

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

        friend constexpr auto operator-(bfloat16 val) noexcept -> bfloat16 {
            return (-float(val));
        }

        constexpr auto is_nan() const noexcept -> bool {
            return std::isnan(float(*this));
        }

        constexpr auto abs() const noexcept -> bfloat16 {
            return { std::abs(float(*this)) };
        }

        constexpr auto is_zero() const noexcept -> bool {
            auto val = std::bit_cast<std::uint16_t>(data);
            return (val & 0) | (val & 0x8000u);
        }

        constexpr auto is_neg() const noexcept -> bool {
            return std::bit_cast<std::uint16_t>(data) & 0x8000u;
        }

        constexpr auto is_inf(bool sign) const noexcept -> bool {
            auto temp = float(*this);
            return std::isinf(temp) && (is_neg() == sign);
        }

        constexpr auto is_inf() const noexcept -> bool {
            return std::isinf(float(*this));
        }
    };

    namespace fp {
        UI_ALWAYS_INLINE static constexpr auto decompose_fp(bfloat16 fp) noexcept -> FloatingPointRep<bfloat16> {
            auto bits = std::bit_cast<std::uint16_t>(fp);
            return {
                .sign = static_cast<bool>(bits >> 15),
                .exponent = static_cast<std::int16_t>(((bits >> 7) & 0xFF) - FloatingPointRep<bfloat16>::bias),
                .mantissa = static_cast<std::uint8_t>(bits & 0x7f),
            };
        }
        /*UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<bfloat16> fp) noexcept -> internal::float16_t {*/
        /*    int biased_exp = fp.exponent + 15;*/
        /*    auto exp_field = static_cast<std::uint16_t>(biased_exp) & 0x1F;*/
        /*    auto bits = static_cast<std::uint16_t>((static_cast<std::uint16_t>(fp.sign) << 15)*/
        /*              | (exp_field << 10)*/
        /*              | (fp.mantissa & 0x3FF));*/
        /*    return std::bit_cast<internal::float16_t>(bits);*/
        /*}*/
    }
    
    template <std::size_t N>
    static inline auto cast_float32_to_float16(
        Vec<N, float> const& v
    ) noexcept -> Vec<N, float16>;

    template <std::size_t N>
    static inline constexpr auto cast_float16_to_float32(
        Vec<N, float16> const& v
    ) noexcept -> Vec<N, float>;

    template <std::size_t N>
    static inline constexpr auto cast_float32_to_bfloat16(
        Vec<N, float> const& v
    ) noexcept -> Vec<N, bfloat16>;

    template <std::size_t N>
    static inline constexpr auto cast_bfloat16_to_float32(
        Vec<N, bfloat16> const& v
    ) noexcept -> Vec<N, float>;

    
    constexpr bool isnan(float16 val) noexcept {
        return val.is_nan();
    }

    constexpr bool isnan(bfloat16 val) noexcept {
        return val.is_nan();
    }

    constexpr bool isinf(float16 val) noexcept {
        return val.is_inf();
    }

    constexpr bool isinf(bfloat16 val) noexcept {
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

    constexpr int fpclassify(bfloat16 val) noexcept {
        return std::fpclassify(float(val));
    }

    constexpr bool signbit(float16 val) noexcept {
        return val.is_neg();
    }

    constexpr bool signbit(bfloat16 val) noexcept {
        return val.is_neg();
    }

    constexpr float16 abs(float16 val) noexcept {
        return val.abs();
    }

    constexpr bfloat16 abs(bfloat16 val) noexcept {
        return val.abs();
    }

} // namespace ui

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
    template <>
    struct is_floating_point<ui::float16>: std::true_type{}; 

    template <>
    struct is_signed<ui::float16>: std::true_type{}; 

    template <>
    struct is_floating_point<ui::bfloat16>: std::true_type{}; 

    template <>
    struct is_signed<ui::bfloat16>: std::true_type{}; 

    template <>
    struct is_arithmetic<ui::float16>: std::true_type{};

    template <>
    struct is_arithmetic<ui::bfloat16>: std::true_type{};

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
        static constexpr type min() noexcept { return __FLT16_MIN__; }
        static constexpr type max() noexcept { return __FLT16_MAX__; }
        #else
        static constexpr type min() noexcept { return std::bit_cast<type>(type::min_rep()); }
        static constexpr type max() noexcept { return std::bit_cast<type>(type::max_rep()); }
        #endif
        static constexpr type lowest() noexcept { return -max(); }

        static constexpr const bool is_integer = false;
        static constexpr const bool is_exact   = false;
        static constexpr const int radix       = FLT_RADIX;
        #ifndef UI_HAS_CUSTOM_FLOAT16_IMPL
        static constexpr type epsilon() noexcept { return __FLT16_EPSILON__; }
        static constexpr const int min_exponent   = __FLT16_MIN_EXP__;
        static constexpr const int min_exponent10 = __FLT16_MIN_10_EXP__;
        static constexpr const int max_exponent   = __FLT16_MAX_EXP__;
        static constexpr const int max_exponent10 = __FLT16_MAX_10_EXP__;
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
        static constexpr type infinity() noexcept { return bit_cast<type>(uint16_t(0x7c00)); }
        static constexpr type quiet_NaN() noexcept { return bit_cast<type>(uint16_t(0x7c01)); }

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

    template <>
    class numeric_limits<ui::bfloat16> {
        using type = ui::bfloat16;
    public:
        static constexpr const bool is_specialized = true;
        static constexpr const bool is_signed   = true;
        static constexpr const int digits       = 8;
        static constexpr const int digits10     = 2;
        static constexpr const int max_digits10 = 2 + (digits * 30103l) / 100000l;
        static constexpr type min() noexcept { return type(numeric_limits<float>::min()); }
        static constexpr type max() noexcept { return type(numeric_limits<float>::max()); }
        static constexpr type lowest() noexcept { return -max(); }

        static constexpr const bool is_integer = numeric_limits<float>::is_integer;
        static constexpr const bool is_exact   = numeric_limits<float>::is_exact;
        static constexpr const int radix       = numeric_limits<float>::radix;
        static constexpr const int min_exponent   = numeric_limits<float>::min_exponent;
        static constexpr const int min_exponent10 = numeric_limits<float>::min_exponent10;
        static constexpr const int max_exponent   = numeric_limits<float>::max_exponent;
        static constexpr const int max_exponent10 = numeric_limits<float>::max_exponent10;
        static constexpr type epsilon() noexcept { return 0.0078125; }
        static constexpr type round_error() noexcept { return numeric_limits<float>::round_error(); }


        static constexpr const bool has_infinity        = numeric_limits<float>::has_infinity;
        static constexpr const bool has_quiet_NaN       = numeric_limits<float>::has_quiet_NaN;
        static constexpr const bool has_signaling_NaN   = false;
        static constexpr type infinity() noexcept { return type(numeric_limits<float>::infinity()); }
        static constexpr type quiet_NaN() noexcept { return type(numeric_limits<float>::quiet_NaN()); }

        static constexpr const bool is_iec559  = numeric_limits<float>::is_iec559;
        static constexpr const bool is_bounded = numeric_limits<float>::is_bounded;
        static constexpr const bool is_modulo  = numeric_limits<float>::is_modulo;

        static constexpr const bool traps = numeric_limits<float>::traps;
#if (defined(__arm__) || defined(__aarch64__))
        static constexpr const bool tinyness_before = numeric_limits<float>::tinyness_before;
#else
        static constexpr const bool tinyness_before = numeric_limits<float>::tinyness_before;
#endif
        static constexpr const float_round_style round_style = numeric_limits<float>::round_style;
    };

    template <>
    struct formatter<ui::float16>: formatter<float> {
        auto format(ui::float16 val, auto& ctx) const {
            return formatter<float>::format(static_cast<float>(val), ctx);
        }
    };

    template <>
    struct formatter<ui::bfloat16>: formatter<float> {
        auto format(ui::bfloat16 val, auto& ctx) const {
            return formatter<float>::format(static_cast<float>(val), ctx);
        }
    };

    template <typename T>
    struct formatter<ui::fp::FloatingPointRep<T>> {
        using fp_rep = ui::fp::FloatingPointRep<T>;

        enum class Radix {
            dec,    // base-10
            bin,    // base-2
            hex,    // base-16
        };
        Radix radix{Radix::dec};

        constexpr auto set_base(char ch) noexcept {
            switch(ch) {
            case 'b': radix = Radix::bin; return true;
            case 'x': radix = Radix::hex; return true;
            default: radix = Radix::dec; return false;
            }
        }

        constexpr auto parse(format_parse_context& ctx) {
            auto it = ctx.begin();
            while (it != ctx.end() && *it != '}') {
                if (*it == '0') {
                    ++it;
                    set_base(*it);
                } 
                set_base(*it);
                ++it;
            }
            return it;
        }

        auto format(fp_rep val, auto& ctx) const {
            auto [s, e, m] = val;
            switch (radix) {
                case Radix::dec: {
                    return format_to(ctx.out(), "| (sign){} | (exp){} | (mantissa){} |", s ? '-' : '+', e + static_cast<int>(fp_rep::bias), m);
                }
                case Radix::hex: {
                    return format_to(ctx.out(), "| (sign)0x{:x} | (exp)0x{:0{}x} | (mantissa)0x{:0{}x} |", s, e + static_cast<int>(fp_rep::bias), fp_rep::exponent_bits / 4, m, fp_rep::mantissa_bits / 4);
                }
                case Radix::bin: {
                    return format_to(ctx.out(), "| (sign)0b{:b} | (exp)0b{:0{}b} | (mantissa)0b{:0{}b} |", s, e + static_cast<int>(fp_rep::bias), fp_rep::exponent_bits, m, fp_rep::mantissa_bits);
                }
            }            
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
        return std::bit_cast<ui::float16>(get_first(vadd_f16(load_f16(lhs), load_f16(rhs))));
        #else
        return std::bit_cast<ui::float16>(static_cast<ui::float16::base_type>(lhs.data + rhs.data));
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
        return std::bit_cast<ui::float16>(get_first(vsub_f16(load_f16(lhs), load_f16(rhs))));
        #else
        return std::bit_cast<ui::float16>(static_cast<ui::float16::base_type>(lhs.data - rhs.data));
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
        return std::bit_cast<ui::float16>(get_first(vmul_f16(load_f16(lhs), load_f16(rhs))));
        #else
        return std::bit_cast<ui::float16>(static_cast<ui::float16::base_type>(lhs.data * rhs.data));
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
        return std::bit_cast<ui::float16>(get_first(vdiv_f16(load_f16(lhs), load_f16(rhs))));
        #else
        return std::bit_cast<ui::float16>(static_cast<ui::float16::base_type>(lhs.data / rhs.data));
        #endif
    #endif
}

static constexpr auto operator==(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return float(lhs) == float(rhs);
}

static constexpr auto operator!=(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return !(lhs == rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator==(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return lhs == ui::bfloat16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator!=(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return !(lhs == rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator==(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return ui::bfloat16(lhs) == rhs;
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator!=(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return !(lhs == rhs);
}

static constexpr auto operator<(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return float(lhs) < float(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator<(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return lhs < ui::bfloat16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator<(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return ui::bfloat16(lhs) < rhs;
}

static constexpr auto operator>(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return float(lhs) > float(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator>(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return lhs > ui::bfloat16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator>(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return ui::bfloat16(lhs) > rhs;
}

static constexpr auto operator<=(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return float(lhs) <= float(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator<=(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return lhs <= ui::bfloat16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator<=(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return ui::bfloat16(lhs) <= rhs;
}

static constexpr auto operator>=(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> bool {
    return float(lhs) >= float(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator>=(ui::bfloat16 lhs, T rhs) noexcept -> bool {
    return lhs >= ui::bfloat16(rhs);
}

template <typename T>
    requires std::constructible_from<ui::bfloat16, T>
static constexpr auto operator>=(T lhs, ui::bfloat16 rhs) noexcept -> bool {
    return ui::bfloat16(lhs) >= rhs;
}

static constexpr auto operator+(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> ui::bfloat16 {
    return ui::bfloat16(float(lhs) + float(rhs));
}

static constexpr auto operator-(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> ui::bfloat16 {
    return ui::bfloat16(float(lhs) - float(rhs));
}

static constexpr auto operator*(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> ui::bfloat16 {
    return ui::bfloat16(float(lhs) * float(rhs));
}

static constexpr auto operator/(ui::bfloat16 lhs, ui::bfloat16 rhs) noexcept -> ui::bfloat16 {
    return ui::bfloat16(float(lhs) / float(rhs));
}

#endif // AMT_UI_bfloat_HPP
