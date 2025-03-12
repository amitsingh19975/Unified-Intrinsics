#ifndef AMT_UI_ARCH_BASIC_HPP
#define AMT_UI_ARCH_BASIC_HPP

#include "../base.hpp"
#include "../base_vec.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cfenv>
#include <type_traits>

namespace ui::internal {

    template <typename T, typename U>
    struct WideningResult {
        using type = T;
    };

    template <>
    struct WideningResult<std::byte, std::byte> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::byte> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::byte, std::uint16_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint8_t, std::uint8_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint8_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint8_t, std::uint16_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::int8_t, std::int8_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int8_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::int8_t, std::int16_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint16_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint16_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint32_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int16_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int16_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int32_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint32_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::uint64_t, std::uint32_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint64_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int32_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int64_t, std::int32_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int64_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int64_t, std::int64_t> {
        using type = float;
    };

    template <>
    struct WideningResult<float, float> {
        using type = double;
    };

    template <typename T, typename U = T>
    using widening_result_t = typename WideningResult<T, U>::type;

    template <typename T>
    struct NarrowingResult;

    template <>
    struct NarrowingResult<std::uint16_t> {
        using type = std::uint8_t;
    };

    template <>
    struct NarrowingResult<std::int16_t> {
        using type = std::int8_t;
    };

    template <>
    struct NarrowingResult<std::uint32_t> {
        using type = std::uint16_t;
    };

    template <>
    struct NarrowingResult<std::int32_t> {
        using type = std::int16_t;
    };

    template <>
    struct NarrowingResult<std::uint64_t> {
        using type = std::uint32_t;
    };

    template <>
    struct NarrowingResult<std::int64_t> {
        using type = std::int32_t;
    };

    template <typename T>
    using narrowing_result_t = typename NarrowingResult<T>::type;


    template <typename From>
    UI_ALWAYS_INLINE constexpr auto halving_round_helper(From lhs, From rhs, op::add_t) noexcept -> std::int64_t {
        using utype = std::make_unsigned_t<From>;
        static constexpr auto bits = sizeof(From) * 8 - 1;
        auto l = lhs >> 1;
        auto r = rhs >> 1;
        From res = lhs | rhs;
        res = static_cast<utype>(res << bits) >> bits;
        return static_cast<std::int64_t>(res + (l + r));
    }

    template <typename From>
    UI_ALWAYS_INLINE constexpr auto halving_round_helper(From lhs, From rhs, op::sub_t) noexcept {
        using wtype = widening_result_t<From>;
        auto l = static_cast<wtype>(lhs);
        auto r = static_cast<wtype>(rhs);
        auto sum = (l - r) >> 1;
        return sum;
    }

    template <typename T>
    UI_ALWAYS_INLINE static constexpr auto max(
        T lhs,
        T rhs
    ) noexcept -> T {
        if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
            return T(std::max(float(lhs), float(rhs)));
        } else {
            return std::max(lhs, rhs);
        }
    }

    template <typename T>
    UI_ALWAYS_INLINE static constexpr auto min(
        T lhs,
        T rhs
    ) noexcept -> T {
        if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
            return T(std::min(float(lhs), float(rhs)));
        } else {
            return std::min(lhs, rhs);
        }
    }

    template <std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto maxnm(T a, T b) noexcept -> T {
        using std::isnan;
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        return max(a, b);
    }

    template <std::floating_point T>
    UI_ALWAYS_INLINE static constexpr auto minnm(T a, T b) noexcept -> T {
        using std::isnan;
        if (isnan(a)) return b;
        if (isnan(b)) return a;
        return min(a, b);
    }

    static inline constexpr auto convert_rounding_style(std::float_round_style mode) noexcept -> int {
        switch (mode) {
        case std::round_toward_zero: return FE_TOWARDZERO;
        case std::round_to_nearest: return FE_TONEAREST;
        case std::round_toward_infinity: return FE_UPWARD;
        case std::round_toward_neg_infinity: return FE_DOWNWARD;
        default: return FE_TONEAREST;
        }
    }

    template <typename To, bool Saturating, typename T>
    UI_ALWAYS_INLINE static constexpr auto saturating_cast_helper(
        T v
    ) noexcept -> To {
        if constexpr (std::is_signed_v<To> == std::is_signed_v<T>) {
            if constexpr (Saturating) {
                if (sizeof(To) < sizeof(T)) {
                    return static_cast<To>(
                        std::clamp<T>(
                            v,
                            std::numeric_limits<To>::min(),
                            std::numeric_limits<To>::max()
                        )
                    );
                }
            }
        } else if constexpr (Saturating) {
            if (std::is_signed_v<To>) {
                // L: Sint, R: UInt
                // L = min(R, maxOf(L))
                if constexpr (sizeof(To) == sizeof(T)) {
                    static constexpr auto max = std::numeric_limits<To>::max();
                    return static_cast<To>(std::min<T>(v, max));
                } else if constexpr (sizeof(To) < sizeof(T)) {
                    // Narrowing casting
                    static constexpr auto max = static_cast<T>(std::numeric_limits<To>::max());
                    return static_cast<To>(std::min(v, max));
                }
            } else {
                // L: Uint, R: SInt
                // L = min(max(R, 0), maxOf(L)) => clamp(R, 0, min(maxOf(L), maxOf(R)))
                if constexpr (sizeof(To) == sizeof(T)) {
                    return static_cast<To>(std::max<T>(v, 0));
                } else if constexpr (sizeof(To) < sizeof(T)) {
                    static constexpr auto max = static_cast<T>(std::numeric_limits<To>::max());
                    return static_cast<To>(std::clamp<T>(v, 0, max));
                } else {
                    return static_cast<To>(std::max<T>(v, 0));
                }
            }
        }
        return static_cast<To>(v);
    }

    template <typename T>
    concept is_fp16 = std::same_as<T, float16> || std::same_as<T, bfloat16>;


    template <std::size_t N, typename Fn>
    struct Case {
        static constexpr auto pattern = N;
        template <typename... Args>
        constexpr decltype(auto) operator()(Args&&... args) const noexcept 
            requires std::invocable<Fn, Args...>
        {
            return std::invoke(fn, std::forward<Args>(args)...);
        }

        template <typename Tn>
        constexpr Case<N, Tn> operator=(Tn&& f) const noexcept {
            return { .fn = std::forward<Tn>(f) };
        }

        Fn fn;
    };

    template <std::size_t N>
    static constexpr auto case_maker = Case<N, void*>(nullptr);

    template <typename T>
    struct is_case: std::false_type {};

    template <std::size_t N, typename Fn>
    struct is_case<Case<N, Fn>>: std::true_type {};

    template <typename T>
    concept CaseMaker = is_case<std::decay_t<T>>::value;

    template <typename... Cs>
    struct Matcher {
        using base_type = std::tuple<Cs...>;
        base_type cases;

        template <std::size_t P>
        static constexpr auto can_match_case = ((P == Cs::pattern) || ...);

        constexpr Matcher() noexcept = default;
        constexpr Matcher(Cs&&... cs) noexcept requires (sizeof...(Cs) > 0 && (CaseMaker<Cs> && ...))
            : cases(std::forward<Cs>(cs)...)
        {}
        constexpr Matcher(Matcher const& fn) noexcept = default;
        constexpr Matcher(Matcher && fn) noexcept = default;
        constexpr Matcher& operator=(Matcher const& fn) noexcept = default;
        constexpr Matcher& operator=(Matcher && fn) noexcept = default;
        constexpr ~Matcher() noexcept = default;

        template <std::size_t P, typename... Args>
            requires can_match_case<P>
        constexpr decltype(auto) match(Args&&... args) const noexcept {
            return invoke_helper<0, P>(std::forward<Args>(args)...); 
        }


    private:
        template <std::size_t I = 0, std::size_t P, typename... Args>
        constexpr decltype(auto) invoke_helper(Args&&... args) const noexcept {
            if constexpr (sizeof... (Cs) <= I) return;
            else if constexpr (std::tuple_element_t<I, base_type>::pattern == P) {
                return std::invoke(std::get<I>(cases), std::forward<Args>(args)...);
            } else {
                return invoke_helper<I + 1, P>(std::forward<Args>(args)...);
            }
        }
    };

    template<typename... Ts>
    Matcher(Ts...) -> Matcher<Ts...>;

    template<typename... Ts>
    struct Overloaded: Ts... { using Ts::operator()...; };

    template<typename... Ts>
    Overloaded(Ts...) -> Overloaded<Ts...>;

    template <typename T>
    concept not_void = !std::is_void_v<T>;

    template <std::size_t P, typename M, typename... Args>
    concept is_case_invocable = (M::template can_match_case<P>) && requires (M const& m) {
        { m.template match<P>(std::declval<Args>()...) } -> not_void;
    }; 
} // namespace ui::internal

#endif // AMT_UI_ARCH_BASIC_HPP
