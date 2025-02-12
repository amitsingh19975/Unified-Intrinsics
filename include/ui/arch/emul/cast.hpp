#ifndef AMT_UI_ARCH_EMUL_CAST_HPP
#define AMT_UI_ARCH_EMUL_CAST_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../basic.hpp"
#include "../../vec_headers.hpp"
#include "../../maths.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include <concepts>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ui::emul {
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return map([](auto v) { return static_cast<To>(v); }, v);
    }

    template <typename To, std::size_t N, std::integral From>
    UI_ALWAYS_INLINE auto sat_cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return map([](auto v) {
            if constexpr (std::is_signed_v<To>) {
                static constexpr auto min = static_cast<std::int64_t>(std::numeric_limits<To>::min());
                static constexpr auto max = static_cast<std::int64_t>(std::numeric_limits<To>::max());
                if constexpr (sizeof(To) < 8) {
                    auto temp = static_cast<std::int64_t>(v);
                    return static_cast<To>(std::clamp(temp, min, max));
                } 
            } else {
                static constexpr auto min = static_cast<std::uint64_t>(std::numeric_limits<To>::min());
                static constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<To>::max());
                if constexpr (sizeof(To) < 8) {
                    auto temp = static_cast<std::uint64_t>(v);
                    return static_cast<To>(std::clamp(temp, min, max));
                } 
            }
            return static_cast<To>(v);
        }, v);
    }

    // retinterpret cast
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE constexpr auto rcast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return std::bit_cast<Vec<N, To>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto to_vec(auto const& v) noexcept {
        return v;
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(auto const& v) noexcept {
        return v;
    }

    template <typename T>
    UI_ALWAYS_INLINE constexpr auto from_vec(auto const& v) noexcept {
        return rcast<T>(v);
    }
} // namespace ui::emul

#endif // AMT_UI_ARCH_EMUL_CAST_HPP
