#ifndef AMT_UI_VEC_OP_HPP
#define AMT_UI_VEC_OP_HPP

#include "base_vec.hpp"
#include "features.hpp"
#include "ui/arch/arm/arm.hpp"
#include <initializer_list>
#include <type_traits>

namespace ui {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto load(std::span<T> in) noexcept -> VecReg<N, T> {
        return load(in.data(), in.size());
    }

    template <std::size_t N, typename T, typename... Ts>
        requires ((... && std::same_as<T, Ts>) && std::is_arithmetic_v<T> && (sizeof...(Ts) + 1 == N))
    UI_ALWAYS_INLINE constexpr auto load(T v0, Ts... args) noexcept -> VecReg<N, T> {
        std::array<T, N> res = { v0, args... };
        return load<N>(res);
    }

    template <typename T, typename... Ts>
        requires ((... && std::same_as<T, Ts>) && std::is_arithmetic_v<T>)
    UI_ALWAYS_INLINE constexpr auto load(T v0, Ts... args) noexcept {
        return load<sizeof...(Ts) + 1, T>(v0, args...);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto load(std::initializer_list<T> in) noexcept -> VecReg<N, T> {
        return load(in.data(), in.size());
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto load(T val) noexcept -> VecReg<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::load<N>(val);
        #else
            #error "not implemented"
        #endif
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto store(
        T* UI_RESTRICT out,
        VecReg<N, T> const& in
    ) noexcept -> void {
        std::memcpy(out, in.data(), N * sizeof(T));
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto store(
        std::span<T> out,
        VecReg<N, T> const& in
    ) noexcept -> void {
        std::memcpy(out, in.data(), std::min(out.size(), N) * sizeof(T));
    }
} // namespace ui

#endif // AMT_UI_VEC_OP_HPP
