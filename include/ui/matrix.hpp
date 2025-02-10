#ifndef AMT_UI_MATRIX_HPP
#define AMT_UI_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <array>
#include <type_traits>
#include "base.hpp"
#include "base_vec.hpp"

namespace ui {
    template <std::size_t R, std::size_t C, typename T>
    struct alignas(sizeof(Vec<C, T>)) VecMat {
        using vec_type = Vec<C, T>;
        using element_t = T;
        using size_type = std::size_t;

        static constexpr size_type rows = R;
        static constexpr size_type cols = C;

        vec_type val[R];

        constexpr VecMat() noexcept = default;
        constexpr VecMat(VecMat const&) noexcept = default;
        constexpr VecMat(VecMat &&) noexcept = default;
        constexpr VecMat& operator=(VecMat const&) noexcept = default;
        constexpr VecMat& operator=(VecMat &&) noexcept = default;
        constexpr ~VecMat() noexcept = default;

        static auto load(element_t const* const UI_RESTRICT in, size_type size) noexcept -> VecMat {
            auto res = VecMat{};
            std::memcpy(res.data(), in, sizeof(element_t) * std::min(size, rows * cols));
            return res;
        }

        static constexpr auto load(std::span<element_t> sp) noexcept -> VecMat {
            return load(sp.data(), sp.size());
        }

        static constexpr auto load(element_t val) noexcept -> VecMat {
            VecMat res;
            for (auto i = 0ul; i < R; ++i) {
                res.val[i] = vec_type::load(val);
            }
            return res;
        }

        template <internal::is_vec... Ts>
            requires (((std::same_as<vec_type, std::decay_t<Ts>>) && ...) && sizeof...(Ts) <= rows)
        static constexpr auto load(Ts&&... args) noexcept -> VecMat {
            std::array<vec_type, rows> temp = { args... }; 
            auto res = VecMat{};
            for (auto i = 0u; i < rows; ++i) res.val[i] = temp[i];
            return res;
        }

        constexpr auto operator()(size_type r, size_type c) const noexcept -> element_t {
            return val[r][c];
        }
        
        constexpr auto operator()(size_type r, size_type c) noexcept -> element_t& {
            return val[r][c];
        }
        
        auto data() noexcept -> element_t* {
            return reinterpret_cast<element_t*>(this);
        }

        auto data() const noexcept -> element_t const* {
            return reinterpret_cast<element_t const*>(this);
        }

        auto lo() const noexcept -> VecMat<R / 2, C, T> {
            using type = VecMat<R / 2, C, T>;
            auto const bp_ptr = reinterpret_cast<vec_type const*>(this);
            return *reinterpret_cast<type const*>(bp_ptr);
        }

        auto hi() const noexcept -> VecMat<R / 2, C, T> {
            using type = VecMat<R / 2, C, T>;
            auto const bp_ptr = reinterpret_cast<vec_type const*>(this) + R / 2;
            return *reinterpret_cast<type const*>(bp_ptr);
        }
    };

    template <std::size_t R0, std::size_t R1, std::size_t C, typename T>
    UI_ALWAYS_INLINE constexpr auto join_rows(
        VecMat<R0, C, T> const& x,
        VecMat<R1, C, T> const& y
    ) noexcept -> VecMat<R0 + R1, C, T> {
        auto res = VecMat<R0 + R1, C, T>{};
        for (auto i = 0ul; i < R0; ++i) res.val[i] = x.val[i];
        for (auto i = 0ul; i < R1; ++i) res.val[R0 + i] = y.val[i];
        return res;
    }

    template <std::size_t R, std::size_t C, typename T>
    UI_ALWAYS_INLINE constexpr auto join_cols(
        VecMat<R, C, T> const& x,
        VecMat<R, C, T> const& y
    ) noexcept -> VecMat<R, 2 * C, T> {
        auto res = VecMat<R, 2 * C, T>{};
        for (auto i = 0ul; i < R; ++i) res.val[i] = join(x.val[i], y.val[i]);
        return res;
    }
} // namespace ui

#endif // AMT_UI_MATRIX_HPP
