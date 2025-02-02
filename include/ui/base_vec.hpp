#ifndef AMT_UI_BASE_VEC_HPP
#define AMT_UI_BASE_VEC_HPP

#include "ui/base.hpp"
#include "ui/maths.hpp"
#include <bit>
#include <cstddef>
#include <cstring>
#include <span>
#include <cstring>
#include <type_traits>

namespace ui {

    template <std::size_t N, typename T>
    struct alignas(N * sizeof(T)) VecReg {
        static_assert(maths::is_power_of_2(N), "N should be a power of two!");
        static_assert(alignof(T) <= sizeof(T), "This should never be the case!");

        using element_t = T;
        using size_type = std::size_t;
        using base_type = VecReg<N / 2, T>;
        static constexpr size_type elements = N;

        base_type lo, hi;

        operator std::span<element_t const>() const noexcept {
            auto ptr = reinterpret_cast<T const*>(this);
            return { ptr, elements };
        }

        auto to_span() const noexcept {
            return static_cast<std::span<element_t const>>(*this);
        }

        constexpr auto data() noexcept -> element_t* {
            return static_cast<element_t*>(static_cast<void*>(this));
        }

        constexpr auto data() const noexcept -> element_t const* {
            return static_cast<element_t const*>(static_cast<void const*>(this));
        }

        constexpr auto operator[](size_type k) noexcept -> element_t& {
            return data()[k];
        }

        constexpr auto operator[](size_type k) const noexcept -> element_t {
            return data()[k];
        }
    };

    template <typename T>
    struct alignas(1 * sizeof(T)) VecReg<1, T> {
        static_assert(alignof(T) <= sizeof(T), "This should never be the case!");

        using element_t = T;
        using size_type = std::size_t;
        static constexpr size_type elements = 1;

        element_t val;

        template <typename U>
        UI_ALWAYS_INLINE constexpr auto cast() const noexcept -> U {
            return std::bit_cast<U>(*this);
        }

        operator std::span<element_t const>() const noexcept {
            auto ptr = reinterpret_cast<T const*>(this);
            return { ptr, elements };
        }

        auto to_span() const noexcept {
            return static_cast<std::span<element_t const>>(*this);
        }

        constexpr auto data() noexcept -> element_t* {
            return static_cast<element_t*>(static_cast<void*>(this));
        }

        constexpr auto data() const noexcept -> element_t const* {
            return static_cast<element_t const*>(static_cast<void const*>(this));
        }

        constexpr auto operator[]([[maybe_unused]] size_type k) noexcept -> element_t& {
            return val;
        }

        constexpr auto operator[]([[maybe_unused]] size_type k) const noexcept -> element_t {
            return val;
        }
    };

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto join(
        VecReg<N, T> const& x,
        VecReg<N, T> const& y
    ) noexcept -> VecReg<2 * N, T> {
        return { .lo = x, .hi = y };
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto load(T const* const UI_RESTRICT in, std::size_t size) noexcept -> VecReg<N, T> {
        auto res = VecReg<N, T>{};
        std::memcpy(res.data(), in, std::min(N, size) * sizeof(T));
        return res;
    }

    namespace internal {
        template <typename T>
        struct is_vec_reg_impl: std::false_type {};

        template <std::size_t N, typename T>
        struct is_vec_reg_impl<VecReg<N, T>>: std::true_type {};

        template <typename T>
        concept is_vec_reg = is_vec_reg_impl<std::decay_t<T>>::value;
        
    
        template <typename A, typename... Args>
            requires (is_vec_reg<A> && (is_vec_reg<Args> && ...))
        constexpr auto is_valid_map_args() {
            return ((A::elements == Args::elements) && ...);
        }

        template <typename... Args>
        concept valid_vec_map_arg = is_valid_map_args<Args...>();

        template <typename A, typename... Args>
        constexpr auto map_arg_vec_result_helper() -> std::decay_t<A>;

        template <typename... Args>
        using map_arg_vec_result_t = decltype(map_arg_vec_result_helper<Args...>());

    } // namespace internal

    template <typename Fn, internal::valid_vec_map_arg... Args>
    UI_ALWAYS_INLINE static constexpr auto map(Fn&& fn, Args&&... args) noexcept {
        using result_t = internal::map_arg_vec_result_t<Args...>;
        constexpr auto N = result_t::elements;

        using vec_type = decltype(fn(args[0]...));

        auto helper = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> VecReg<N, vec_type> {
            auto lane = [&](std::size_t i) {
                return fn(args[i]...);
            };

            std::array<vec_type, N> res = { lane(Is)... };
            return load<N>(res.data(), res.size());
        };
        return helper(std::make_index_sequence<N>{});
    }

} // namespace ui

#endif // AMT_UI_BASE_VEC_HPP
