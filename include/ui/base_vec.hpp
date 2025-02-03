#ifndef AMT_UI_BASE_VEC_HPP
#define AMT_UI_BASE_VEC_HPP

#include "ui/base.hpp"
#include "ui/maths.hpp"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <span>
#include <cstring>
#include <type_traits>

namespace ui {

    template<std::size_t N, typename T>
    struct alignas(N * sizeof(T)) Vec;

    template <std::size_t... Is, std::size_t N, typename T>
    static inline constexpr auto shuffle(Vec<N, T> const&) noexcept -> Vec<sizeof...(Is), T>;

    template <std::size_t N, typename T>
    struct alignas(N * sizeof(T)) Vec {
        static_assert(maths::is_power_of_2(N), "N should be a power of two!");
        static_assert(alignof(T) <= sizeof(T), "This should never be the case!");

        using element_t = T;
        using size_type = std::size_t;
        using base_type = Vec<N / 2, T>;
        static constexpr size_type elements = N;

        base_type lo, hi;

        UI_ALWAYS_INLINE constexpr Vec() noexcept = default;
        UI_ALWAYS_INLINE constexpr Vec(Vec const&) noexcept = default;
        UI_ALWAYS_INLINE constexpr Vec(Vec &&) noexcept = default;
        UI_ALWAYS_INLINE constexpr Vec& operator=(Vec const&) noexcept = default;
        UI_ALWAYS_INLINE constexpr Vec& operator=(Vec &&) noexcept = default;
        UI_ALWAYS_INLINE constexpr ~Vec() noexcept = default;

        UI_ALWAYS_INLINE constexpr Vec(element_t val) noexcept
            : lo(val)
            , hi(val)
        {}

        UI_ALWAYS_INLINE constexpr Vec(
            Vec<2, element_t> xy,
            element_t z,
            element_t w
        ) noexcept requires (elements == 4)
            : lo(xy)
            , hi(z, w)
        {}

        UI_ALWAYS_INLINE constexpr Vec(
            Vec<2, element_t> xy,
            Vec<2, element_t> zw
        ) noexcept requires (elements == 4)
            : lo(xy)
            , hi(zw)
        {}

        UI_ALWAYS_INLINE constexpr Vec(std::initializer_list<element_t> li) noexcept {
            store(li.data(), li.size());
        }

        UI_ALWAYS_INLINE constexpr Vec(std::span<element_t> li) noexcept {
            store(li.data(), li.size());
        }

        operator std::span<element_t>() const noexcept {
            return to_span();
        }

        auto to_span() const noexcept -> std::span<element_t> {
            auto ptr = const_cast<element_t*>(reinterpret_cast<element_t const*>(this));
            return { ptr, elements };
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

        constexpr auto size() const noexcept { return elements; }

        // MARK: Swizzling
        UI_ALWAYS_INLINE constexpr Vec<2, element_t>& xy() noexcept requires (N == 4) { return lo; }
        UI_ALWAYS_INLINE constexpr Vec<2, element_t>& zw() noexcept requires (N == 4) { return hi; }
        UI_ALWAYS_INLINE constexpr Vec<2, element_t> xy() const noexcept requires (N == 4) { return lo; }
        UI_ALWAYS_INLINE constexpr Vec<2, element_t> zw() const noexcept requires (N == 4) { return hi; }
        UI_ALWAYS_INLINE constexpr element_t& x() noexcept requires (N == 4 || N == 2) {
            if constexpr (elements == 4) return lo.lo.val;
            else return lo.val;
        }
        UI_ALWAYS_INLINE constexpr element_t x() const noexcept requires (N == 4 || N == 2) {
            if constexpr (elements == 4) return lo.lo.val;
            else return lo.val;
        }
        UI_ALWAYS_INLINE constexpr element_t& y() noexcept requires (N == 4 || N == 2) {
            if constexpr (elements == 4) return lo.hi.val;
            else return hi.val;
        }
        UI_ALWAYS_INLINE constexpr element_t y() const noexcept requires (N == 4 || N == 2) {
            if constexpr (elements == 4) return lo.hi.val;
            else return hi.val;
        }
        UI_ALWAYS_INLINE constexpr element_t& z() noexcept requires (N == 4) {
            return hi.lo.val;
        }
        UI_ALWAYS_INLINE constexpr element_t z() const noexcept requires (N == 4) {
            return hi.lo.val;
        }
        UI_ALWAYS_INLINE constexpr element_t& w() noexcept requires (N == 4) {
            return hi.hi.val;
        }
        UI_ALWAYS_INLINE constexpr element_t w() const noexcept requires (N == 4) {
            return hi.hi.val;
        }
        
        UI_ALWAYS_INLINE constexpr Vec<4, element_t> yxwz() const noexcept requires (N == 4) {
            return shuffle<1, 0, 3, 2>(*this);
        }
        UI_ALWAYS_INLINE constexpr Vec<4, element_t> zwxy() const noexcept requires (N == 4) {
            return shuffle<2, 3, 0, 1>(*this);
        }
        UI_ALWAYS_INLINE constexpr Vec<2, element_t> yx() const noexcept requires (N == 2) {
            return shuffle<1, 0>(*this);
        }
        UI_ALWAYS_INLINE constexpr Vec<4, element_t> xyxy() const noexcept requires (N == 2) {
            return { *this, *this };
        }

        // !MARK

        UI_ALWAYS_INLINE static constexpr auto load(element_t const* const UI_RESTRICT in, size_type size) noexcept {
            auto res = Vec{};
            res.store(in, size);
            return res;
        }

        UI_ALWAYS_INLINE static constexpr auto load(std::span<element_t> data) noexcept {
            auto res = Vec{};
            res.store(data);
            return res;
        }

        template <typename... Us>
            requires ((... && std::same_as<element_t, Us>) && (sizeof...(Us) > 1) && (sizeof...(Us) <= N))
        UI_ALWAYS_INLINE constexpr auto load(Us... args) noexcept -> Vec {
            std::array<element_t, elements> res = { args... };
            return load(res);
        }

        UI_ALWAYS_INLINE static constexpr auto load(element_t val) noexcept -> Vec;

        UI_ALWAYS_INLINE constexpr auto store(element_t const* const UI_RESTRICT in, size_type size) noexcept {
            assert(size >= N);
            if (std::is_constant_evaluated()) {
                std::copy_n(in, std::min(elements, size), data());
            } else {
                std::memcpy(data(), in, std::min(elements, size) * sizeof(T));
            }
        }

        UI_ALWAYS_INLINE constexpr auto store(std::span<element_t> data) noexcept {
            store(data.data(), data.size());
        }
    };

    template <typename T>
    struct alignas(1 * sizeof(T)) Vec<1, T> {
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

        static constexpr auto load(T const* const UI_RESTRICT in, size_type size) noexcept {
            auto res = Vec{};
            res.store(in, size);
            return res;
        }

        static constexpr auto load(std::span<T> data) noexcept {
            auto res = Vec{};
            res.store(data);
            return res;
        }

        static constexpr auto load(element_t val) noexcept -> Vec {
            return { .val = val };
        }

        constexpr auto store(T const* const UI_RESTRICT in, [[maybe_unused]] size_type size) noexcept {
            assert(size >= 1);
            val = in[0];
        }
    };

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto join(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<2 * N, T> {
        return { .lo = x, .hi = y };
    }

    namespace internal {
        template <typename T>
        struct is_vec_reg_impl: std::false_type {};

        template <std::size_t N, typename T>
        struct is_vec_reg_impl<Vec<N, T>>: std::true_type {};

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

        auto helper = [&]<std::size_t... Is>(std::index_sequence<Is...>) -> Vec<N, vec_type> {
            auto lane = [&](std::size_t i) {
                return fn(args[i]...);
            };

            std::array<vec_type, N> res = { lane(Is)... };
            return Vec<N, vec_type>::load(res.data(), res.size());
        };
        return helper(std::make_index_sequence<N>{});
    }

} // namespace ui

#endif // AMT_UI_BASE_VEC_HPP
