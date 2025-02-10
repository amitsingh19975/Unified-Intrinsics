#ifndef AMT_UI_BASE_VEC_HPP
#define AMT_UI_BASE_VEC_HPP

#include "base.hpp"
#include "maths.hpp"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <cstring>
#include <type_traits>
#include "features.hpp"

#ifdef UI_ARM_HAS_NEON
    #include "arch/arm/join.hpp"
#endif

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

        UI_ALWAYS_INLINE constexpr Vec(element_t x, element_t y) noexcept requires (N == 2)
            : lo(x)
            , hi(y)
        {}

        UI_ALWAYS_INLINE constexpr Vec(
            Vec<elements / 2, element_t> const& lo,
            Vec<elements / 2, element_t> const& hi
        ) noexcept requires (elements != 4)
            : lo(lo)
            , hi(hi)
        {}

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
            store(li.begin(), li.size());
        }

        UI_ALWAYS_INLINE constexpr Vec(std::span<element_t> li) noexcept {
            store(li.data(), li.size());
        }

        explicit operator std::span<element_t>() const noexcept {
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
            requires ((... && (std::convertible_to<Us, element_t>)) && (sizeof...(Us) > 1) && (sizeof...(Us) <= N))
        UI_ALWAYS_INLINE static constexpr auto load(Us... args) noexcept -> Vec {
            std::array<element_t, elements> res = { args... };
            return load(res);
        }

        UI_ALWAYS_INLINE static constexpr auto load(element_t val) noexcept -> Vec;

        template <unsigned Lane, std::size_t M>
        UI_ALWAYS_INLINE static constexpr auto load(Vec<M, T> const&) noexcept -> Vec; 

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

        UI_ALWAYS_INLINE static constexpr auto load(T const* const UI_RESTRICT in, size_type size) noexcept {
            auto res = Vec{};
            res.store(in, size);
            return res;
        }

        UI_ALWAYS_INLINE static constexpr auto load(std::span<T> data) noexcept {
            auto res = Vec{};
            res.store(data);
            return res;
        }

        UI_ALWAYS_INLINE static constexpr auto load(element_t val) noexcept -> Vec {
            return { .val = val };
        }

        template <unsigned Lane, std::size_t M>
        UI_ALWAYS_INLINE static constexpr auto load(Vec<M, T> const&) noexcept -> Vec;

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
        #ifdef UI_ARM_HAS_NEON
        return arm::neon::join_impl(x, y); 
        #else
        return { x, y };
        #endif
    }

    namespace internal {
        template <typename T>
        struct is_vec_impl: std::false_type {};

        template <std::size_t N, typename T>
        struct is_vec_impl<Vec<N, T>>: std::true_type {};

        template <typename T>
        concept is_vec = is_vec_impl<std::decay_t<T>>::value;
        
    
        template <typename A, typename... Args>
            requires (is_vec<A> && (is_vec<Args> && ...))
        constexpr auto is_valid_map_args() {
            return ((A::elements == Args::elements) && ...);
        }

        template <typename... Args>
        concept valid_vec_map_arg = is_valid_map_args<Args...>();

        template <typename A, typename... Args>
        constexpr auto map_arg_vec_result_helper() -> std::decay_t<A>;

        template <typename... Args>
        using map_arg_vec_result_t = decltype(map_arg_vec_result_helper<Args...>());

        template <typename T>
        struct Mask {
            using type = std::make_unsigned_t<T>;
        };

        template <>
        struct Mask<float> {
            using type = std::uint32_t;
        };

        template <>
        struct Mask<double> {
            using type = std::uint64_t;
        };

        template <std::size_t N, typename T>
        struct Mask<Vec<N, T>>: Mask<T> {};
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

    template <typename T>
    using mask_inner_t = internal::Mask<std::decay_t<T>>::type;

    template <std::size_t N, typename T>
    using mask_t = Vec<N, mask_inner_t<T>>;

} // namespace ui

#endif // AMT_UI_BASE_VEC_HPP
