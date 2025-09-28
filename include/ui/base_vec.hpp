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
#include <iterator>
#include <span>
#include <cstring>
#include <type_traits>
#include "features.hpp"
#include "float.hpp"
#include "forward.hpp"

#ifdef UI_ARM_HAS_NEON
    #include "arch/arm/join.hpp"
#endif

#if UI_CPU_SSE_LEVEL > UI_CPU_SSE_LEVEL_SSE41
    #include "arch/x86/join.hpp"
#endif

#ifdef UI_EMPSCRIPTEN
    #include "arch/wasm/join.hpp"
#endif

namespace ui {

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
            Vec<elements / 2, element_t> const& low,
            Vec<elements / 2, element_t> const& high
        ) noexcept requires (elements != 4)
            : lo(low)
            , hi(high)
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

        auto data() noexcept -> element_t* {
            return reinterpret_cast<element_t*>(this);
        }

        auto data() const noexcept -> element_t const* {
            return reinterpret_cast<element_t const*>(this);
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
            requires ((... && (std::convertible_to<Us, element_t> || std::same_as<Us, element_t>)) && (sizeof...(Us) > 1) && (sizeof...(Us) <= N))
        UI_ALWAYS_INLINE static constexpr auto load(Us... args) noexcept -> Vec {
            std::array<element_t, elements> res = { static_cast<element_t>(args)... };
            return load(res);
        }

        UI_ALWAYS_INLINE static constexpr auto load(element_t val) noexcept -> Vec;
        UI_ALWAYS_INLINE static constexpr auto zeroed() noexcept -> Vec;

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

        auto data() noexcept -> element_t* {
            return reinterpret_cast<element_t*>(this);
        }

        auto data() const noexcept -> element_t const* {
            return reinterpret_cast<element_t const*>(this);
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
        #elif UI_CPU_SSE_LEVEL > UI_CPU_SSE_LEVEL_SSE41
            return x86::join_impl(x, y); 
        #elif defined(UI_EMPSCRIPTEN)
            return wasm::join_impl(x, y); 
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
        struct Mask<float16> {
            using type = std::uint16_t;
        };

        template <>
        struct Mask<bfloat16> {
            using type = std::uint16_t;
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
            auto lane = [&](std::size_t i) -> vec_type {
                return fn(args[i]...);
            };

            std::array<vec_type, N> res { lane(Is)... };
            return Vec<N, vec_type>::load(res.data(), res.size());
        };
        return helper(std::make_index_sequence<N>{});
    }

    template <typename T>
    using mask_inner_t = internal::Mask<std::decay_t<T>>::type;

    template <std::size_t N, typename T>
    using mask_t = Vec<N, mask_inner_t<T>>;

    template <std::size_t Len, std::size_t N, typename T>
        requires (Len >= 2)
    UI_ALWAYS_INLINE constexpr auto join(
        std::span<Vec<N, T>> vs
    ) noexcept -> Vec<Len * N, T> {
        if constexpr (Len == 2) return join(vs[0], vs[1]);
        else return join(
            join<Len / 2, N, T>(vs),
            join<Len / 2, N, T>({ vs.data() + Len / 2, Len / 2 })
        );
    }

    template <internal::is_vec T0, internal::is_vec... Ts>
        requires (maths::is_power_of_2(sizeof...(Ts) + 1) && (sizeof...(Ts) + 1 > 3))
    UI_ALWAYS_INLINE constexpr auto join(
        T0 const& v0,
        Ts const&... vs
    ) noexcept {
        std::array<T0, sizeof...(Ts) + 1> temp = { v0, vs... };
        return join<sizeof...(Ts) + 1, T0::elements, typename T0::element_t>({ temp.data(), temp.size() });
    }

    namespace internal {
        template <std::size_t N, typename T>
        constexpr auto get_int_mask_type() {
            if constexpr (N == 8) return std::uint8_t{};
            else if constexpr (N == 16) return std::uint16_t{};
            else if constexpr (N == 32) return std::uint32_t{};
            else if constexpr (N == 64) return std::uint64_t{};
        }
    } // namespace internal

    template <std::size_t N, typename T>
    struct IntMask {
        using size_type = std::size_t;
        static constexpr bool is_packed = []{
            if constexpr (sizeof(T) == 1 && N == 16) {
                #ifdef UI_CPU_ARM64
                return false;
                #else
                return true;
                #endif
            }
            return true;
        }();
        using base_type = std::conditional_t<is_packed, decltype(internal::get_int_mask_type<N, T>()), std::uint64_t>;

        static_assert(!std::is_void_v<base_type>, "invalid N; it cannot be represented using machine integer type");

        static constexpr base_type all_mask = is_packed ? ~base_type{} : 0xffff'ffff'ffff'ffffll;

        base_type mask;

        constexpr IntMask() noexcept = default;
        constexpr IntMask(IntMask const&) noexcept = default;
        constexpr IntMask(IntMask &&) noexcept = default;
        constexpr IntMask& operator=(IntMask const&) noexcept = default;
        constexpr IntMask& operator=(IntMask &&) noexcept = default;
        constexpr ~IntMask() noexcept = default;

        constexpr IntMask(base_type m) noexcept
            : mask(m)
        {}

        constexpr IntMask(mask_t<N, T> const& m) noexcept;

        constexpr auto operator[](size_type k) const noexcept -> bool {
            assert((k < sizeof(T) * 8) && "out of bound");
            if constexpr (is_packed) {
                return mask & (1 << k);
            } else {
                return mask & (1 << (k * 8 - 1));
            }
        }

        constexpr auto all() const noexcept -> bool {
            if constexpr (is_packed) {
                return mask == ~base_type{};
            } else {
                return mask == all_mask;
            }
        }

        constexpr auto any() const noexcept -> bool {
            if constexpr (is_packed) {
                return mask & ~base_type{};
            } else {
                return mask & all_mask;
            }
        }

        constexpr auto none() const noexcept -> bool {
            return !any();
        }

        constexpr auto operator&(IntMask other) const noexcept -> IntMask {
            return { mask & other.mask };
        }

        constexpr auto operator&(base_type other) const noexcept -> IntMask {
            return { mask & other };
        }

        constexpr auto operator|(IntMask other) const noexcept -> IntMask {
            return { mask | other.mask };
        }

        constexpr auto operator|(base_type other) const noexcept -> IntMask {
            return { mask | other };
        }

        constexpr auto operator^(IntMask other) const noexcept -> IntMask {
            return { mask ^ other.mask };
        }

        constexpr auto operator^(base_type other) const noexcept -> IntMask {
            return { mask ^ other };
        }

        constexpr auto operator~() const noexcept -> IntMask {
            return { ~mask };
        }

        constexpr auto first_match() const noexcept -> size_type {
            auto res = static_cast<size_type>(std::countr_zero(mask));
            if constexpr (is_packed) return res;
            else return res >> 2;
        }

        constexpr auto last_match() const noexcept -> size_type {
            return N - first_match();
        }

        constexpr operator bool() const noexcept {
            return static_cast<bool>(mask);
        }

        struct Iterator {
            using value_type = unsigned;
            using reference = value_type&;
            using const_reference = value_type const&;
            using pointer = value_type*;
            using const_pointer = value_type const*;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;

            constexpr Iterator() noexcept = default; 
            constexpr Iterator(Iterator const&) noexcept = default;
            constexpr Iterator(Iterator &&) noexcept = default;
            constexpr Iterator& operator=(Iterator const&) noexcept = default;
            constexpr Iterator& operator=(Iterator &&) noexcept = default;
            constexpr ~Iterator() noexcept = default;

            constexpr Iterator(base_type m) noexcept
                : m_mask(m)
                , m_index(get_index(m))
            {
                if constexpr (!is_packed) {
                    m_mask &= 0x8888'8888'8888'8888;
                }
            }

            constexpr const_pointer operator*() const noexcept { return &m_index; }
            constexpr pointer operator->() noexcept { return m_index; }

            constexpr Iterator& operator++() noexcept { 
                m_mask &= m_mask - 1;
                m_index = get_index(m_mask);
                return *this;
            }

            constexpr Iterator& operator++(int) noexcept { 
                auto temp = *this;
                ++(*this);
                return temp;
            }

            friend constexpr auto operator==(Iterator lhs, Iterator rhs) noexcept {
                return lhs.m_mask == rhs.m_mask;
            }

            friend constexpr auto operator!=(Iterator lhs, Iterator rhs) noexcept {
                return lhs.m_mask != rhs.m_mask;
            }
        private:
            static constexpr auto get_index(base_type m) noexcept -> value_type {
                auto temp = static_cast<value_type>(std::countr_zero(m));
                if constexpr (!is_packed) return temp >> 2;
                return temp;
            }
        private:
            base_type m_mask{};
            value_type m_index{};
        };


        using iterator = Iterator;
        using const_iterator = iterator const;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        constexpr auto begin() const noexcept -> const_iterator {
            return { mask };
        }

        constexpr auto end() const noexcept -> const_iterator {
            return {};
        }

        constexpr auto begin() noexcept -> iterator {
            return { mask };
        }

        constexpr auto end() noexcept -> iterator {
            return {};
        }

        constexpr auto rbegin() const noexcept -> const_reverse_iterator {
           return std::reverse_iterator(begin()); 
        }

        constexpr auto rend() const noexcept -> const_reverse_iterator {
           return std::reverse_iterator(end()); 
        }

        constexpr auto rbegin() noexcept -> reverse_iterator {
           return std::reverse_iterator(begin()); 
        }

        constexpr auto rend() noexcept -> reverse_iterator {
           return std::reverse_iterator(end()); 
        }
    };


    template <std::size_t N, typename T>
    IntMask(Vec<N, T> const& m) noexcept -> IntMask<N, T>;
} // namespace ui

#endif // AMT_UI_BASE_VEC_HPP
