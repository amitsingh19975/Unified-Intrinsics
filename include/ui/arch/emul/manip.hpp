#ifndef AMT_ARCH_EMUL_MANIP_HPP
#define AMT_ARCH_EMUL_MANIP_HPP

#include "cast.hpp"
#include <concepts>
#include <utility>

namespace ui::emul {

// MARK: Copy vector lane
    template <unsigned ToLane, unsigned FromLane, std::size_t N, std::size_t M, typename T>
        requires (ToLane < N && FromLane < M && std::is_arithmetic_v<T>)
    UI_ALWAYS_INLINE static constexpr auto copy(
        Vec<N, T> const& to,
        Vec<M, T> const& from
    ) noexcept -> Vec<N, T> {
        auto res = to;
        res[ToLane] = from[FromLane]; 
        return res;
    }
// !MARK

// MARK: Reverse bits within elements
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto reverse_bits(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            return maths::bit_reverse(v_);
        }, v);
    }
// !MARK

// MARK: Reverse elements
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto reverse(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) return v;
        else {
            constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>, Vec<N, T> const& v) {
                auto res = Vec<N, T>{};
                ((res[N - Is - 1] = v[Is]),...);
                return res;
            };
            return helper(std::make_index_sequence<N>{}, v);
        }
    }
// !MARK

// MARK: Zip
    /*
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_low(a, b) == Vec<4, int>::load(0, 4, 1, 5))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto zip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((res[2 * Is] = a_[Is], res[2 * Is + 1] = b_[Is]), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }

    /**
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(zip_high(a, b) == Vec<4, int>::load(2, 6, 3, 7))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto zip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((res[2 * Is] = a_[N / 2 + Is], res[2 * Is + 1] = b_[N / 2 + Is]), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }

    /**
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(unzip_low(a, b) == Vec<4, int>::load(0, 2, 4, 6))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto unzip_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((
                res[Is]         = a_[2 * Is],
                res[Is + N / 2] = b_[2 * Is]
            ), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }

    /**
     * @code
     * auto a = Vec<4, int>::load(0, 1, 2, 3);
     * auto b = Vec<4, int>::load(4, 5, 6, 7);
     * assert(unzip_low(a, b) == Vec<4, int>::load(1, 3, 5, 7))
     * @codeend
    */
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto unzip_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((
                res[Is]         = a_[2 * Is + 1],
                res[Is + N / 2] = b_[2 * Is + 1]
            ), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }
// !MARK

// MARK: Transpose elements
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto transpose_low(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((
                res[2 * Is]     = a_[2 * Is],
                res[2 * Is + 1] = b_[2 * Is]
            ), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE static constexpr auto transpose_high(
        Vec<N, T> const& a,
        Vec<N, T> const& b
    ) noexcept -> Vec<N, T> {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& a_,
            Vec<N, T> const& b_
        ) {
            auto res = Vec<N, T>{};
            ((
                res[2 * Is]     = a_[2 * Is + 1],
                res[2 * Is + 1] = b_[2 * Is + 1]
            ), ...);
            return res;
        };
        return helper(std::make_index_sequence<N / 2>{}, a, b);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_MANIP_HPP
