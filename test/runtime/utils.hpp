#include <concepts>
#include <limits>
#include <numeric>
#include <print>
#include "ui.hpp"
#include <random>
#include <utility>

template <unsigned I>
using index_t = std::integral_constant<unsigned, I>;

template <std::size_t N, typename T>
struct DataGenerator {
    static constexpr auto make() noexcept -> ui::Vec<N, T> {
        std::array<T, N> data;
        data[0] = std::numeric_limits<T>::min();
        data[1] = std::numeric_limits<T>::max();
        if constexpr (N > 2) std::iota(data.begin() + 2, data.end(), 0);
        return ui::Vec<N, T>::load(data.data(), data.size());
    }

    static constexpr auto cyclic_make(T min, T max) noexcept -> ui::Vec<N, T> {
        std::array<T, N> data;
        for (auto i = 0u; i < N; ++i) {
            data[i] = static_cast<T>(min + (i % (max + 1)));
        }
        return ui::Vec<N, T>::load(data.data(), data.size());
    }

    static constexpr auto random(std::size_t seed = 0) noexcept -> ui::Vec<N, T> {
        std::mt19937 rng(seed);
        std::array<T, N> data;
        if constexpr (std::integral<T>) {
            std::uniform_int_distribution<T> dist(
               std::numeric_limits<T>::min(),
               std::numeric_limits<T>::max()
            );
            for (auto i = 0ul; i < N; ++i) data[i] = dist(rng);
        } else {
            std::uniform_real_distribution<T> dist(-100, 100);
            for (auto i = 0ul; i < N; ++i) data[i] = dist(rng);
        }
        return ui::Vec<N, T>::load(data.data(), data.size());
    }

    template <typename Fn>
    static constexpr auto make_mask(Fn&& fn) -> ui::mask_t<N, T> {
        std::array<ui::mask_inner_t<T>, N> data;
        for (auto i = 0ul; i < N; ++i) {
            if constexpr (
                std::invocable<Fn, int> ||
                std::invocable<Fn, int&>
            ) {
                data[i] = std::invoke(fn, i);
            } else {
                data[i] = std::invoke(fn);
            }
        }
        return ui::mask_t<N, T>::load(data.data(), data.size());
    }
};

template <std::floating_point T = float>
static constexpr T eps = T(0.1);
