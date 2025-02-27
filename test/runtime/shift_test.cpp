#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <functional>
#include <algorithm>
#include <limits>
#include <numeric>
#include <print>
#include "ui.hpp"
#include "ui/arch/arm/shift.hpp"
#include "ui/arch/basic.hpp"
#include <cstdint>
#include <type_traits>
#include <random>
#include <vector>

using namespace ui;

template <std::size_t N, typename T>
struct DataGenerator {
    static constexpr auto make() noexcept -> Vec<N, T> {
        std::array<T, N> data;
        data[0] = std::numeric_limits<T>::min();
        data[1] = std::numeric_limits<T>::max();
        if constexpr (N > 2) std::iota(data.begin() + 2, data.end(), 0);
        return Vec<N, T>::load(data.data(), data.size());
    }

    static constexpr auto cyclic_make(T min, T max) noexcept -> Vec<N, T> {
        std::array<T, N> data;
        for (auto i = 0u; i < N; ++i) {
            data[i] = static_cast<T>(min + (N % (max + 1)));
        }
        return Vec<N, T>::load(data.data(), data.size());
    }

    static constexpr auto random(std::size_t seed = 0) noexcept -> Vec<N, T> {
        std::mt19937 rng(seed);
        std::array<T, N> data;
        if constexpr (std::integral<T>) {
            std::uniform_int_distribution<T> dist(
               std::numeric_limits<T>::min(),
               std::numeric_limits<T>::max()
            );
            for (auto i = 0ul; i < N; ++i) data[i] = dist(rng);
        } else {
            std::uniform_int_distribution<T> dist(-100, 100);
            for (auto i = 0ul; i < N; ++i) data[i] = dist(rng);
        }
        return Vec<N, T>::load(data.data(), data.size());
    }
};

template <unsigned I>
using index_t = std::integral_constant<unsigned, I>;

template <std::size_t... Is, typename Fn>
    requires (std::invocable<Fn, index_t<2>>)
auto for_each(Fn&& fn) {
    std::tuple<index_t<Is>...> ts;
    #define INVOKE(I) std::invoke(fn, std::get<I>(ts))
    // For some reason this is crashing the compiler.
    /*(std::invoke(fn, index_t<Is>{}), ...);*/
    INVOKE(0);
    INVOKE(1);
    INVOKE(2);
    INVOKE(3);
    INVOKE(4);
    INVOKE(5);
    INVOKE(6);
    #undef INVOKE
}

TEST_CASE( VEC_ARCH_NAME " Left Shift", "[shift][left]" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << 2 << ")[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::int16_t>);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE((v[i] << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Unsigned 8bit Integer") {
        using type = std::uint8_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << 2 << ")[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::uint16_t>);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE((v[i] << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Signed 16bit Integer") {
        using type = std::int16_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 16ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << 2 << ")[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::int32_t>);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE((v[i] << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Unsigned 16bit Integer") {
        using type = std::uint16_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 16ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << 2 << ")[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE(static_cast<type>(std::min(int(v[i]) << 2, int(max))) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::uint32_t>);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << 2)[" << int(type(v[i] << 2)) <<"] == " << int(res[i]));
                REQUIRE((v[i] << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Signed 32bit Integer") {
        using type = std::int32_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 8ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << 2 << ")[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::int64_t>);
            for (auto i = 0u; i < N; ++i) {
                auto val = std::int64_t(v[i]);
                INFO("['" << i << "']: (" << val << " << 2)[" << val << 2 <<"] == " << res[i]);
                REQUIRE((val << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Unsigned 32bit Integer") {
        using type = std::uint32_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 8ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << 2 << ")[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Widening left shift") {
            auto res = widening_shift_left<2>(v);
            static_assert(std::same_as<decltype(res)::element_t, std::uint64_t>);
            for (auto i = 0u; i < N; ++i) {
                auto val = std::int64_t(v[i]);
                INFO("['" << i << "']: (" << val << " << 2)[" << val << 2 <<"] == " << res[i]);
                REQUIRE((val << 2) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Signed 64bit Integer") {
        using type = std::int64_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 4ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << 2 << ")[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }


        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }

    GIVEN("Unsigned 64bit Integer") {
        using type = std::uint64_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 4ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[i] << I) == res[i]);
                }
            });
        }

        WHEN("Normal left shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_left(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << s[i] << ")[" << int(type(v[i] << s[i])) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift compile-time count") {
            auto res = sat_shift_left<2>(v); // v << 2
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_left(v, Vec<N, utype>::load(2)); // sat(v << 2)
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Rounding left shift") {
            auto res = rounding_shift_left(v, Vec<N, utype>::load(2));
            for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << 2 << ")[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(v[i] << 2) == res[i]);
            }
        }

        WHEN("Saturting rounding left shift") {
            auto res = sat_rounding_shift_left(v, Vec<N, utype>::load(2));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max);
            for (auto i = 2u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << 2)[" << type(v[i] << 2) <<"] == " << res[i]);
                REQUIRE(static_cast<type>(std::min(v[i] << 2, max)) == res[i]);
            }
        }

        WHEN("Insert left shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            auto res = insert_shift_left<2>(v, bs);
            constexpr auto mask = utype((1 << 2) - 1);
            for (auto i = 0u; i < N; ++i) {
                auto a = type(v[i]) & mask;
                auto b = type(bs[i]); 
                auto sb = type((b << 2) & ~mask);

                auto ans = a | sb;

                INFO("['" << i << "']: " <<
                    std::format("{} == {}", ans, res[i])
                );
                REQUIRE(ans == res[i]);
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Right Shift", "[shift][right]" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using utype = std::make_unsigned_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::make();
        INFO("[Vec]: " << std::format("{}", v));
        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        THEN("Elements are correct") {
            REQUIRE(v[0] == min);
            REQUIRE(v[1] == max);
            for (auto i = 2u; i < N; ++i) REQUIRE(v[i] == i - 2);
        }

        WHEN("Normal right shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_right<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO(std::format("[{}]: ({:08b} >> {})[{:08b}] == {:08b}", i, v[i], I, type(v[i] >> I), res[i]));
                    REQUIRE(static_cast<type>(v[i] >> I) == res[i]);
                }
            });
        }

        WHEN("Normal right shift using runtime count") {
            auto s = DataGenerator<N, std::make_unsigned_t<type>>::cyclic_make(1, 7);
            auto res = shift_right(v, s);
            for (auto i = 0u; i < N; ++i) {
                INFO(std::format("[{}]: ({:08b} >> {})[{:08b}] == {:08b}", i, v[i], s[i], type(v[i] >> s[i]), res[i]));
                REQUIRE(static_cast<type>(v[i] >> s[i]) == res[i]);
            }
        }

        WHEN("Saturating left shift runtime count") {
            auto res = sat_shift_right(v, Vec<N, utype>::load(2)); // sat(v >> 2)
            REQUIRE(res[0] == min >> 2);
            REQUIRE(res[1] == max >> 2);
            for (auto i = 2u; i < N; ++i) {
                INFO(std::format("[{}]: ({:08b} >> {})[{:08b}] == {:08b}", i, v[i], 2, type(v[i] >> 2), res[i]));
                REQUIRE(static_cast<type>(std::clamp(type(v[i] >> 2), min, max)) == res[i]);
            }
        }

        WHEN("Rounding right shift") {
            auto res = rounding_shift_right(v, Vec<N, utype>::load(2));
            using wtype = ui::internal::widening_result_t<type>;
            for (auto i = 0u; i < N; ++i) {
                auto temp = wtype(v[i] + (wtype(1) << (2 - 1))); 
                INFO(std::format("[{}]: ({:08b} >> {})[{:08b}] == {:08b}", i, temp, 2, type(temp >> 2), res[i]));
                REQUIRE(static_cast<type>(temp >> 2) == res[i]);
            }
        }

        WHEN("Saturting rounding right shift") {
            auto res = sat_rounding_shift_right(v, Vec<N, utype>::load(2));
            INFO(std::format("[sat_round(V >> 2)]: {}", res));
            REQUIRE(res[0] == -0x20);
            REQUIRE(res[1] == 0x20);
            REQUIRE(res[2] == 0);
            REQUIRE(res[3] == 0);
            REQUIRE(res[4] == 1);
            REQUIRE(res[5] == 1);
            REQUIRE(res[6] == 1);
            REQUIRE(res[7] == 1);
            REQUIRE(res[8] == 2);
            REQUIRE(res[9] == 2);
            REQUIRE(res[10] == 2);
            REQUIRE(res[11] == 2);
            REQUIRE(res[12] == 3);
            REQUIRE(res[13] == 3);
            REQUIRE(res[14] == 3);
            REQUIRE(res[15] == 3);
            REQUIRE(res[16] == 4);
            REQUIRE(res[17] == 4);
            REQUIRE(res[18] == 4);
            REQUIRE(res[19] == 4);
            REQUIRE(res[20] == 5);
            REQUIRE(res[31] == 7);
        }

        WHEN("Accumulating rounding right shift") {
            auto res = rounding_shift_right_accumulate<2>(Vec<N, type>::load(1), v);
            INFO(std::format("Vec({}s) + ({} >> 2) == {:0x}", 1, v, res));
            REQUIRE(res[0] == -0x1f);
            REQUIRE(res[1] == 0x21);
            REQUIRE(res[2] == 1);
            REQUIRE(res[3] == 1);
            REQUIRE(res[4] == 2);
            REQUIRE(res[5] == 2);
            REQUIRE(res[6] == 2);
            REQUIRE(res[7] == 2);
            REQUIRE(res[8] == 3);
            REQUIRE(res[9] == 3);
            REQUIRE(res[10] == 3);
            REQUIRE(res[11] == 3);
            REQUIRE(res[12] == 4);
            REQUIRE(res[13] == 4);
            REQUIRE(res[14] == 4);
            REQUIRE(res[15] == 4);
            REQUIRE(res[16] == 5);
            REQUIRE(res[17] == 5);
            REQUIRE(res[18] == 5);
            REQUIRE(res[19] == 5);
            REQUIRE(res[20] == 6);
            REQUIRE(res[31] == 8);
        }

        WHEN("Narrowing right shift") {
            /*auto res = narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Saturating narrowing right shift") {
            /*auto res = sat_narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Rounding narrowing right shift") {
            /*auto res = rounding_narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Saturating unsigned narrowing right shift") {
            /*auto res = sat_unsigned_narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Saturating rounding narrowing right shift") {
            /*auto res = sat_rounding_narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Saturating rounding unsigned narrowing right shift") {
            /*auto res = sat_rounding_unsigned_narrowing_shift_right<2>(v);*/
            // This not compile
        }

        WHEN("Insert right shift") {
            auto bs = sat_add(v, Vec<N, type>::load(33));
            std::println("BS: {}", bs);
            auto res = insert_shift_right<2>(v, bs);
            INFO(std::format("insert(v, bs): {:x}", res));
            REQUIRE(res[0] == -0x7c);
            REQUIRE(res[1] == -0x1);
            REQUIRE(res[2] == -0x7c);
            REQUIRE(res[3] == -0x77);
            REQUIRE(res[4] == -0x72);
            REQUIRE(res[5] == -0x6d);
            REQUIRE(res[6] == -0x6c);
            REQUIRE(res[7] == -0x67);
            REQUIRE(res[8] == -0x62);
            REQUIRE(res[9] == -0x5d);
            REQUIRE(res[10] == -0x5c);
            REQUIRE(res[11] == -0x57);
            REQUIRE(res[12] == -0x52);
            REQUIRE(res[13] == -0x4d);
            REQUIRE(res[31] == -0x7);
        }
    }
}

