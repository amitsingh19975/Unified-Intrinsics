#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <functional>
#include <limits>
#include <numeric>
#include <print>
#include "ui.hpp"
#include <cstdint>
#include <type_traits>

using namespace ui;

template <std::size_t N, typename T>
struct DataGenerator {
    static constexpr auto make() noexcept -> Vec<N, T> {
        std::array<T, N> data;
        std::iota(data.begin(), data.end(), 1);
        data[0] = std::numeric_limits<T>::min();
        data[1] = std::numeric_limits<T>::max();
        return Vec<N, T>::load(data.data(), data.size());
    }

    static constexpr auto cyclic_make(T min, T max) noexcept -> Vec<N, T> {
        std::array<T, N> data;
        for (auto i = 0u; i < N; ++i) {
            data[i] = static_cast<T>(min + (N % (max + 1)));
        }
        return Vec<N, T>::load(data.data(), data.size());
    }
};

template <unsigned I>
using index_t = std::integral_constant<unsigned, I>;

template <std::size_t... Is, typename Fn>
    requires (std::invocable<Fn, index_t<0>>)
auto for_each(Fn&& fn) {
    (std::invoke(fn, index_t<Is>{}), ...);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << int(v[i]) << " << " << I << ")[" << int(type(v[i] << I)) <<"] == " << int(res[i]));
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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
            for (auto i = 3u; i < N; ++i) REQUIRE(v[i - 1] == i);
        }

        WHEN("Normal left shift using compile-time count") {
            for_each<1, 2, 3, 4, 5, 6, 7>([&]<unsigned I>(index_t<I>){
                auto res = shift_left<I>(v);
                for (auto i = 0u; i < N; ++i) {
                INFO("['" << i << "']: (" << v[i] << " << " << I << ")[" << type(v[i] << I) <<"] == " << res[i]);
                    REQUIRE(static_cast<type>(v[0] << I) == res[0]);
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

