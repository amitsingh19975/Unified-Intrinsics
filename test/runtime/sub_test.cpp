
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <print>
#include "catch2/matchers/catch_matchers.hpp"
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

TEST_CASE( VEC_ARCH_NAME " 8bit Subtraction", "[subtraction][8bit]" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using wtype = std::int16_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = (l - r) >> 1;
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            /*auto res = high_narrowing_sub(v, Vec<N, type>::load(10));*/
            /*STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);*/
            /*INFO(std::format("hnsub(v, 10s): {}", res));*/
            // Cannot narrow below 8bit integer
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max - 10);
            REQUIRE(res[2] == -10);
            REQUIRE(res[3] == -9);
            REQUIRE(res[4] == -8);
            REQUIRE(res[5] == -7);
            REQUIRE(res[6] == -6);
            REQUIRE(res[7] == -5);
            REQUIRE(res[8] == -4);
            REQUIRE(res[9] == -3);
            REQUIRE(res[10] == -2);
            REQUIRE(res[11] == -1);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[31] == 19);
        }
    }
    GIVEN("Unsigned 8bit Integer") {
        using type = std::uint8_t;
        using wtype = std::uint16_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = type((l - r) >> 1);
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            /*auto res = high_narrowing_sub(v, Vec<N, type>::load(10));*/
            /*STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);*/
            /*INFO(std::format("hnsub(v, 10s): {}", res));*/
            // Cannot narrow below 8bit integer
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max - 10);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 0);
            REQUIRE(res[ 4] == 0);
            REQUIRE(res[ 5] == 0);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[10] == 0);
            REQUIRE(res[11] == 0);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[31] == 19);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 16bit Subtraction", "[subtraction][16bit]" ) {
    GIVEN("Signed 16bit Integer") {
        using type = std::int16_t;
        using wtype = std::int32_t;
        using ntype = std::int8_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = (l - r) >> 1;
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 127, 127, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max - 10);
            REQUIRE(res[2] == -10);
            REQUIRE(res[3] == -9);
            REQUIRE(res[4] == -8);
            REQUIRE(res[5] == -7);
            REQUIRE(res[6] == -6);
            REQUIRE(res[7] == -5);
            REQUIRE(res[8] == -4);
            REQUIRE(res[9] == -3);
            REQUIRE(res[10] == -2);
            REQUIRE(res[11] == -1);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
    GIVEN("Unsigned 16bit Integer") {
        using type = std::uint16_t;
        using wtype = std::uint32_t;
        using ntype = std::uint8_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = type((l - r) >> 1);
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max - 10);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 0);
            REQUIRE(res[ 4] == 0);
            REQUIRE(res[ 5] == 0);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[10] == 0);
            REQUIRE(res[11] == 0);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 32bit Subtraction", "[subtraction][16bit]" ) {
    GIVEN("Signed 32bit Integer") {
        using type = std::int32_t;
        using wtype = std::int64_t;
        using ntype = std::int16_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = l - r;
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = (l - r) >> 1;
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 32767, 32767, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max - 10);
            REQUIRE(res[2] == -10);
            REQUIRE(res[3] == -9);
            REQUIRE(res[4] == -8);
            REQUIRE(res[5] == -7);
            REQUIRE(res[6] == -6);
            REQUIRE(res[7] == -5);
            REQUIRE(res[8] == -4);
            REQUIRE(res[9] == -3);
            REQUIRE(res[10] == -2);
            REQUIRE(res[11] == -1);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
    GIVEN("Unsigned 32bit Integer") {
        using type = std::uint32_t;
        using wtype = std::uint64_t;
        using ntype = std::uint16_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("Widening subtraction with same type") {
            auto res = widening_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider right") {
            auto res = widening_sub(v, Vec<N, wtype>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(v, w(10s)): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Widening subtraction with wider left") {
            auto res = widening_sub(Vec<N, wtype>::load(10), v);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wsub(w(10s), v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = wtype(10);
                auto r = static_cast<wtype>(v[i]);
                auto ans = wtype(l - r);
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("Halving subtraction") {
            auto res = halving_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, type>);
            INFO(std::format("halving(v, 10s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = static_cast<wtype>(v[i]);
                auto r = wtype(10);
                auto ans = type((l - r) >> 1);
                INFO(std::format("[{}]: ({} - {}) / 2 == {}", i, l, r, ans));
                REQUIRE(ans == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max - 10);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 0);
            REQUIRE(res[ 4] == 0);
            REQUIRE(res[ 5] == 0);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[10] == 0);
            REQUIRE(res[11] == 0);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 64bit Subtraction", "[subtraction][16bit]" ) {
    GIVEN("Signed 64bit Integer") {
        using type = std::int64_t;
        using ntype = std::int32_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 2147483647, 2147483647, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[0] == min);
            REQUIRE(res[1] == max - 10);
            REQUIRE(res[2] == -10);
            REQUIRE(res[3] == -9);
            REQUIRE(res[4] == -8);
            REQUIRE(res[5] == -7);
            REQUIRE(res[6] == -6);
            REQUIRE(res[7] == -5);
            REQUIRE(res[8] == -4);
            REQUIRE(res[9] == -3);
            REQUIRE(res[10] == -2);
            REQUIRE(res[11] == -1);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
    GIVEN("Unsigned 64bit Integer") {
        using type = std::uint64_t;
        using ntype = std::uint32_t;
        using otype = std::conditional_t<std::is_signed_v<type>, std::make_unsigned_t<type>, std::make_signed_t<type>>;
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

        WHEN("Normal subtraction") {
            auto res = sub(v, Vec<N, type>::load(10));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(static_cast<type>(v[i] - 10) == res[i]);
            }
        }

        WHEN("High bit narrowing subtraction") {
            auto res = high_narrowing_sub(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, ntype>);
            INFO(std::format("hnsub(v, 10s): {}", res));
            ntype ts[] = { 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 0, 0, 0, 0 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Saturating subtraction") {
            auto res = sat_sub(v, Vec<N, type>::load(10));
            INFO(std::format("ssub(v, 10s): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max - 10);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 0);
            REQUIRE(res[ 4] == 0);
            REQUIRE(res[ 5] == 0);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[10] == 0);
            REQUIRE(res[11] == 0);
            REQUIRE(res[12] == 0);
            REQUIRE(res[13] == 1);
            REQUIRE(res[15] == 3);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Float16 Subtraction", "[subtraction][float16]" ) {
    using type = float16;
    static constexpr auto N = 8ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(float(v[ 0]), Catch::Matchers::WithinRel(float(min), eps<float>));
        REQUIRE_THAT(float(v[ 1]), Catch::Matchers::WithinRel(float(max), eps<float>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(float(v[ i]), Catch::Matchers::WithinRel(float(i - 2), eps<float>));
        }
    }

    WHEN("Normal subtraction") {
        auto res = sub(v, Vec<N, type>::load(10));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(float(v[i]) - 10, eps<float>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Bfloat16 Subtraction", "[subtraction][bfloat16]" ) {
    using type = bfloat16;
    static constexpr auto N = 8ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(float(v[ 0]), Catch::Matchers::WithinRel(float(min), eps<float>));
        REQUIRE_THAT(float(v[ 1]), Catch::Matchers::WithinRel(float(max), eps<float>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(float(v[ i]), Catch::Matchers::WithinRel(float(i - 2), eps<float>));
        }
    }

    WHEN("Normal subtraction") {
        auto res = sub(v, Vec<N, type>::load(10));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(float(v[i] - 10), eps<float>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " float32 Subtraction", "[subtraction][float32]" ) {
    using type = float;
    static constexpr auto N = 8ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[ 0], Catch::Matchers::WithinRel(min, eps<float>));
        REQUIRE_THAT(v[ 1], Catch::Matchers::WithinRel(max, eps<float>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(v[ i], Catch::Matchers::WithinRel(float(i - 2), eps<float>));
        }
    }

    WHEN("Normal subtraction") {
        auto res = sub(v, Vec<N, type>::load(10));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(v[i] - 10, eps<float>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " float64 Subtraction", "[subtraction][float64]" ) {
    using type = double;
    static constexpr auto N = 8ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[ 0], Catch::Matchers::WithinRel(min, eps<double>));
        REQUIRE_THAT(v[ 1], Catch::Matchers::WithinRel(max, eps<double>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(v[ i], Catch::Matchers::WithinRel(double(i - 2), eps<double>));
        }
    }

    WHEN("Normal subtraction") {
        auto res = sub(v, Vec<N, type>::load(10));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(v[i] - 10, eps<double>));
        }
    }
}

