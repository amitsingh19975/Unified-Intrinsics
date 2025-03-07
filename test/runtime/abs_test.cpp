
#include <catch2/catch_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <print>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

TEST_CASE( VEC_ARCH_NAME " 8bit Absolute Operations", "[absolute][8bit]" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using wtype = std::int16_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { -118, 117, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{}]: {} == {}", i, int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 138, 117, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { -117, 118, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Absolute") {
            auto res = abs(v);
            INFO(std::format("abs(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(std::abs(v[i]))));
                REQUIRE(res[i] == static_cast<type>(std::abs(v[i])));
            }
        }

        WHEN("Saturating Absoulute") {
            auto res = sat_abs(v);
            INFO(std::format("sat_abs(v): {}", res));
            wtype ts[] = { 127, 127, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
    GIVEN("Unsigned 8bit Integer") {
        using type = std::uint8_t;
        using wtype = std::uint16_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { 10, 245, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{}]: {} == {}", i, int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 10, 245, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { 11, 246, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 16bit Absolute Operations", "[absolute][16bit]" ) {
    GIVEN("Signed 16bit Integer") {
        using type = std::int16_t;
        using wtype = std::int32_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { -32758, 32757, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 32778, 32757, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { -32757, 32758, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Absolute") {
            auto res = abs(v);
            INFO(std::format("abs(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(std::abs(v[i]))));
                REQUIRE(res[i] == static_cast<type>(std::abs(v[i])));
            }
        }

        WHEN("Saturating Absoulute") {
            auto res = sat_abs(v);
            INFO(std::format("sat_abs(v): {}", res));
            wtype ts[] = { 32767, 32767, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
    GIVEN("Unsigned 16bit Integer") {
        using type = std::uint16_t;
        using wtype = std::uint32_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { 10, 65525, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 10, 65525, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { 11, 65526, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 32bit Absolute Operations", "[absolute][32bit]" ) {
    GIVEN("Signed 32bit Integer") {
        using type = std::int32_t;
        using wtype = std::int64_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { -2147483638, 2147483637, 10, 9, 8, 7, 6, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 2147483658, 2147483637, 10, 9, 8, 7, 6, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { -2147483637, 2147483638, 11, 10, 9, 8, 7, 6 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Absolute") {
            auto res = abs(v);
            INFO(std::format("abs(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], int(std::abs(v[i]))));
                REQUIRE(res[i] == static_cast<type>(std::abs(v[i])));
            }
        }

        WHEN("Saturating Absoulute") {
            auto res = sat_abs(v);
            INFO(std::format("sat_abs(v): {}", res));
            wtype ts[] = { 2147483647, 2147483647, 0, 1, 2, 3, 4, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
    GIVEN("Unsigned 32bit Integer") {
        using type = std::uint32_t;
        using wtype = std::uint64_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { 10, 4294967285, 10, 9, 8, 7, 6, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Widening abs difference") {
            auto res = widening_abs_diff(v, Vec<N, type>::load(10));
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wabs_diff(v, 10s): {}", res));
            wtype ts[] = { 10, 4294967285, 10, 9, 8, 7, 6, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            wtype ts[] = { 11, 4294967286, 11, 10, 9, 8, 7, 6 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 64bit Absolute Operations", "[absolute][64bit]" ) {
    GIVEN("Signed 64bit Integer") {
        using type = std::int64_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { -9223372036854775798ll, 9223372036854775797ll, 10, 9, 8, 7, 6,
  5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{}]: {} == {}", i, res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }


        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            type ts[] = { -9223372036854775797ll, 9223372036854775798ll, 11, 10,
  9, 8, 7, 6 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Absolute") {
            auto res = abs(v);
            INFO(std::format("abs(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], int(std::abs(v[i]))));
                REQUIRE(res[i] == static_cast<type>(std::abs(v[i])));
            }
        }

        WHEN("Saturating Absoulute") {
            auto res = sat_abs(v);
            INFO(std::format("sat_abs(v): {}", res));
            type ts[] = { 9223372036854775807ll, 9223372036854775807ll, 0, 1, 2, 3, 4, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", res[i], ts[i]));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
    GIVEN("Unsigned 64bit Integer") {
        using type = std::uint64_t;
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

        WHEN("Abs difference") {
            auto res = abs_diff(v, Vec<N, type>::load(10));
            INFO(std::format("abs_diff(v, 10s): {}", res));
            type ts[] = { 10, 18446744073709551605ull, 10, 9, 8, 7, 6, 5 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }

        WHEN("Accumulating abs difference") {
            auto res = abs_acc_diff(Vec<N, type>::load(1), v, Vec<N, type>::load(10));
            INFO(std::format("abs_acc_diff(1s, v, 10s): {}", res));
            type ts[] = { 11, 18446744073709551606ull, 11, 10, 9, 8, 7, 6 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("{} == {}", int(res[i]), int(ts[i])));
                REQUIRE(res[i] == ts[i]);
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Float16 Absolute Operations", "[absolute][float16]" ) {
    using type = float16;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(float(v[0]), Catch::Matchers::WithinRel(float(min), eps<float>));
        REQUIRE_THAT(float(v[1]), Catch::Matchers::WithinRel(float(max), eps<float>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(float(v[i]), Catch::Matchers::WithinRel(float(i - 2), eps<float>));
        }
    }

    WHEN("Abs difference") {
        auto res = abs_diff(v, Vec<N, type>::load(10));
        INFO(std::format("abs_diff(v, 10s): {}", res));
        float ts[] = { 10.f, 65504.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 1.f, 2.f, 3.f };
        for (auto i = 0ul; i < N; ++i) {
            INFO(std::format("{} == {}", res[i], ts[i]));
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(ts[i], eps<float>));
        }
    }

    WHEN("Absolute") {
        auto res = abs(v);
        INFO(std::format("abs(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto t = float(ui::abs(v[i]));
            INFO(std::format("{} == {}", res[i], t));
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(t, eps<float>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Bfloat16 Absolute Operations", "[absolute][bfloat16]" ) {
    using type = bfloat16;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(float(v[0]), Catch::Matchers::WithinRel(float(min), eps<float>));
        REQUIRE_THAT(float(v[1]), Catch::Matchers::WithinRel(float(max), eps<float>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(float(v[i]), Catch::Matchers::WithinRel(float(i - 2), eps<float>));
        }
    }

    WHEN("Abs difference") {
        auto res = abs_diff(v, Vec<N, type>::load(10));
        INFO(std::format("abs_diff(v, 10s): {}", res));
        float ts[] = { 10.f, float(max), 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 1.f, 2.f, 3.f };
        for (auto i = 0ul; i < N; ++i) {
            INFO(std::format("{} == {}", res[i], ts[i]));
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(ts[i], eps<float>));
        }
    }

    WHEN("Absolute") {
        auto res = abs(v);
        INFO(std::format("abs(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto t = float(ui::abs(v[i]));
            INFO(std::format("{} == {}", res[i], t));
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(t, eps<float>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " float32 Absolute Operations", "[absolute][float32]" ) {
    using type = float;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[0], Catch::Matchers::WithinRel(min, eps<type>));
        REQUIRE_THAT(v[1], Catch::Matchers::WithinRel(max, eps<type>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(v[i], Catch::Matchers::WithinRel(type(i - 2), eps<type>));
        }
    }

    WHEN("Abs difference") {
        auto res = abs_diff(v, Vec<N, type>::load(10));
        INFO(std::format("abs_diff(v, 10s): {}", res));
        type ts[] = { 10.f, max, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 1.f, 2.f, 3.f };
        for (auto i = 0ul; i < N; ++i) {
            INFO(std::format("{} == {}", res[i], ts[i]));
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(ts[i], eps<type>));
        }
    }

    WHEN("Absolute") {
        auto res = abs(v);
        INFO(std::format("abs(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto t = std::abs(v[i]);
            INFO(std::format("{} == {}", res[i], t));
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(t, eps<type>));
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " float64 Absolute Operations", "[absolute][float64]" ) {
    using type = double;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[0], Catch::Matchers::WithinRel(min, eps<type>));
        REQUIRE_THAT(v[1], Catch::Matchers::WithinRel(max, eps<type>));
        for (auto i = 2u; i < N; ++i) {
            REQUIRE_THAT(v[i], Catch::Matchers::WithinRel(type(i - 2), eps<type>));
        }
    }

    WHEN("Abs difference") {
        auto res = abs_diff(v, Vec<N, type>::load(10));
        INFO(std::format("abs_diff(v, 10s): {}", res));
        type ts[] = { 10.f, max, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f, 1.f, 2.f, 3.f };
        for (auto i = 0ul; i < N; ++i) {
            INFO(std::format("{} == {}", res[i], ts[i]));
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(ts[i], eps<type>));
        }
    }

    WHEN("Absolute") {
        auto res = abs(v);
        INFO(std::format("abs(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto t = std::abs(v[i]);
            INFO(std::format("{} == {}", res[i], t));
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(t, eps<type>));
        }
    }
}





