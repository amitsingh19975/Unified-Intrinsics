#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <print>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

TEST_CASE( VEC_ARCH_NAME " 8bit Addition", "[addition][8bit]" ) {
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -69);
            REQUIRE(res[1] ==  59);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[31] == 25);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -69);
            REQUIRE(res[1] ==  59);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[31] == 25);
        }

        WHEN("High Narrowing Addition") {
            /*auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);*/
            /*auto res = high_narrowing_add(v, bs);*/
            // Narrowing it not implemented
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[31] == 58);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[31] == 58);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == 29);
            REQUIRE(res[ 9] == 33);
            REQUIRE(res[10] == 37);
            REQUIRE(res[11] == 41);
            REQUIRE(res[12] == 45);
            REQUIRE(res[13] == 49);
            REQUIRE(res[14] == 53);
            REQUIRE(res[15] == 57);
            REQUIRE(res[16] == -1);
            REQUIRE(res[17] ==  1);
            REQUIRE(res[18] ==  5);
            REQUIRE(res[19] ==  9);
            REQUIRE(res[20] == 13);
            REQUIRE(res[21] == 17);
            REQUIRE(res[22] == 21);
            REQUIRE(res[23] == 25);
            REQUIRE(res[24] == 29);
            REQUIRE(res[25] == 33);
            REQUIRE(res[26] == 37);
            REQUIRE(res[27] == 41);
            REQUIRE(res[28] == 45);
            REQUIRE(res[29] == 49);
            REQUIRE(res[30] == 53);
            REQUIRE(res[31] == 57);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == -78);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == 29);
            REQUIRE(res[ 9] == 33);
            REQUIRE(res[10] == 37);
            REQUIRE(res[11] == 41);
            REQUIRE(res[12] == 45);
            REQUIRE(res[13] == 49);
            REQUIRE(res[14] == 53);
            REQUIRE(res[15] == 57);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] ==  0);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
            REQUIRE(res[ 8] == 30);
            REQUIRE(res[ 9] == 34);
            REQUIRE(res[10] == 38);
            REQUIRE(res[11] == 42);
            REQUIRE(res[12] == 46);
            REQUIRE(res[13] == 50);
            REQUIRE(res[14] == 54);
            REQUIRE(res[15] == 58);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == -78);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 434);
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] ==  123);
            REQUIRE(res[1] ==  251);
            REQUIRE(res[2] ==  124);
            REQUIRE(res[3] ==  125);
            REQUIRE(res[4] ==  126);
            REQUIRE(res[5] ==  127);
            REQUIRE(res[6] ==  128);
            REQUIRE(res[7] ==  129);
            REQUIRE(res[8] ==  130);
            REQUIRE(res[9] ==  131);
            REQUIRE(res[31] == 25);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] ==  123);
            REQUIRE(res[1] ==  251);
            REQUIRE(res[2] ==  124);
            REQUIRE(res[3] ==  125);
            REQUIRE(res[4] ==  126);
            REQUIRE(res[5] ==  127);
            REQUIRE(res[6] ==  128);
            REQUIRE(res[7] ==  129);
            REQUIRE(res[8] ==  130);
            REQUIRE(res[9] ==  131);
            REQUIRE(res[31] == 25);
        }

        WHEN("High Narrowing Addition") {
            /*auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);*/
            /*auto res = high_narrowing_add(v, bs);*/
            // Narrowing it not implemented
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[31] == 58);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[31] == 58);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == 255);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == 29);
            REQUIRE(res[ 9] == 33);
            REQUIRE(res[10] == 37);
            REQUIRE(res[11] == 41);
            REQUIRE(res[12] == 45);
            REQUIRE(res[13] == 49);
            REQUIRE(res[14] == 53);
            REQUIRE(res[15] == 57);
            REQUIRE(res[16] == 255);
            REQUIRE(res[17] ==  1);
            REQUIRE(res[18] ==  5);
            REQUIRE(res[19] ==  9);
            REQUIRE(res[20] == 13);
            REQUIRE(res[21] == 17);
            REQUIRE(res[22] == 21);
            REQUIRE(res[23] == 25);
            REQUIRE(res[24] == 29);
            REQUIRE(res[25] == 33);
            REQUIRE(res[26] == 37);
            REQUIRE(res[27] == 41);
            REQUIRE(res[28] == 45);
            REQUIRE(res[29] == 49);
            REQUIRE(res[30] == 53);
            REQUIRE(res[31] == 57);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 178);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 255);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == 29);
            REQUIRE(res[ 9] == 33);
            REQUIRE(res[10] == 37);
            REQUIRE(res[11] == 41);
            REQUIRE(res[12] == 45);
            REQUIRE(res[13] == 49);
            REQUIRE(res[14] == 53);
            REQUIRE(res[15] == 57);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 256);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
            REQUIRE(res[ 8] == 30);
            REQUIRE(res[ 9] == 34);
            REQUIRE(res[10] == 38);
            REQUIRE(res[11] == 42);
            REQUIRE(res[12] == 46);
            REQUIRE(res[13] == 50);
            REQUIRE(res[14] == 54);
            REQUIRE(res[15] == 58);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 178);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 690);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 16bit Addition", "[addition][16bit]" ) {
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -16389);
            REQUIRE(res[1] ==  16379);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[15] == 9);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -16389);
            REQUIRE(res[1] ==  16379);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[15] == 9);
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));
            REQUIRE(res[ 0] == 127);
            REQUIRE(res[ 1] == 127);
            REQUIRE(res[ 2] == -1);
            REQUIRE(res[ 3] == -1);
            REQUIRE(res[ 4] == -1);
            REQUIRE(res[ 5] == -1);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == -1);
            REQUIRE(res[ 9] ==  1);
            REQUIRE(res[10] ==  5);
            REQUIRE(res[11] ==  9);
            REQUIRE(res[12] == 13);
            REQUIRE(res[13] == 17);
            REQUIRE(res[14] == 21);
            REQUIRE(res[15] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] ==  0);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 90);
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] ==  32763);
            REQUIRE(res[1] ==  65531);
            REQUIRE(res[2] ==  32764);
            REQUIRE(res[3] ==  32765);
            REQUIRE(res[4] ==  32766);
            REQUIRE(res[5] ==  32767);
            REQUIRE(res[6] ==  32768);
            REQUIRE(res[7] ==  32769);
            REQUIRE(res[8] ==  32770);
            REQUIRE(res[9] ==  32771);
            REQUIRE(res[15] == 9);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] ==  32763);
            REQUIRE(res[1] ==  65531);
            REQUIRE(res[2] ==  32764);
            REQUIRE(res[3] ==  32765);
            REQUIRE(res[4] ==  32766);
            REQUIRE(res[5] ==  32767);
            REQUIRE(res[6] ==  32768);
            REQUIRE(res[7] ==  32769);
            REQUIRE(res[8] ==  32770);
            REQUIRE(res[9] ==  32771);
            REQUIRE(res[15] == 9);
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));
            REQUIRE(res[ 0] == 255);
            REQUIRE(res[ 1] == 255);
            REQUIRE(res[ 2] == 255);
            REQUIRE(res[ 3] == 255);
            REQUIRE(res[ 4] == 255);
            REQUIRE(res[ 5] == 255);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == 65535);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 65535);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 65536);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 65626);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 32bit Addition", "[addition][32bit]" ) {
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -1073741829);
            REQUIRE(res[1] ==  1073741819);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[15] == 9);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] == -1073741829);
            REQUIRE(res[1] ==  1073741819);
            REQUIRE(res[2] == -4);
            REQUIRE(res[3] == -3);
            REQUIRE(res[4] == -2);
            REQUIRE(res[5] == -1);
            REQUIRE(res[6] ==  0);
            REQUIRE(res[7] ==  1);
            REQUIRE(res[8] ==  2);
            REQUIRE(res[9] ==  3);
            REQUIRE(res[15] == 9);
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));
            REQUIRE(res[ 0] == 32767);
            REQUIRE(res[ 1] == 32767);
            REQUIRE(res[ 2] == -1);
            REQUIRE(res[ 3] == -1);
            REQUIRE(res[ 4] == -1);
            REQUIRE(res[ 5] == -1);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == -1);
            REQUIRE(res[ 9] ==  1);
            REQUIRE(res[10] ==  5);
            REQUIRE(res[11] ==  9);
            REQUIRE(res[12] == 13);
            REQUIRE(res[13] == 17);
            REQUIRE(res[14] == 21);
            REQUIRE(res[15] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] ==  0);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 90);
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("Widening Addition") {
            auto res = widening_add(v, v); 
            INFO(std::format("widening_add(v, v): {}", res));
            using wtype = ui::internal::widening_result_t<type>;
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<wtype>(v[i]) + static_cast<wtype>(v[i]));
            }
        }

        WHEN("Halving Addition without rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));

            REQUIRE(res[0] ==  2147483643);
            REQUIRE(res[1] ==  4294967291);
            REQUIRE(res[2] ==  2147483644);
            REQUIRE(res[3] ==  2147483645);
            REQUIRE(res[4] ==  2147483646);
            REQUIRE(res[5] ==  2147483647);
            REQUIRE(res[6] ==  2147483648);
            REQUIRE(res[7] ==  2147483649);
            REQUIRE(res[8] ==  2147483650);
            REQUIRE(res[9] ==  2147483651);
            REQUIRE(res[15] == 9);
        }

        WHEN("Halving Addition with rounding") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = halving_add<true>(v, bs);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("halving_add(v, bs): {}", res));
            REQUIRE(res[0] ==  2147483643);
            REQUIRE(res[1] ==  4294967291);
            REQUIRE(res[2] ==  2147483644);
            REQUIRE(res[3] ==  2147483645);
            REQUIRE(res[4] ==  2147483646);
            REQUIRE(res[5] ==  2147483647);
            REQUIRE(res[6] ==  2147483648);
            REQUIRE(res[7] ==  2147483649);
            REQUIRE(res[8] ==  2147483650);
            REQUIRE(res[9] ==  2147483651);
            REQUIRE(res[15] == 9);
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));

            REQUIRE(res[ 0] == 65535);
            REQUIRE(res[ 1] == 65535);
            REQUIRE(res[ 2] == 65535);
            REQUIRE(res[ 3] == 65535);
            REQUIRE(res[ 4] == 65535);
            REQUIRE(res[ 5] == 65535);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == 4294967295);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening pairwise Addition") {
            auto res = widening_padd(v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 4294967295);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Accumulating widening pairwise Addition") {
            auto res = widening_padd(Vec<N/2, wtype>::load(1), v);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, wtype>);
            INFO(std::format("wpadd(v): {}", res));
            REQUIRE(res[ 0] == 4294967296);
            REQUIRE(res[ 1] ==  2);
            REQUIRE(res[ 2] ==  6);
            REQUIRE(res[ 3] == 10);
            REQUIRE(res[ 4] == 14);
            REQUIRE(res[ 5] == 18);
            REQUIRE(res[ 6] == 22);
            REQUIRE(res[ 7] == 26);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Widening fold") {
            auto res = widening_fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", res));
            STATIC_REQUIRE(std::same_as<decltype(res), wtype>);
            REQUIRE(res == 4294967386);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 64bit Addition", "[addition][64bit]" ) {
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));
            REQUIRE(res[ 0] == 2147483647);
            REQUIRE(res[ 1] == 2147483647);
            REQUIRE(res[ 2] == -1);
            REQUIRE(res[ 3] == -1);
            REQUIRE(res[ 4] == -1);
            REQUIRE(res[ 5] == -1);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == -1);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
            REQUIRE(res[ 8] == -1);
            REQUIRE(res[ 9] ==  1);
            REQUIRE(res[10] ==  5);
            REQUIRE(res[11] ==  9);
            REQUIRE(res[12] == 13);
            REQUIRE(res[13] == 17);
            REQUIRE(res[14] == 21);
            REQUIRE(res[15] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
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

        WHEN("Normal Addition") {
            auto res = add(v, v); 
            INFO(std::format("add(v, v): {}", res));

            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == static_cast<type>(v[i] + v[i]));
            }
        }

        WHEN("High Narrowing Addition") {
            auto bs = DataGenerator<N, type>::cyclic_make(-10, 100);
            auto res = high_narrowing_add(v, bs);
            STATIC_REQUIRE(std::same_as<decltype(res)::element_t, ntype>);
            INFO(std::format("bs: {}", bs));
            INFO(std::format("high_narrowing_add(v, bs): {}", res));

            REQUIRE(res[ 0] == 4294967295);
            REQUIRE(res[ 1] == 4294967295);
            REQUIRE(res[ 2] == 4294967295);
            REQUIRE(res[ 3] == 4294967295);
            REQUIRE(res[ 4] == 4294967295);
            REQUIRE(res[ 5] == 4294967295);
            REQUIRE(res[ 6] == 0);
            REQUIRE(res[ 7] == 0);
            REQUIRE(res[ 8] == 0);
            REQUIRE(res[ 9] == 0);
            REQUIRE(res[15] == 0);
        }

        WHEN("Saturating Add") {
            auto res = sat_add(v, v);
            INFO(std::format("sat_add(v, v): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Saturating Add with unsigned") {
            auto bs = DataGenerator<N, otype>::make();
            auto res = sat_add(v, bs);
            INFO(std::format("sat_add(v, bs): {}", res));
            REQUIRE(res[ 0] == min);
            REQUIRE(res[ 1] == max);
            REQUIRE(res[ 2] == 0);
            REQUIRE(res[ 3] == 2);
            REQUIRE(res[ 4] == 4);
            REQUIRE(res[ 5] == 6);
            REQUIRE(res[ 6] == 8);
            REQUIRE(res[ 7] == 10);
            REQUIRE(res[ 8] == 12);
            REQUIRE(res[ 9] == 14);
            REQUIRE(res[15] == 26);
        }

        WHEN("Pairwise Addition") {
            auto res = padd(v, v);
            INFO(std::format("padd(v, v): {}", res));
            REQUIRE(res[ 0] == 18446744073709551615ull);
            REQUIRE(res[ 1] ==  1);
            REQUIRE(res[ 2] ==  5);
            REQUIRE(res[ 3] ==  9);
            REQUIRE(res[ 4] == 13);
            REQUIRE(res[ 5] == 17);
            REQUIRE(res[ 6] == 21);
            REQUIRE(res[ 7] == 25);
        }

        WHEN("Pairwise fold") {
            auto res = fold(v, op::padd_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }

        WHEN("Fold") {
            auto res = fold(v, op::add_t{});
            INFO(std::format("fold(v): {}", int64_t(res)));
            REQUIRE(res == 90);
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " Float16 Addition", "[addition][float16]" ) {
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
            auto lhs = float(v[i]);
            auto rhs = float(i - 2); 
            INFO(std::format("{} == {}", lhs, rhs));
            REQUIRE_THAT(lhs, Catch::Matchers::WithinRel(rhs, eps<float>));
        }
    }

    WHEN("Normal Addition") {
        auto res = add(v, v); 
        INFO(std::format("add(v, v): {}", res));

        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(float(v[i] * 2), eps<float>));
        }
    }

    WHEN("Pairwise Addition") {
        auto res = padd(v, v);
        INFO(std::format("padd(v, v): {}", res));
        
        REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(float(65504), eps<float>));
        REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(float(1), eps<float>));
        REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(float(5), eps<float>));
        REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(float(9), eps<float>));
        REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(float(13), eps<float>));
        REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(float(17), eps<float>));
        REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(float(21), eps<float>));
        REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(float(25), eps<float>));
        REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(float(65504), eps<float>));
        REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(float(1), eps<float>));
        REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(float(5), eps<float>));
        REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(float(9), eps<float>));
        REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(float(13), eps<float>));
        REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(float(17), eps<float>));
        REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(float(21), eps<float>));
        REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(float(25), eps<float>));
    }

    WHEN("Pairwise fold") {
        auto res = fold(v, op::padd_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(INFINITY, eps<float>));
    }

    WHEN("Fold") {
        auto res = fold(v, op::add_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(INFINITY, eps<float>));
    }
}

TEST_CASE( VEC_ARCH_NAME " Bfloat16 Addition", "[addition][bfloat16]" ) {
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
            auto lhs = float(v[i]);
            auto rhs = float(i - 2); 
            INFO(std::format("{} == {}", lhs, rhs));
            REQUIRE_THAT(lhs, Catch::Matchers::WithinRel(rhs, eps<float>));
        }
    }

    WHEN("Normal Addition") {
        auto res = add(v, v); 
        INFO(std::format("add(v, v): {}", res));

        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(float(v[i] * 2), eps<float>));
        }
    }

    WHEN("Pairwise Addition") {
        auto res = padd(v, v);
        INFO(std::format("padd(v, v): {}", res));
        
        REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(float(max), eps<float>));
        REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(float(1), eps<float>));
        REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(float(5), eps<float>));
        REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(float(9), eps<float>));
        REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(float(13), eps<float>));
        REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(float(17), eps<float>));
        REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(float(21), eps<float>));
        REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(float(25), eps<float>));
        REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(float(max), eps<float>));
        REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(float(1), eps<float>));
        REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(float(5), eps<float>));
        REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(float(9), eps<float>));
        REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(float(13), eps<float>));
        REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(float(17), eps<float>));
        REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(float(21), eps<float>));
        REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(float(25), eps<float>));
    }

    WHEN("Pairwise fold") {
        auto res = fold(v, op::padd_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(float(max), eps<float>));
    }

    WHEN("Fold") {
        auto res = fold(v, op::add_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(float(max), eps<float>));
    }
}

TEST_CASE( VEC_ARCH_NAME " float32 Addition", "[addition][float32]" ) {
    using type = float;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[0], Catch::Matchers::WithinRel(min, eps<float>));
        REQUIRE_THAT(v[1], Catch::Matchers::WithinRel(max, eps<float>));
        for (auto i = 2u; i < N; ++i) {
            auto lhs = float(v[i]);
            auto rhs = float(i - 2); 
            INFO(std::format("{} == {}", lhs, rhs));
            REQUIRE_THAT(lhs, Catch::Matchers::WithinRel(rhs, eps<float>));
        }
    }

    WHEN("Normal Addition") {
        auto res = add(v, v); 
        INFO(std::format("add(v, v): {}", res));

        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(float(res[i]), Catch::Matchers::WithinRel(float(v[i] * 2), eps<float>));
        }
    }

    WHEN("Pairwise Addition") {
        auto res = padd(v, v);
        INFO(std::format("padd(v, v): {}", res));
        
        REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(max, eps<float>));
        REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(1, eps<float>));
        REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(5, eps<float>));
        REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(9, eps<float>));
        REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(13, eps<float>));
        REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(17, eps<float>));
        REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(21, eps<float>));
        REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(25, eps<float>));
        REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(max, eps<float>));
        REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(1, eps<float>));
        REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(5, eps<float>));
        REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(9, eps<float>));
        REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(13, eps<float>));
        REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(17, eps<float>));
        REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(21, eps<float>));
        REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(25, eps<float>));
    }

    WHEN("Pairwise fold") {
        auto res = fold(v, op::padd_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(max, eps<float>));
    }

    WHEN("Fold") {
        auto res = fold(v, op::add_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(float(res), Catch::Matchers::WithinRel(max, eps<float>));
    }
}

TEST_CASE( VEC_ARCH_NAME " float64 Addition", "[addition][float64]" ) {
    using type = double;
    static constexpr auto N = 16ul;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    THEN("Elements are correct") {
        REQUIRE_THAT(v[0], Catch::Matchers::WithinRel(min, eps<double>));
        REQUIRE_THAT(v[1], Catch::Matchers::WithinRel(max, eps<double>));
        for (auto i = 2u; i < N; ++i) {
            auto lhs = v[i];
            auto rhs = double(i - 2); 
            INFO(std::format("{} == {}", lhs, rhs));
            REQUIRE_THAT(lhs, Catch::Matchers::WithinRel(rhs, eps<double>));
        }
    }

    WHEN("Normal Addition") {
        auto res = add(v, v); 
        INFO(std::format("add(v, v): {}", res));

        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(res[i], Catch::Matchers::WithinRel(v[i] * 2, eps<double>));
        }
    }

    WHEN("Pairwise Addition") {
        auto res = padd(v, v);
        INFO(std::format("padd(v, v): {}", res));
        
        REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(max, eps<double>));
        REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(1,   eps<double>));
        REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(5,   eps<double>));
        REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(9,   eps<double>));
        REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(13,  eps<double>));
        REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(17,  eps<double>));
        REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(21,  eps<double>));
        REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(25,  eps<double>));
        REQUIRE_THAT(res[ 8], Catch::Matchers::WithinRel(max, eps<double>));
        REQUIRE_THAT(res[ 9], Catch::Matchers::WithinRel(1,   eps<double>));
        REQUIRE_THAT(res[10], Catch::Matchers::WithinRel(5,   eps<double>));
        REQUIRE_THAT(res[11], Catch::Matchers::WithinRel(9,   eps<double>));
        REQUIRE_THAT(res[12], Catch::Matchers::WithinRel(13,  eps<double>));
        REQUIRE_THAT(res[13], Catch::Matchers::WithinRel(17,  eps<double>));
        REQUIRE_THAT(res[14], Catch::Matchers::WithinRel(21,  eps<double>));
        REQUIRE_THAT(res[15], Catch::Matchers::WithinRel(25,  eps<double>));
    }

    WHEN("Pairwise fold") {
        auto res = fold(v, op::padd_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(res, Catch::Matchers::WithinRel(max, eps<double>));
    }

    WHEN("Fold") {
        auto res = fold(v, op::add_t{});
        INFO(std::format("fold(v): {}", res));
        REQUIRE_THAT(res, Catch::Matchers::WithinRel(max, eps<double>));
    }
}


