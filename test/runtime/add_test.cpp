#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <print>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

TEST_CASE( VEC_ARCH_NAME " Addition", "addition" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using wtype = std::int16_t;
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
            auto bs = DataGenerator<N, utype>::make();
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
}

