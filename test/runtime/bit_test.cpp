
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <print>
#include <type_traits>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

TEST_CASE( VEC_ARCH_NAME " 8bit Bit-Manipulation Operations", "[bit][8bit]" ) {
    GIVEN("Signed 8bit Integer") {
        using type = std::int8_t;
        using wtype = std::int16_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 0, 0, 5, 6, 7, 7, 6, 5, 5, 4, 4, 4, 5, 5, 6, 7, 7, 6, 5, 5, 4, 4,
  4, 5, 5, 6, 7, 7, 6, 5, 5, 4 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }

    GIVEN("Signed 8bit Integer") {
        using type = std::uint8_t;
        using wtype = std::uint16_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 7, 7, 5, 6, 7, 7, 6, 5, 5, 4, 4, 4, 5, 5, 6, 7, 7, 6, 5, 5, 4, 4,
  4, 5, 5, 6, 7, 7, 6, 5, 5, 4 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }
        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 16bit Bit-Manipulation Operations", "[bit][16bit]" ) {
    GIVEN("Signed 16bit Integer") {
        using type = std::int16_t;
        using wtype = std::int32_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 0, 0, 13, 14, 15, 15, 14, 13, 13, 12, 12, 12, 13, 13, 14, 15, 15, 14, 13, 13, 12, 12, 12, 13, 13, 14, 15, 15, 14, 13, 13, 12 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }
        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }

    GIVEN("Signed 16bit Integer") {
        using type = std::uint16_t;
        using wtype = std::uint32_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 15, 15, 13, 14, 15, 15, 14, 13, 13, 12, 12, 12, 13, 13, 14, 15, 15, 14, 13, 13, 12, 12, 12, 13, 13, 14, 15, 15, 14, 13, 13, 12 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }
        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 32bit Bit-Manipulation Operations", "[bit][32bit]" ) {
    GIVEN("Signed 32bit Integer") {
        using type = std::int32_t;
        using wtype = std::int64_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 0, 0, 29, 30, 31, 31, 30, 29, 29, 28, 28, 28, 29, 29, 30, 31, 31, 30, 29, 29, 28, 28, 28, 29, 29, 30, 31, 31, 30, 29, 29, 28 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }

    GIVEN("Signed 32bit Integer") {
        using type = std::uint32_t;
        using wtype = std::uint64_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 31, 31, 29, 30, 31, 31, 30, 29, 29, 28, 28, 28, 29, 29, 30, 31, 31, 30, 29, 29, 28, 28, 28, 29, 29, 30, 31, 31, 30, 29, 29, 28 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }
}

TEST_CASE( VEC_ARCH_NAME " 64bit Bit-Manipulation Operations", "[bit][64bit]" ) {
    GIVEN("Signed 64bit Integer") {
        using type = std::int64_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 0, 0, 61, 62, 63, 63, 62, 61, 61, 60, 60, 60, 61, 61, 62, 63, 63, 62, 61, 61, 60, 60, 60, 61, 61, 62, 63, 63, 62, 61, 61, 60 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }

    GIVEN("Signed 64bit Integer") {
        using type = std::uint64_t;
        using utype = std::make_unsigned_t<type>;
        using mtype = mask_inner_t<type>;
        static constexpr auto N = 32ul;
        auto v = DataGenerator<N, type>::cyclic_make(-5, 10);

        INFO("[Vec]: " << std::format("{}", v));

        static constexpr auto min = std::numeric_limits<type>::min();
        static constexpr auto max = std::numeric_limits<type>::max();
        v[0] = min;
        v[1] = max;

        WHEN("Count leading sign bits") {
            auto res = count_leading_sign_bits(v);
            INFO(std::format("sign(v): {}", res));
            unsigned cs[] = { 63, 63, 61, 62, 63, 63, 62, 61, 61, 60, 60, 60, 61, 61, 62, 63, 63, 62, 61, 61, 60, 60, 60, 61, 61, 62, 63, 63, 62, 61, 61, 60 };
            for (auto i = 0ul; i < N; ++i) {
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], cs[i]));
                REQUIRE(res[i] == cs[i]);
            }
        }

        WHEN("Count leading zeros") {
            auto res = count_leading_zeros(v);
            INFO(std::format("lcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::countl_zero(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Popcount") {
            auto res = popcount(v);
            INFO(std::format("popcount(v): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = std::popcount(static_cast<utype>(v[i]));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise clear") {
            auto res = bitwise_clear(v, Vec<N, type>::load(13));
            INFO(std::format("clear(v, 13s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto t = static_cast<type>(v[i] & ~type(13));
                INFO(std::format("[{} => {:08b}]: {} == {}", i, v[i], res[i], t));
                REQUIRE(res[i] == t);
            }
        }

        WHEN("Bitwise Select") {
            // odd numbers will be selected
            auto cond = DataGenerator<N, type>::make_mask([](auto i) -> mtype {
                return i & 1 ? ~mtype(0) : 0;
            });
            auto res = bitwise_select(cond, v, Vec<N, type>::load(100));
            INFO(std::format("select(cond, v, 100s): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                if(i % 2 == 0) {
                    INFO(std::format("Even[{}]: {} == {}", i, int(res[i]), 100));
                    REQUIRE(res[i] == 100);
                } else {
                    INFO(std::format("Odd[{}]: {} == {}", i, int(res[i]), int(v[i])));
                    REQUIRE(res[i] == v[i]);
                }
            }
        }
    }
}

