
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <print>
#include <cmath>
#include <type_traits>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

template <typename T>
struct Fixture{
    using type = T;
    static constexpr std::size_t N = 32ul / std::max<unsigned>(sizeof(T) / 2, 1);

};

using SignedTypes = std::tuple<
    std::int8_t,
    std::uint8_t,
    std::int16_t,
    std::uint16_t,
    std::int32_t,
    std::uint32_t,
    std::int64_t,
    std::uint64_t
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    Fixture,
    VEC_ARCH_NAME " Logical Operations",
    "[logical][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    if constexpr (std::is_signed_v<type>) {
        WHEN("Negate") {
            auto res = negate(v);
            for (auto i = 0ul; i < N; ++i) {
                type l = res[i];
                type r = type(-v[i]);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE(l == r);
            }
        }

        WHEN("Saturating Negate") {
            auto res = sat_negate(v);
            for (auto i = 0ul; i < N; ++i) {
                type l = res[i];
                type r = v[i] == min ? max : -v[i];
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE(l == r);
            }
        }
    }

    WHEN("Bitwise not") {
        auto res = bitwise_not(v);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = ~v[i];
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Bitwise and") {
        auto res = bitwise_and(v, Vec<N, type>::load(33));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = (v[i] & 33);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Bitwise or") {
        auto res = bitwise_or(v, Vec<N, type>::load(33));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = (v[i] | 33);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Bitwise xor") {
        auto res = bitwise_xor(v, Vec<N, type>::load(33));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = (v[i] ^ 33);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Bitwise or-not") {
        auto res = bitwise_ornot(v, Vec<N, type>::load(33));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = (v[i] | ~type(33));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Bitwise not-and") {
        auto res = bitwise_notand(v, Vec<N, type>::load(33));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = (~v[i] & type(33));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }
}

template <std::floating_point T>
struct FloatFixture{
    using type = T;
    using ftype = std::conditional_t<
        std::same_as<float16, T> || std::same_as<T, bfloat16>,
        float,
        T
    >;
    using mtype = mask_inner_t<T>;
    static constexpr std::size_t N = 32ul / std::max<unsigned>(sizeof(T) / 2, 1);

};

using FTypes = std::tuple<
    float16,
    bfloat16,
    float,
    double
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    FloatFixture,
    VEC_ARCH_NAME " Logical Operations",
    "[logical][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Negate") {
        auto res = negate(v);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(-v[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }
}
