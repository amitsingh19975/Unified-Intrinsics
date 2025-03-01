
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <print>
#include <cmath>
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
    VEC_ARCH_NAME " Div Operations",
    "[div][signed]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Floor division") {
        auto den = DataGenerator<N, type>::random();
        auto res = div(v, den);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] / den[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Remainder") {
        // negative remainder is undefined
        auto den = sat_abs(DataGenerator<N, type>::random());
        auto num = sat_abs(v);
        auto res = rem(num, den);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = num[i] % den[i];
            INFO(std::format("[{}]: {} % {} == {}", i, int64_t(v[i]), int64_t(den[i]), r));
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
    VEC_ARCH_NAME " Div Operations",
    "[div][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();
    v[0] = 102.32;
    v[1] = 112.32;

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Floor division") {
        auto den = cast<type>(DataGenerator<N, ftype>::random());
        auto res = div(v, den);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] / den[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Remainder") {
        // negative remainder is undefined
        auto num = abs(v);
        auto den = cast<type>(abs(DataGenerator<N, ftype>::random()));
        auto res = rem(num, den);
        for (auto i = 0ul; i < N; ++i) {
            ftype l, r;
            auto n = ftype(num[i]);
            auto d = ftype(den[i]);
            auto t0 = std::floor(n / d);
            t0 = t0 * d;
            if constexpr (std::same_as<type, ftype>) {
                l = res[i];
                r = n - t0;
            } else {
                // there are errors due to precision loss
                l = std::round(float(res[i]));
                r = std::round(n - t0);
            }
            INFO(std::format("[{}]: {} % {} = {}, {}", i, float(n), float(d), r, l));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }
}
