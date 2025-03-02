
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <print>
#include <type_traits>
#include "ui.hpp"
#include "ui/maths.hpp"
#include "utils.hpp"

using namespace ui;

template <typename T>
struct Fixture{
    using type = T;
    using wtype = std::conditional_t<sizeof(T) == 1, int, T>;
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
    VEC_ARCH_NAME " Integer Manipulation",
    "[manip][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using wtype = Fixture<TestType>::wtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Copy from lane") {
        auto d = DataGenerator<N, type>::random();
        auto res = copy</*ToLane=*/0, /*FromLane=*/1>(v, d);
        REQUIRE(res[0] == d[1]);

        res = copy</*ToLane=*/0, /*FromLane=*/N - 1>(v, d);
        REQUIRE(res[0] == d[N - 1]);
    }

    WHEN("Reverse bits") {
        auto res = reverse_bits(v);
        for (auto i = 0ul; i < N; ++i) {
            auto l = res[i];
            auto r = maths::bit_reverse(v[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Reverse vector") {
        auto res = reverse(v);
        INFO(std::format("reverse(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto l = res[i];
            auto r = v[N - i - 1];
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Zip vector low") {
        auto d = DataGenerator<N, type>::random();
        auto res = zip_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("zip_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            REQUIRE(res[2 * i + 0] == v[i]);
            REQUIRE(res[2 * i + 1] == d[i]);
        }
    }

    WHEN("Zip vector high") {
        auto d = DataGenerator<N, type>::random();
        auto res = zip_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("zip_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            REQUIRE(res[2 * i + 0] == v[N/ 2 + i]);
            REQUIRE(res[2 * i + 1] == d[N/ 2 + i]);
        }
    }

    WHEN("Unzip vector low") {
        auto d = DataGenerator<N, type>::random();
        auto res = unzip_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("unzip_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", i, wtype(res[i]), wtype(v[s])));
            REQUIRE(res[i + 0    ] == v[s]);

            INFO(std::format("[{}]: {} == {}", i, wtype(res[i + N / 2]), wtype(d[s])));
            REQUIRE(res[i + N / 2] == d[s]);
        }
    }

    WHEN("Unzip vector high") {
        auto d = DataGenerator<N, type>::random();
        auto res = unzip_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("unzip_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i + 1;
            INFO(std::format("[{}]: {} == {}", i, wtype(res[i]), wtype(v[s])));
            REQUIRE(res[i + 0    ] == v[s]);

            INFO(std::format("[{}]: {} == {}", i, wtype(res[i + N / 2]), wtype(d[s])));
            REQUIRE(res[i + N / 2] == d[s]);
        }
    }

    WHEN("Transpose low") {
        auto d = DataGenerator<N, type>::random();
        auto res = transpose_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("transpose_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", s, wtype(res[i]), wtype(v[i])));
            REQUIRE(res[s + 0] == v[s]);

            INFO(std::format("[{}]: {} == {}", s + 1, wtype(res[s + i]), wtype(d[i])));
            REQUIRE(res[s + 1] == d[s]);
        }
    }

    WHEN("Transpose high") {
        auto d = DataGenerator<N, type>::random();
        auto res = transpose_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("transpose_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", s, wtype(res[s]), wtype(v[s + 1])));
            REQUIRE(res[s + 0] == v[s + 1]);

            INFO(std::format("[{}]: {} == {}", s + 1, wtype(res[s + i]), wtype(d[s + 1])));
            REQUIRE(res[s + 1] == d[s + 1]);
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
    VEC_ARCH_NAME " Float Manipulation",
    "[manip][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Copy from lane") {
        auto d = cast<type>(DataGenerator<N, ftype>::random());
        auto res = copy</*ToLane=*/0, /*FromLane=*/1>(v, d);
        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(d[1]), eps<ftype>));

        res = copy</*ToLane=*/0, /*FromLane=*/N-1>(v, d);
        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(d[N-1]), eps<ftype>));
    }

    WHEN("Reverse vector") {
        auto res = reverse(v);
        for (auto i = 0ul; i < N; ++i) {
            auto l = res[i];
            auto r = v[N - i - 1];
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Zip vector low") {
        auto d = DataGenerator<N, type>::random();
        auto res = zip_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("zip_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            REQUIRE_THAT(ftype(res[2 * i + 0]), Catch::Matchers::WithinRel(ftype(v[i]), eps<ftype>));
            REQUIRE_THAT(ftype(res[2 * i + 1]), Catch::Matchers::WithinRel(ftype(d[i]), eps<ftype>));
        }
    }

    WHEN("Zip vector high") {
        auto d = DataGenerator<N, type>::random();
        auto res = zip_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("zip_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            REQUIRE_THAT(ftype(res[2 * i + 0]), Catch::Matchers::WithinRel(ftype(v[N / 2 + i]), eps<ftype>));
            REQUIRE_THAT(ftype(res[2 * i + 1]), Catch::Matchers::WithinRel(ftype(d[N / 2 + i]), eps<ftype>));
        }
    }

    WHEN("Unzip vector low") {
        auto d = DataGenerator<N, type>::random();
        auto res = unzip_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("unzip_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", i, ftype(res[i]), ftype(v[s])));
            REQUIRE_THAT(ftype(res[i + 0]), Catch::Matchers::WithinRel(ftype(v[s]), eps<ftype>));

            INFO(std::format("[{}]: {} == {}", i, ftype(res[i + N / 2]), ftype(d[s])));
            REQUIRE_THAT(ftype(res[i + N / 2]), Catch::Matchers::WithinRel(ftype(d[s]), eps<ftype>));
        }
    }

    WHEN("Unzip vector high") {
        auto d = DataGenerator<N, type>::random();
        auto res = unzip_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("unzip_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i + 1;
            INFO(std::format("[{}]: {} == {}", i, ftype(res[i]), ftype(v[s])));
            REQUIRE_THAT(ftype(res[i + 0]), Catch::Matchers::WithinRel(ftype(v[s]), eps<ftype>));

            INFO(std::format("[{}]: {} == {}", i, ftype(res[i + N / 2]), ftype(d[s])));
            REQUIRE_THAT(ftype(res[i + N / 2]), Catch::Matchers::WithinRel(ftype(d[s]), eps<ftype>));
        }
    }

    WHEN("Transpose low") {
        auto d = DataGenerator<N, type>::random();
        auto res = transpose_low(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("transpose_low(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", i, ftype(res[i]), ftype(v[s])));
            REQUIRE_THAT(ftype(res[s + 0]), Catch::Matchers::WithinRel(ftype(v[s]), eps<ftype>));

            INFO(std::format("[{}]: {} == {}", i, ftype(res[s + 1]), ftype(d[s])));
            REQUIRE_THAT(ftype(res[s + 1]), Catch::Matchers::WithinRel(ftype(d[s]), eps<ftype>));
        }
    }

    WHEN("Transpose high") {
        auto d = DataGenerator<N, type>::random();
        auto res = transpose_high(v, d);
        INFO(std::format("d: {}", d));
        INFO(std::format("transpose_high(v, d): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            auto const s = 2 * i;
            INFO(std::format("[{}]: {} == {}", i, ftype(res[i]), ftype(v[s + 1])));
            REQUIRE_THAT(ftype(res[s + 0]), Catch::Matchers::WithinRel(ftype(v[s + 1]), eps<ftype>));

            INFO(std::format("[{}]: {} == {}", i, ftype(res[s + 1]), ftype(d[s + 1])));
            REQUIRE_THAT(ftype(res[s + 1]), Catch::Matchers::WithinRel(ftype(d[s + 1]), eps<ftype>));
        }
    }
}
