
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <cstdint>
#include <format>
#include <print>
#include <type_traits>
#include <vector>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

template <typename T>
struct Fixture{
    using type = T;
    using wtype = ui::internal::widening_result_t<T>;
    static constexpr std::size_t N = 32ul / std::max<unsigned>(sizeof(T) / 2, 1);

};

using SignedTypes = std::tuple<
    std::uint8_t,
    std::uint16_t,
    std::uint32_t,
    std::uint64_t
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    Fixture,
    VEC_ARCH_NAME " Load",
    "[load][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using wtype = Fixture<TestType>::wtype;
    static constexpr auto N = Fixture<TestType>::N;

    std::vector<type> data(100);
    DataGenerator<N, type>::random(data.data(), data.size());

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Broadcasting a scalar value") {
        auto res = load<N>(type(1));
        INFO(std::format("load(1): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE(res[i] == 1);
        }
    }

    WHEN("Broadcasting a scalar value from a given lane") {
        auto v = DataGenerator<N, type>::make();
        INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));
        {
            auto res = load<N, /*Lane=*/0>(v);
            INFO(std::format("load(1): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == v[0]);
            }
        }
        {
            auto res = load<N, /*Lane=*/N-1>(v);
            INFO(std::format("load(1): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == v[N-1]);
            }
        }
    }

    WHEN("Strided load with stride 2") {
        Vec<N, type> a, b;
        strided_load(data.data(), a, b);

        INFO(std::format("sload2(d, a, b) =>\na: {}\nb: {}", a, b));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE(a[i] == data[2 * i + 0]);
            REQUIRE(b[i] == data[2 * i + 1]);
        }
    }

    WHEN("Strided load with stride 3") {
        Vec<N, type> a, b, c;
        strided_load(data.data(), a, b, c);

        INFO(std::format("sload3(d, a, b, c) =>\na: {}\nb: {}\nc: {}", a, b, c));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE(a[i] == data[3 * i + 0]);
            REQUIRE(b[i] == data[3 * i + 1]);
            REQUIRE(c[i] == data[3 * i + 2]);
        }
    }

    WHEN("Strided load with stride 4") {
        Vec<N, type> a, b, c, d;
        strided_load(data.data(), a, b, c, d);

        INFO(std::format("sload4(d, a, b, c, d) =>\na: {}\nb: {}\nc: {}\nd: {}", a, b, c, d));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE(a[i] == data[4 * i + 0]);
            REQUIRE(b[i] == data[4 * i + 1]);
            REQUIRE(c[i] == data[4 * i + 2]);
            REQUIRE(d[i] == data[4 * i + 3]);
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
    static constexpr std::size_t N = std::max(32ul / std::max<unsigned>(sizeof(T) / 2, 1), 8ul);

};

using FTypes = std::tuple<
    float16,
    bfloat16,
    float,
    double
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    FloatFixture,
    VEC_ARCH_NAME " Sqrt",
    "[load][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;

    std::vector<type> data(100);
    DataGenerator<N, type>::random(data.data(), data.size());

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Broadcasting a scalar value") {
        auto res = load<N>(type(1));
        INFO(std::format("load(1): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto l = ftype(res[i]);
            auto r = ftype(1);
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Broadcasting a scalar value from a given lane") {
        auto v = DataGenerator<N, type>::make();
        INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));
        {
            auto res = load<N, /*Lane=*/0>(v);
            INFO(std::format("load(1): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = ftype(res[i]);
                auto r = ftype(v[0]);
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
        {
            auto res = load<N, /*Lane=*/N-1>(v);
            INFO(std::format("load(1): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto l = ftype(res[i]);
                auto r = ftype(v[N-1]);
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
    }

    WHEN("Strided load with stride 2") {
        Vec<N, type> a, b;
        strided_load(data.data(), a, b);

        INFO(std::format("sload2(d, a, b) =>\na: {}\nb: {}", a, b));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE_THAT(ftype(a[i]), Catch::Matchers::WithinRel(ftype(data[2 * i + 0]), eps<ftype>));
            REQUIRE_THAT(ftype(b[i]), Catch::Matchers::WithinRel(ftype(data[2 * i + 1]), eps<ftype>));
        }
    }

    WHEN("Strided load with stride 3") {
        Vec<N, type> a, b, c;
        strided_load(data.data(), a, b, c);

        INFO(std::format("sload3(d, a, b, c) =>\na: {}\nb: {}\nc: {}", a, b, c));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE_THAT(ftype(a[i]), Catch::Matchers::WithinRel(ftype(data[3 * i + 0]), eps<ftype>));
            REQUIRE_THAT(ftype(b[i]), Catch::Matchers::WithinRel(ftype(data[3 * i + 1]), eps<ftype>));
            REQUIRE_THAT(ftype(c[i]), Catch::Matchers::WithinRel(ftype(data[3 * i + 2]), eps<ftype>));
        }
    }

    WHEN("Strided load with stride 4") {
        Vec<N, type> a, b, c, d;
        strided_load(data.data(), a, b, c, d);

        INFO(std::format("sload4(d, a, b, c, d) =>\na: {}\nb: {}\nc: {}\nd: {}", a, b, c, d));
        for (auto i = 0ul; i < N; i += 2) {
            REQUIRE_THAT(ftype(a[i]), Catch::Matchers::WithinRel(ftype(data[4 * i + 0]), eps<ftype>));
            REQUIRE_THAT(ftype(b[i]), Catch::Matchers::WithinRel(ftype(data[4 * i + 1]), eps<ftype>));
            REQUIRE_THAT(ftype(c[i]), Catch::Matchers::WithinRel(ftype(data[4 * i + 2]), eps<ftype>));
            REQUIRE_THAT(ftype(d[i]), Catch::Matchers::WithinRel(ftype(data[4 * i + 3]), eps<ftype>));
        }
    }
}
