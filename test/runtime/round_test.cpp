
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <format>
#include <limits>
#include <print>
#include <type_traits>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

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
    VEC_ARCH_NAME " Rounding Operations",
    "[rounding]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N =FloatFixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Round toward zero") {
        auto res = round<std::float_round_style::round_toward_zero>(v);
        auto t = emul::round<std::float_round_style::round_toward_zero>(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("round(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Round to nearest") {
        auto res = round<std::float_round_style::round_to_nearest>(v);
        auto t = emul::round<std::float_round_style::round_to_nearest>(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("round(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Round toward infinity") {
        auto res = round<std::float_round_style::round_toward_infinity>(v);
        auto t = emul::round<std::float_round_style::round_toward_infinity>(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("round(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Round toward -infinity") {
        auto res = round<std::float_round_style::round_toward_neg_infinity>(v);
        auto t = emul::round<std::float_round_style::round_toward_neg_infinity>(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("round(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }
}
