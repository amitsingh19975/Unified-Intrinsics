
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <cstdint>
#include <format>
#include <print>
#include <type_traits>
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

template<std::integral V, std::integral E>
    requires (!std::is_signed_v<E> && sizeof(E) < 8)
struct RelativeMatcher : Catch::Matchers::MatcherGenericBase {
    RelativeMatcher(V const& v, E const& e)
        : val(v)
        , err(static_cast<std::int64_t>(e))
    {}

    bool match(V const& other) const {
        auto t = std::int64_t(other - val);
        t = t < 0 ? -t : t;
        return t <= err;
    }

    std::string describe() const override {
        return std::format("Relative({}): {}", err, val);
    }

    V val;
    std::int64_t err;
};

template<std::integral V, std::integral E>
    requires (!std::is_signed_v<E> && sizeof(E) < 8)
auto RelativeErr(V val, E err) -> RelativeMatcher<V, E> {
    return RelativeMatcher(val, err);
}

TEMPLATE_LIST_TEST_CASE_METHOD(
    Fixture,
    VEC_ARCH_NAME " Sqrt",
    "[sqrt][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using wtype = Fixture<TestType>::wtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Sqrt") {
        auto res = sqrt(v);
        INFO(std::format("sqrt(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto t = static_cast<type>(std::sqrt(v[i]));
            REQUIRE_THAT(res[i], RelativeErr(t, 1u)); // sqrt could be rounded up
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
    "[sqrt][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Sqrt") {
        auto res = sqrt(v);
        INFO(std::format("restimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(std::sqrt(ftype(v[i])));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }
}
