
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
    std::uint32_t
#ifdef UI_HAS_INT128
    , std::uint64_t
#endif
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
    VEC_ARCH_NAME " Reciprocal Operations",
    "[reciprocal][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using wtype = Fixture<TestType>::wtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Reciprocal estimate") {
        auto res = reciprocal_estimate(v);
        auto t = emul::reciprocal_estimate(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("restimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(res[i], RelativeErr(t[i], 0u));
        }
    }

    WHEN("Reciprocal refine") {
        auto t = reciprocal_estimate(v);
        auto res = reciprocal_refine(v, t); 
        INFO(std::format("t: {}", t));
        INFO(std::format("rrefine(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE_THAT(res[i], RelativeErr(t[i], 0u));
        }
    }

    WHEN("Sqrt inv estimate") {
        auto res = sqrt_inv_estimate(v);
        auto t = emul::sqrt_inv_estimate(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("restimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto l = res[i];
            auto r = t[i];
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, RelativeErr(r, 0u));
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
    VEC_ARCH_NAME " Reciprocal Operations",
    "[reciprocal][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<type>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Reciprocal estimate") {
        auto res = reciprocal_estimate(v);
        auto t = emul::reciprocal_estimate(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("restimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, ftype(0.11)));
        }
    }

    WHEN("Reciprocal refine") {
        auto re = reciprocal_estimate(v);
        auto te = emul::reciprocal_estimate(v);
        auto res = reciprocal_refine(v, re);
        auto t = emul::reciprocal_refine(v, te);
        INFO(std::format("te: {}\n", te));
        INFO(std::format("re: {}\n", re));
        INFO(std::format("t: {}\n", t));
        INFO(std::format("rrefine(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, ftype(0.15)));
        }
    }

    WHEN("Inverse sqrt estimate") {
        auto res = sqrt_inv_estimate(v);
        auto t = emul::sqrt_inv_estimate(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("sqrt_estimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, ftype(0.11)));
        }
    }

    // FIXME: Find a good way to test this. Hardware saturates the output.
    /*WHEN("Inverse sqrt refine") {*/
    /*    auto re = sqrt_inv_estimate(v);*/
    /*    auto te = emul::sqrt_inv_estimate(v);*/
    /*    auto res = sqrt_inv_refine(v, re);*/
    /*    auto t = emul::sqrt_inv_refine(v, te);*/
    /*    INFO(std::format("te: {}\n", te));*/
    /*    INFO(std::format("re: {}\n", re));*/
    /*    INFO(std::format("t: {}\n", t));*/
    /*    INFO(std::format("sqrt_refine(v): {}", res));*/
    /*    for (auto i = 0ul; i < N; ++i) {*/
    /*        auto fp = ui::fp::decompose_fp(t[i]);*/
    /*        ftype l = ftype(res[i]);*/
    /*        ftype r = ftype(t[i]);*/
    /*        INFO(std::format("[{}]: {} == {}", i, l, r));*/
    /*        REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, ftype(0.15)));*/
    /*    }*/
    /*}*/


    WHEN("Exponent reciprocal estimate") {
        auto res = exponent_reciprocal_estimate(v);
        auto t = emul::exponent_reciprocal_estimate(v);
        INFO(std::format("t: {}", t));
        INFO(std::format("exp_r_estimate(v): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(t[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, ftype(0.5)));
        }
    }
}
