
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <numeric>
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
    VEC_ARCH_NAME " Integer Min-Max",
    "[min-max][integer]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Normal max") {
        auto o = DataGenerator<N, type>::random();
        auto res = ui::max(v, o);
        INFO(std::format("max(v, o): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = std::max(v[i], o[i]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Normal min") {
        auto o = DataGenerator<N, type>::random();
        auto res = ui::min(v, o);
        INFO(std::format("min(v, o): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = std::min(v[i], o[i]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Pairwise Max") {
        auto o = DataGenerator<N, type>::random();
        auto res = ui::pmax(v, o);
        INFO(std::format("pmax(v, o): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::max(v[2 * i], v[2 * i + 1]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::max(o[2 * i], o[2 * i + 1]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Pairwise Min") {
        auto o = DataGenerator<N, type>::random();
        auto res = ui::pmin(v, o);
        INFO(std::format("o: {}", o));
        INFO(std::format("pmax(v, o): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::min(v[2 * i], v[2 * i + 1]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::min(o[2 * i], o[2 * i + 1]); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Pairwise Max Fold") {
        auto res = ui::fold(v, op::pmax_t{});
        REQUIRE(res == max);
    }

    WHEN("Pairwise Min Fold") {
        auto res = ui::fold(v, op::pmin_t{});
        REQUIRE(res == min);
    }

    WHEN("Pairwise Max Fold") {
        auto res = ui::fold(v, op::max_t{});
        REQUIRE(res == max);
    }

    WHEN("Pairwise Min Fold") {
        auto res = ui::fold(v, op::min_t{});
        REQUIRE(res == min);
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
    VEC_ARCH_NAME " Float Min-Max",
    "[min-max][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Normal max") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        auto res = ui::max(v, o);
        INFO(std::format("max(v, o): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = std::max(ftype(v[i]), ftype(o[i])); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Normal min") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        auto res = ui::min(v, o);
        INFO(std::format("min(v, o): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = std::min(ftype(v[i]), ftype(o[i])); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Number-Max") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        v[0] = NAN;
        o[1] = NAN;
        auto res = maxnm(v, o);
        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(o[0]), eps<ftype>));
        REQUIRE_THAT(ftype(res[1]), Catch::Matchers::WithinRel(ftype(v[1]), eps<ftype>));
        for (auto i = 2ul; i < N; ++i) {
            type l = res[i];
            type r = std::max(ftype(v[i]), ftype(o[i])); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Number-Min") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        v[0] = NAN;
        o[1] = NAN;
        auto res = minnm(v, o);
        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(o[0]), eps<ftype>));
        REQUIRE_THAT(ftype(res[1]), Catch::Matchers::WithinRel(ftype(v[1]), eps<ftype>));
        for (auto i = 2ul; i < N; ++i) {
            type l = res[i];
            type r = std::min(ftype(v[i]), ftype(o[i])); 
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Pairwise Max") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        auto res = ui::pmax(v, o);
        INFO(std::format("pmax(v, o): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::max(ftype(v[2 * i]), ftype(v[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::max(ftype(o[2 * i]), ftype(o[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Pairwise Min") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        auto res = ui::pmin(v, o);
        INFO(std::format("pmax(v, o): {}", res));
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::min(ftype(v[2 * i]), ftype(v[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
        for (auto i = 0ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::min(ftype(o[2 * i]), ftype(o[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Pairwise Number-Max") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        v[0] = NAN;
        o[1] = NAN;

        auto res = ui::pmaxnm(v, o);
        INFO(std::format("o: {}", o));
        INFO(std::format("pmax(v, o): {}", res));

        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(v[1]), eps<ftype>));
        REQUIRE_THAT(ftype(res[N / 2]), Catch::Matchers::WithinRel(ftype(o[0]), eps<ftype>));
        for (auto i = 2ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::max(ftype(v[2 * i]), ftype(v[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
        for (auto i = 2ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::max(ftype(o[2 * i]), ftype(o[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Pairwise Number-Min") {
        auto o = cast<type>(DataGenerator<N, ftype>::random());
        v[0] = NAN;
        o[1] = NAN;

        auto res = ui::pminnm(v, o);
        INFO(std::format("o: {}", o));
        INFO(std::format("pmax(v, o): {}", res));

        REQUIRE_THAT(ftype(res[0]), Catch::Matchers::WithinRel(ftype(v[1]), eps<ftype>));
        REQUIRE_THAT(ftype(res[N / 2]), Catch::Matchers::WithinRel(ftype(o[0]), eps<ftype>));
        for (auto i = 2ul; i < N / 2; ++i) {
            type l = res[i];
            type r = std::min(ftype(v[2 * i]), ftype(v[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
        for (auto i = 2ul; i < N / 2; ++i) {
            type l = res[N / 2 + i];
            type r = std::min(ftype(o[2 * i]), ftype(o[2 * i + 1]));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Pairwise Max Fold") {
        auto res = ui::fold(v, op::pmax_t{});
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(std::max(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Pairwise Min Fold") {
        auto res = ui::fold(v, op::pmin_t{});
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(std::min(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Pairwise Number-Max Fold") {
        v[0] = NAN;
        auto res = ui::fold(v, op::pmaxnm_t{});
        REQUIRE(!std::isnan(ftype(res)));
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(ui::internal::maxnm(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Pairwise Number-Min Fold") {
        v[0] = NAN;
        auto res = ui::fold(v, op::pminnm_t{});
        REQUIRE(!std::isnan(ftype(res)));
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(ui::internal::minnm(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Max Fold") {
        auto res = ui::fold(v, op::max_t{});
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(std::max(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Min Fold") {
        auto res = ui::fold(v, op::min_t{});
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(std::min(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Number-Max Fold") {
        v[0] = NAN;
        auto res = ui::fold(v, op::maxnm_t{});
        REQUIRE(!std::isnan(ftype(res)));
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(ui::internal::maxnm(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }

    WHEN("Number-Min Fold") {
        v[0] = NAN;
        auto res = ui::fold(v, op::minnm_t{});
        REQUIRE(!std::isnan(ftype(res)));
        INFO(std::format("pmfold(v): {}", res));
        ftype m = std::accumulate(v.data(), v.data() + N, ftype{}, [](auto l, auto r) {
            return ftype(ui::internal::minnm(ftype(l), ftype(r)));
        });
        REQUIRE_THAT(ftype(res), Catch::Matchers::WithinRel(ftype(m), eps<ftype>));
    }
}
