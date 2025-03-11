
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "catch2/matchers/catch_matchers_floating_point.hpp"

#include <cstdint>
#include <format>
#include <print>
#include <cmath>
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
    VEC_ARCH_NAME " Mul Operations",
    "[mul][signed]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using wtype = Fixture<TestType>::wtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO(std::format("[Vec<{}>]: {}", get_type_name<TestType>(), v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Normal multiplication") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul(v, d);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] * d[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Multiplication with accumulating addition") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, d, op::add_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE(res[i] == type(1 + v[i] * d[i]));
        }
    }

    WHEN("Multiplication with accumulating subtraction") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, d, op::sub_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE(res[i] == type(1 - v[i] * d[i]));
        }
    }

    if constexpr (sizeof(type) < 8) {
        WHEN("Widening multiplication with accumulating addition") {
            auto d = DataGenerator<N, type>::random();
            auto res = mul_acc(Vec<N, wtype>::load(1), v, d, op::add_t{});
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("amul(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto v0 = wtype(v[i]);
                auto d0 = wtype(d[i]); 
                REQUIRE(res[i] == wtype(1 + v0 * d0));
            }
        }

        WHEN("Widening multiplication with accumulating subtraction") {
            auto d = DataGenerator<N, type>::random();
            auto res = mul_acc(Vec<N, wtype>::load(1), v, d, op::sub_t{});
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("amul(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto v0 = wtype(v[i]);
                auto d0 = wtype(d[i]); 
                REQUIRE(res[i] == wtype(1 - v0 * d0));
            }
        }

        WHEN("Widening multiplication") {
            auto d = DataGenerator<N, type>::random();
            auto res = widening_mul(v, d);
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            INFO(std::format("wmul(v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                auto v0 = wtype(v[i]);
                auto d0 = wtype(d[i]); 
                REQUIRE(res[i] == wtype(v0 * d0));
            }
        }
    }

    WHEN("Multiplication with accumulating addition at given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = mul_acc</*Lane=*/0>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == type(1 + v[i] * d[0]));
            }
        }
        {
            auto res = mul_acc</*Lane=*/N-1>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                REQUIRE(res[i] == type(1 + v[i] * d[N-1]));
            }
        }
    }

    WHEN("Multiplication by a constant with accumulating addition") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, type(2), op::add_t{});
        INFO(std::format("amul(1s, v, c): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE(res[i] == type(1 + v[i] * 2));
        }
    }

    WHEN("Multiplication by a constant with accumulating subtraction") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, type(2), op::sub_t{});
        INFO(std::format("amul(1s, v, c): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            REQUIRE(res[i] == type(1 - v[i] * 2));
        }
    }

    WHEN("Multiplication by a constant") {
        auto res = mul(v, type(2));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] * 2);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Multiplication by a constant at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = mul</*Lane=*/0>(v, d);
            for (auto i = 0ul; i < N; ++i) {
                type l = res[i];
                type r = type(v[i] * d[0]);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE(l == r);
            }
        }
        {
            auto res = mul</*Lane=*/N-1>(v, d);
            for (auto i = 0ul; i < N; ++i) {
                type l = res[i];
                type r = type(v[i] * d[N-1]);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE(l == r);
            }
        }
    }

    if constexpr (sizeof(type) < 8) {
        WHEN("Widening multiplication by a constant") {
            auto res = widening_mul(v, type(2));
            STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
            for (auto i = 0ul; i < N; ++i) {
                type l = res[i];
                type r = wtype(v[i] * 2);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE(l == r);
            }
        }

        WHEN("Widening multiplication by a constant at a given lane") {
            auto d = DataGenerator<N, type>::random();
            {
                auto res = widening_mul</*Lane=*/0>(v, d);
                STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
                for (auto i = 0ul; i < N; ++i) {
                    type l = res[i];
                    type r = type(v[i] * d[0]);
                    INFO(std::format("[{}]: {} == {}", i, l, r));
                    REQUIRE(l == r);
                }
            }
            {
                auto res = widening_mul</*Lane=*/N-1>(v, d);
                STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
                for (auto i = 0ul; i < N; ++i) {
                    type l = res[i];
                    type r = type(v[i] * d[N-1]);
                    INFO(std::format("[{}]: {} == {}", i, l, r));
                    REQUIRE(l == r);
                }
            }
        }

        WHEN("Widening accumulating multiplication with addition by a constant") {
            auto a = Vec<N, wtype>::load(1);
            {
                auto res = widening_mul_acc(a, v, type(2), op::add_t{});
                STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
                for (auto i = 0ul; i < N; ++i) {
                    type l = res[i];
                    type r = wtype(1 + v[i] * 2);
                    INFO(std::format("[{}]: {} == {}", i, l, r));
                    REQUIRE(l == r);
                }
            }
            {
                auto res = widening_mul_acc(a, v, type(2), op::add_t{});
                STATIC_REQUIRE(std::same_as<typename decltype(res)::element_t, wtype>);
                for (auto i = 0ul; i < N; ++i) {
                    type l = res[i];
                    type r = wtype(1 + v[i] * 2);
                    INFO(std::format("[{}]: {} == {}", i, l, r));
                    REQUIRE(l == r);
                }
            }
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
    VEC_ARCH_NAME " Mul Operations",
    "[mul][float]",
    FTypes
) {
    using type = FloatFixture<TestType>::type;
    using ftype = FloatFixture<TestType>::ftype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Normal multiplication") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul(v, d);
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] * d[i]);
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Multiplication with accumulating addition") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, d, op::add_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(type(1 + (ftype(v[i]) * ftype(d[i]))));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Multiplication with accumulating subtraction") {
        auto d = DataGenerator<N, type>::random();
        auto res = mul_acc(Vec<N, type>::load(1), v, d, op::sub_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            auto t0 = type(ftype(v[i]) * ftype(d[i]));
            ftype r = ftype(type(1 - ftype(t0)));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Fused-Multiplication addition") {
        auto d = DataGenerator<N, type>::random();
        auto res = fused_mul_acc(Vec<N, type>::load(1), v, d, op::add_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            ftype r = ftype(type(1 + (ftype(v[i]) * ftype(d[i]))));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Fused-Multiplication subtraction") {
        auto d = DataGenerator<N, type>::random();
        auto res = fused_mul_acc(Vec<N, type>::load(1), v, d, op::sub_t{});
        INFO(std::format("amul(1s, v, d): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            ftype l = ftype(res[i]);
            auto t0 = type(ftype(v[i]) * ftype(d[i]));
            ftype r = ftype(type(1 - ftype(t0)));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Fused-Multiplication addition at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = fused_mul_acc</*Lane=*/0>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[0]));
                ftype r = ftype(type(1 + ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }

        {
            auto res = fused_mul_acc</*Lane=*/N - 1>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[N - 1]));
                ftype r = ftype(type(1 + ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
    }

    WHEN("Fused-Multiplication subtraction at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = fused_mul_acc</*Lane=*/0>(Vec<N, type>::load(1), v, d, op::sub_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[0]));
                ftype r = ftype(type(1 - ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }

        {
            auto res = fused_mul_acc</*Lane=*/N - 1>(Vec<N, type>::load(1), v, d, op::sub_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[N - 1]));
                ftype r = ftype(type(1 - ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
    }

    WHEN("Accumulating multiplication with addition at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = mul_acc</*Lane=*/0>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[0]));
                ftype r = ftype(type(1 + ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }

        {
            auto res = mul_acc</*Lane=*/N - 1>(Vec<N, type>::load(1), v, d, op::add_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[N - 1]));
                ftype r = ftype(type(1 + ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
    }

    WHEN("Accumulating multiplication with subtraction at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = mul_acc</*Lane=*/0>(Vec<N, type>::load(1), v, d, op::sub_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[0]));
                ftype r = ftype(type(1 - ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }

        {
            auto res = mul_acc</*Lane=*/N - 1>(Vec<N, type>::load(1), v, d, op::sub_t{});
            INFO(std::format("amul<L>(1s, v, d): {}", res));
            for (auto i = 0ul; i < N; ++i) {
                ftype l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[N - 1]));
                ftype r = ftype(type(1 - ftype(t0)));
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
            }
        }
    }

    WHEN("Multiplication by a constant with accumulating addition") {
        auto res = mul_acc(Vec<N, type>::load(1), v, type(2), op::add_t{});
        INFO(std::format("amul(1s, v, c): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto l = ftype(res[i]);
            auto t0 = type(ftype(v[i]) * ftype(2));
            ftype r = ftype(type(1 + ftype(t0)));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Multiplication by a constant with accumulating subtraction") {
        auto res = mul_acc(Vec<N, type>::load(1), v, type(2), op::sub_t{});
        INFO(std::format("amul(1s, v, c): {}", res));
        for (auto i = 0ul; i < N; ++i) {
            auto l = ftype(res[i]);
            auto t0 = type(ftype(v[i]) * ftype(2));
            ftype r = ftype(type(1 - ftype(t0)));
            REQUIRE_THAT(l, Catch::Matchers::WithinRel(r, eps<ftype>));
        }
    }

    WHEN("Multiplication by a constant") {
        auto res = mul(v, type(2));
        for (auto i = 0ul; i < N; ++i) {
            type l = res[i];
            type r = type(v[i] * type(2));
            INFO(std::format("[{}]: {} == {}", i, l, r));
            REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
        }
    }

    WHEN("Multiplication by a constant at a given lane") {
        auto d = DataGenerator<N, type>::random();
        {
            auto res = mul</*Lane=*/0>(v, d);
            for (auto i = 0ul; i < N; ++i) {
                auto l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[0]));
                ftype r = ftype(t0);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
            }
        }
        {
            auto res = mul</*Lane=*/N-1>(v, d);
            for (auto i = 0ul; i < N; ++i) {
                auto l = ftype(res[i]);
                auto t0 = type(ftype(v[i]) * ftype(d[N-1]));
                ftype r = ftype(t0);
                INFO(std::format("[{}]: {} == {}", i, l, r));
                REQUIRE_THAT(ftype(l), Catch::Matchers::WithinRel(ftype(r), eps<ftype>));
            }
        }
    }
}
