
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <concepts>
#include <cstdint>
#include <print>
#include "ui.hpp"
#include "utils.hpp"

using namespace ui;

template <typename T>
struct Fixture{
    using type = T;
    using mtype = mask_inner_t<T>;
    static constexpr std::size_t N = 32ul / std::max<unsigned>(sizeof(T) / 2, 1);

};

using SignedTypes = std::tuple<
    std::int8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    Fixture,
    VEC_ARCH_NAME " Compare Operations",
    "[compare][signed]",
    SignedTypes
) {
    using type = Fixture<TestType>::type;
    using mtype = Fixture<TestType>::mtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Equal(=)") {
        auto res = cmp(v, v, op::equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = ~mtype(0);
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("And test") {
        auto res = cmp(v, v, op::and_test_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] & v[i]) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal to 0(>=0)") {
        auto res = cmp(v, op::greater_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than(>)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than 0(>0)") {
        auto res = cmp(v, op::greater_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Greater Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_greater_equal_t{});
        auto tv = sat_abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] >= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Greater than(>)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_greater_t{});
        auto tv = sat_abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] > 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {} > {}", i, std::int64_t(v[i]), 5));
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal(<=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal to 0(<=0)") {
        auto res = cmp(v, op::less_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than(<)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than 0(<0)") {
        auto res = cmp(v, op::less_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Less Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_less_equal_t{});
        auto tv = sat_abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] <= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Less than(>)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_less_t{});
        auto tv = sat_abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] < 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {} < {}", i, std::int64_t(v[i]), 5));
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }
}

using UnsignedTypes = std::tuple<
    std::uint8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t
>;

TEMPLATE_LIST_TEST_CASE_METHOD(
    Fixture,
    VEC_ARCH_NAME " Compare Operations",
    "[compare][unsigned]",
    UnsignedTypes
) {
    using type = Fixture<TestType>::type;
    using mtype = Fixture<TestType>::mtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Equal(=)") {
        auto res = cmp(v, v, op::equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = ~mtype(0);
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("And test") {
        auto res = cmp(v, v, op::and_test_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] & v[i]) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal to 0(>=0)") {
        auto res = cmp(v, op::greater_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {} >= 0", i, int64_t(v[i])));
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than(>)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than 0(>0)") {
        auto res = cmp(v, op::greater_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal(<=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal to 0(<=0)") {
        auto res = cmp(v, op::less_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than(<)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than 0(<0)") {
        auto res = cmp(v, op::less_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
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
    VEC_ARCH_NAME " Compare Operations",
    "[compare][float]",
    FTypes
) {
    using type = Fixture<TestType>::type;
    using mtype = Fixture<TestType>::mtype;
    static constexpr auto N = Fixture<TestType>::N;
    auto v = DataGenerator<N, type>::make();

    INFO("[Vec]: " << std::format("{}", v));

    static constexpr auto min = std::numeric_limits<type>::min();
    static constexpr auto max = std::numeric_limits<type>::max();

    WHEN("Equal(=)") {
        auto res = cmp(v, v, op::equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = ~mtype(0);
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater Equal to 0(>=0)") {
        auto res = cmp(v, op::greater_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] >= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than(>)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::greater_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Greater than 0(>0)") {
        auto res = cmp(v, op::greater_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] > 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Greater Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_greater_equal_t{});
        auto tv = abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] >= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Greater than(>)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_greater_t{});
        auto tv = abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] > 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {} > {}", i, std::int64_t(v[i]), 5));
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal(<=)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_equal_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less Equal to 0(<=0)") {
        auto res = cmp(v, op::less_equal_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] <= 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than(<)") {
        auto res = cmp(v, Vec<N, type>::load(5), op::less_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Less than 0(<0)") {
        auto res = cmp(v, op::less_zero_t{});
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (v[i] < 0) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Less Equal(>=)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_less_equal_t{});
        auto tv = abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] <= 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }

    WHEN("Abs Less than(>)") {
        auto res = cmp(v, Vec<N, type>::load(-5), op::abs_less_t{});
        auto tv = abs(v);
        for (auto i = 0ul; i < N; ++i) {
            mtype l = res[i];
            mtype r = (tv[i] < 5) ? ~mtype(0) : 0;
            INFO(std::format("[{}]: {} < {}", i, std::int64_t(v[i]), 5));
            INFO(std::format("[{}]: {:x} == {:x}", i, l, r));
            REQUIRE(l == r);
        }
    }
}
