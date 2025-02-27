#ifndef AMT_UI_FORMAT_HPP
#define AMT_UI_FORMAT_HPP

#include "float.hpp"
#include "base_vec.hpp"
#include "ui/matrix.hpp"
#include <concepts>
#include <format>
#include <iterator>
#include <type_traits>

namespace std {
    template <std::size_t N, typename T>
    struct formatter<ui::Vec<N, T>> {
        enum class Radix {
            dec,    // base-10
            bin,    // base-2
            hex,    // base-16
        };
        Radix radix{Radix::dec};

        constexpr auto set_base(char ch) noexcept {
            switch(ch) {
            case 'b': radix = Radix::bin; return true;
            case 'x': radix = Radix::hex; return true;
            default: radix = Radix::dec; return false;
            }
        }

        constexpr auto parse(format_parse_context& ctx) {
            auto it = ctx.begin();
            while (it != ctx.end() && *it != '}') {
                if (*it == '0') {
                    ++it;
                    set_base(*it);
                } 
                set_base(*it);
                ++it;
            }
            return it;
        }

        auto format(ui::Vec<N, T> const& v, auto& ctx) const {
            using namespace ui;
            auto&& out = ctx.out();
            format_to(out, "[");
            for (auto i = std::size_t{}; i < N; ++i) {
                switch (radix) {
                    case Radix::dec: {
                        format_to(out, "{}", v[i]);
                    } break;
                    case Radix::bin: {
                        if constexpr (std::floating_point<T>) {
                            // TODO: Do we really need binary rep for floats?
                            format_to(out, "{}", v[i]);
                        } else {
                            format_to(out, "{:0{}b}", v[i], sizeof(T) * 8);
                        }
                    } break;
                    case Radix::hex: {
                        if constexpr (std::floating_point<T>) {
                            auto f = fp::decompose_fp(v[i]);
                            if (f.sign) {
                                format_to(out, "-{:a}", -v[i], sizeof(T) * 8);
                            } else {
                                format_to(out, "{:a}", v[i], sizeof(T) * 8);
                            }
                        } else {
                            format_to(out, "{:0{}x}", v[i], sizeof(T) * 8 / 4);
                        }
                    } break;
                    default: break;
                }

                if (i + 1 != N) {
                    format_to(out, ", ");
                }
            }
            return format_to(out, "]");
        }
    };

    template <std::size_t R, std::size_t C, typename T>
    struct formatter<ui::VecMat<R, C, T>> {
        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();
        }

        auto format(ui::VecMat<R, C, T> const& v, auto& ctx) const {
            auto&& out = ctx.out();
            format_to(out, "[\n");
            std::size_t width = 0u;
            std::string buff;
            buff.reserve(100);
            for (auto i = 0u; i < R * C; ++i) {
                format_to(std::back_inserter(buff), "{}", v.data()[i]);
                width = std::max(width, buff.size());
                buff.clear();
            }

            for (auto i = 0u; i < R; ++i) {
                format_to(out, "  [");
                for (auto j = 0u; j < C; ++j) {
                    format_to(out, "{:{}}", v(i, j), width);
                    if (j + 1 < C) {
                        format_to(out, ", ");
                    }
                }
                format_to(out, "]\n");
            }
            return format_to(out, "]");
        }
    };
}

#endif // AMT_UI_FORMAT_HPP
