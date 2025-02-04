#ifndef AMT_UI_FORMAT_HPP
#define AMT_UI_FORMAT_HPP

#include "ui/base_vec.hpp"
#include <concepts>
#include <format>

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
                            format_to(out, "0b{:0{}b}", v[i], sizeof(T) * 8);
                        }
                    } break;
                    case Radix::hex: {
                        if constexpr (std::floating_point<T>) {
                            format_to(out, "0x{:a}", v[i]);
                        } else {
                            format_to(out, "0x{:0{}x}", v[i], sizeof(T) * 8 / 4);
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
}

#endif // AMT_UI_FORMAT_HPP
