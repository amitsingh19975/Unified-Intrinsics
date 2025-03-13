#ifndef AMT_UI_ARCH_ARM_INFO_HPP
#define AMT_UI_ARCH_ARM_INFO_HPP

#include "../features.hpp"
#include <cstdint>
#include <vector>
#include <optional>

#define UI_CPU_API_ID_WIN 1
#define UI_CPU_API_ID_BSD 2
#define UI_CPU_API_ID_LINUX 3

#if defined(UI_OS_WIN)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#    define WIN32_IS_MEAN_WAS_LOCALLY_DEFINED
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#    define NOMINMAX_WAS_LOCALLY_DEFINED
#  endif
#
#  include <windows.h>
#
#  ifdef WIN32_IS_MEAN_WAS_LOCALLY_DEFINED
#    undef WIN32_IS_MEAN_WAS_LOCALLY_DEFINED
#    undef WIN32_LEAN_AND_MEAN
#  endif
#  ifdef NOMINMAX_WAS_LOCALLY_DEFINED
#    undef NOMINMAX_WAS_LOCALLY_DEFINED
#    undef NOMINMAX
#  endif
#  define UI_CPU_API UI_CPU_API_ID_WIN
#elif defined(UI_OS_MAC) || defined(UI_OS_BSD)
    #include <sys/sysctl.h>
    #define UI_CPU_API UI_CPU_API_ID_BSD
#elif defined(UI_OS_LINUX) || defined(UI_OS_ANDROID)
    #include <sys/sysinfo.h>
    #define UI_CPU_API UI_CPU_API_ID_LINUX
#endif


namespace ui {
    struct CacheInfo {
        std::uint8_t level;
        unsigned size;
    };

    struct CpuInfo {
        std::vector<CacheInfo> cache;
        std::vector<CacheInfo> icache;
        unsigned cacheline;
        std::size_t mem;
    };

    namespace internal {
        #if UI_CPU_API == UI_CPU_API_ID_BSD
        template <typename T>
        inline static auto read_info_by_name(std::string_view name) -> std::optional<T> {
            if constexpr (std::same_as<T, std::string>) {
                std::string temp;
                temp.resize(1024);
                std::size_t len = 1024;
                auto ret = sysctlbyname(name.data(), temp.data(), &len, 0, 0);

                temp.resize(len);
                if (ret == 0) return temp;
                return {};
            } else if constexpr (std::integral<T>) {
                char buff[128];
                std::size_t len = 128;
                auto ret = sysctlbyname(name.data(), buff, &len, 0, 0);
                if (ret == 0) return *reinterpret_cast<T const*>(buff);
                return {};
            }
        }

        inline static auto cpu_info_helper() -> CpuInfo {
            auto res = CpuInfo {
                .cache = {},
                .icache = {},
                .cacheline = UI_CACHE_LINE_SIZE,
                .mem = 4 * 1024 * 1024,
            };
            res.cache.reserve(3);
            res.icache.reserve(3);
            auto mem = read_info_by_name<std::size_t>("hw.memsize_usable");
            auto cacheline = read_info_by_name<unsigned>("hw.cachelinesize");
            auto l1c = read_info_by_name<unsigned>("hw.l1dcachesize");  
            auto l2c = read_info_by_name<unsigned>("hw.l2cachesize");  
            auto l3c = read_info_by_name<unsigned>("hw.l3cachesize");  
            auto il1c = read_info_by_name<unsigned>("hw.l1icachesize");  
            auto il2c = read_info_by_name<unsigned>("hw.l2icachesize");  
            auto il3c = read_info_by_name<unsigned>("hw.l3icachesize");  

            if (l1c) res.cache.push_back({ .level = 1, .size = *l1c });
            if (l2c) res.cache.push_back({ .level = 2, .size = *l2c });
            if (l3c) res.cache.push_back({ .level = 3, .size = *l3c });

            if (il1c) res.icache.push_back({ .level = 1, .size = *il1c });
            if (il2c) res.icache.push_back({ .level = 2, .size = *il2c });
            if (il3c) res.icache.push_back({ .level = 3, .size = *il3c });
            if (mem) res.mem = *mem;
            if (cacheline) res.cacheline = *cacheline;
            return res;
        }
        #elif UI_CPU_API == UI_CPU_API_ID_WIN
        inline static auto cpu_info_helper() -> CpuInfo {
            auto res = CpuInfo {
                .cache = {},
                .icache = {},
                .cacheline = UI_CACHE_LINE_SIZE,
                .mem = 4 * 1024 * 1024,
            };
            res.cache.reserve(3);
            res.icache.reserve(3);

            // Get total physical memory using GlobalMemoryStatusEx.
            MEMORYSTATUSEX memStatus = {};
            memStatus.dwLength = sizeof(memStatus);
            if (GlobalMemoryStatusEx(&memStatus)) {
                res.mem = static_cast<std::size_t>(memStatus.ullTotalPhys);
            }

            // Get cache information using GetLogicalProcessorInformation.
            DWORD len = 0;
            // First call to determine the buffer size needed.
            if (GetLogicalProcessorInformation(nullptr, &len) == FALSE &&
                GetLastError() == ERROR_INSUFFICIENT_BUFFER) {

                std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
                if (GetLogicalProcessorInformation(buffer.data(), &len)) {
                    for (const auto& info : buffer) {
                        if (info.Relationship == RelationCache) {
                            int level = info.Cache.Level;
                            std::size_t size = info.Cache.Size;
                            // For Windows, info.Cache.Type may be:
                            // CacheUnified, CacheData, or CacheInstruction.
                            if (info.Cache.Type == CacheUnified || info.Cache.Type == CacheData) {
                                res.cache.push_back(CacheInfo { .level = static_cast<std::uint8_t>(level), .size = static_cast<unsigned>(size)});
                            }
                            if (info.Cache.Type == CacheInstruction) {
                                res.icache.push_back(CacheInfo{ .level = static_cast<std::uint8_t>(level), .size = static_cast<unsigned>(size) });
                            }
                            // Use the reported cache line size if available.
                            if (info.Cache.LineSize > 0) {
                                res.cacheline = info.Cache.LineSize;
                            }
                        }
                    }
                }
            }

            return res;
        }
        #elif UI_CPU_API == UI_CPU_API_ID_LINUX
        inline static auto cpu_info_helper() -> CpuInfo {
            auto res = CpuInfo {
                .cache = {},
                .icache = {},
                .cacheline = UI_CACHE_LINE_SIZE,
                .mem = 4 * 1024 * 1024,
            };
            res.cache.reserve(3);
            res.icache.reserve(3);
            // Get total memory using sysinfo
            struct sysinfo info;
            if (sysinfo(&info) == 0) {
                res.mem = info.totalram * info.mem_unit;
            }

            // Get cache line size using sysconf.
            // _SC_LEVEL1_DCACHE_LINESIZE is available on some systems.
            long cl = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
            if (cl > 0)
                res.cacheline = static_cast<unsigned>(cl);

            // Get cache sizes (if available)
            long l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
            if (l1d > 0)
                res.cache.push_back({ 1, static_cast<unsigned>(l1d) });

            long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
            if (l2 > 0)
                res.cache.push_back({ 2, static_cast<unsigned>(l2) });

            long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
            if (l3 > 0)
                res.cache.push_back({ 3, static_cast<unsigned>(l3) });

            // For instruction caches, some systems define these macros.
            #ifdef _SC_LEVEL1_ICACHE_SIZE
            long l1i = sysconf(_SC_LEVEL1_ICACHE_SIZE);
            if (l1i > 0)
                res.icache.push_back({ 1, static_cast<unsigned>(l1i) });
            #endif

            #ifdef _SC_LEVEL2_ICACHE_SIZE
            long l2i = sysconf(_SC_LEVEL2_ICACHE_SIZE);
            if (l2i > 0)
                res.icache.push_back({ 2, static_cast<unsigned>(l2i) });
            #endif

            #ifdef _SC_LEVEL3_ICACHE_SIZE
            long l3i = sysconf(_SC_LEVEL3_ICACHE_SIZE);
            if (l3i > 0)
                res.icache.push_back({ 3, static_cast<unsigned>(l3i) });
            #endif

            return res;
        }
        #else
        inline static auto cpu_info_helper() -> CpuInfo {
            auto res = CpuInfo {
                .cache = {},
                .icache = {},
                .cacheline = UI_CACHE_LINE_SIZE,
                .mem = 4 * 1024 * 1024,
            };
            return res;
        }
        #endif
    } // namespace interanl

    inline static auto cpu_info() noexcept {
        return internal::cpu_info_helper();
    }

} // ui

#undef UI_CPU_API
#undef UI_CPU_API_ID_LINUX
#undef UI_CPU_API_ID_WIN
#undef UI_CPU_API_ID_BSD

#endif // AMT_UI_ARCH_ARM_INFO_HPP 
