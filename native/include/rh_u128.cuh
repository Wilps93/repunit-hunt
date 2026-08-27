/*==============================================================================
 * rh_u128.cuh — 128-битные целые для CUDA на inline-PTX.
 *
 * Компилятор NVCC поддерживает __int128 только на хосте. На устройстве
 * используем пару uint64_t + PTX-инструкции с переносом (add.cc / addc.cc),
 * что даёт оптимальный код без вызовов libdevice.
 *============================================================================*/
#ifndef RH_U128_CUH
#define RH_U128_CUH
#include <stdint.h>

struct alignas(16) u128 { uint64_t lo, hi; };

__device__ __forceinline__ u128 mk128(uint64_t h, uint64_t l) { u128 r; r.hi=h; r.lo=l; return r; }

/* 64x64 -> 128 */
__device__ __forceinline__ u128 mul64x64(uint64_t a, uint64_t b) {
    u128 r;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;" : "=l"(r.lo), "=l"(r.hi) : "l"(a), "l"(b));
    return r;
}

__device__ __forceinline__ u128 add128(u128 a, u128 b) {
    u128 r;
    asm("add.cc.u64  %0, %2, %4;\n\t"
        "addc.u64    %1, %3, %5;"
        : "=l"(r.lo), "=l"(r.hi) : "l"(a.lo),"l"(a.hi),"l"(b.lo),"l"(b.hi));
    return r;
}

__device__ __forceinline__ u128 sub128(u128 a, u128 b) {
    u128 r;
    asm("sub.cc.u64  %0, %2, %4;\n\t"
        "subc.u64    %1, %3, %5;"
        : "=l"(r.lo), "=l"(r.hi) : "l"(a.lo),"l"(a.hi),"l"(b.lo),"l"(b.hi));
    return r;
}

__device__ __forceinline__ bool ge128(u128 a, u128 b) {
    return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo);
}
__device__ __forceinline__ bool eq128(u128 a, u128 b) { return a.hi==b.hi && a.lo==b.lo; }
__device__ __forceinline__ bool is_zero128(u128 a)    { return (a.hi|a.lo)==0; }

/* Сдвиг влево на 1 с переносом */
__device__ __forceinline__ u128 shl1_128(u128 a) {
    u128 r; r.hi = (a.hi<<1) | (a.lo>>63); r.lo = a.lo<<1; return r;
}

/* 128 mod 64-bit (используется только вне горячего цикла) */
__device__ __forceinline__ uint64_t mod128_64(u128 a, uint64_t m) {
    /* Классический shift-and-subtract; вызывается 1 раз на кандидата. */
    uint64_t r = a.hi % m;
    #pragma unroll
    for (int i = 63; i >= 0; --i) {
        /* r = (r*2 + bit) mod m, r < m < 2^63 => нет переполнения */
        r = (r << 1) | ((a.lo >> i) & 1ull);
        if (r >= m) r -= m;
    }
    return r;
}

/* Количество ведущих нулей */
__device__ __forceinline__ int clz128(u128 a) {
    return a.hi ? __clzll(a.hi) : 64 + __clzll(a.lo);
}
__device__ __forceinline__ int bitlen128(u128 a) { return 128 - clz128(a); }

#endif