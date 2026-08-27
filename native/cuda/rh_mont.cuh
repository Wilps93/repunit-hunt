/*==============================================================================
 * rh_mont.cuh — Montgomery-арифметика трёх ширин, реализованная шаблонами.
 *
 * Выбор ширины делает хост по величине q_max:
 *   q < 2^63   -> Mont64   (~1 mul + 2 mulhi на mulmod, максимальная скорость)
 *   q < 2^95   -> Mont96   (3x32-битные лимбы; выгодна на Turing/Ampere)
 *   q < 2^127  -> Mont128  (2x64 + PTX carry chain)
 *
 * Все три реализуют один и тот же интерфейс:
 *     using T;                     // тип представления
 *     static T   ninv(T q);        // -q^{-1} mod R
 *     static T   to_mont(u64 a, T q);
 *     static T   one(T q);         // R mod q
 *     static T   mul(T a,T b,T q,T ninv);
 *     static bool eq(T a, T b);
 *============================================================================*/
#ifndef RH_MONT_CUH
#define RH_MONT_CUH
#include "rh_u128.cuh"

/*──────────────────────────── 64-битная ────────────────────────────*/
struct Mont64 {
    using T = uint64_t;

    /* -q^{-1} mod 2^64. Ньютон по 2-адике: x_{i+1}=x_i(2-q x_i). */
    __device__ static __forceinline__ T ninv(T q) {
        T x = (3ull*q) ^ 2ull;          /* 5 верных бит  */
        x *= 2ull - q*x;                /* 10 */
        x *= 2ull - q*x;                /* 20 */
        x *= 2ull - q*x;                /* 40 */
        x *= 2ull - q*x;                /* 64+ */
        x *= 2ull - q*x;
        return (T)(0ull - x);
    }

    __device__ static __forceinline__ T mul(T a, T b, T q, T ni) {
        T lo, hi;
        asm("mul.lo.u64 %0,%2,%3; mul.hi.u64 %1,%2,%3;" : "=l"(lo),"=l"(hi) : "l"(a),"l"(b));
        T m = lo * ni;
        T mq_lo, mq_hi;
        asm("mul.lo.u64 %0,%2,%3; mul.hi.u64 %1,%2,%3;" : "=l"(mq_lo),"=l"(mq_hi) : "l"(m),"l"(q));
        /* t = (T + m*q) >> 64; младшие 64 бита обнуляются по построению */
        T carry; T sum;
        asm("add.cc.u64 %0,%2,%3; addc.u64 %1,0,0;" : "=l"(sum),"=l"(carry) : "l"(lo),"l"(mq_lo));
        T t = hi + mq_hi + carry;
        return (t >= q) ? (t - q) : t;   /* q < 2^63 => одно вычитание */
    }

    /* R mod q = 2^64 mod q через 2 шага по 32 бита (безветвочно) */
    __device__ static __forceinline__ T shl32_mod(T a, T q) {
        #pragma unroll
        for (int i=0;i<32;++i) { a <<= 1; a = (a>=q)? a-q : a; }
        return a;
    }
    __device__ static __forceinline__ T to_mont(uint64_t a, T q) {
        T r = a % q; r = shl32_mod(r,q); r = shl32_mod(r,q); return r;
    }
    __device__ static __forceinline__ T one(T q)             { return to_mont(1ull,q); }
    __device__ static __forceinline__ bool eq(T a, T b)      { return a==b; }
    __device__ static __forceinline__ T from_u64(uint64_t x) { return x; }
    __device__ static __forceinline__ uint64_t lo64(T x)     { return x; }
    __device__ static __forceinline__ uint64_t hi64(T x)     { return 0ull; }
};

/*──────────────────────────── 128-битная ───────────────────────────*/
struct Mont128 {
    using T = u128;

    /* -q^{-1} mod 2^128. Ньютон: сначала 64 бита, потом лифт до 128. */
    __device__ static __forceinline__ T ninv(T q) {
        uint64_t x = (3ull*q.lo) ^ 2ull;
        #pragma unroll
        for (int i=0;i<5;++i) x *= 2ull - q.lo*x;   /* x = q^{-1} mod 2^64 */
        /* Лифт: X = x*(2 - q*x) mod 2^128 */
        u128 X = mk128(0, x);
        u128 qx = mul_low128(q, X);                  /* q*x mod 2^128 */
        u128 two = mk128(0,2);
        u128 t = sub128(two, qx);
        X = mul_low128(X, t);
        /* -q^{-1} */
        u128 zero = mk128(0,0);
        return sub128(zero, X);
    }

    /* Младшие 128 бит произведения (для Ньютона и REDC) */
    __device__ static __forceinline__ u128 mul_low128(u128 a, u128 b) {
        u128 r = mul64x64(a.lo, b.lo);
        r.hi += a.lo*b.hi + a.hi*b.lo;
        return r;
    }

    /* Полное 128x128 -> 256, возвращаем через 4 лимба */
    __device__ static __forceinline__ void mul_full(u128 a, u128 b,
                                                    uint64_t &r0, uint64_t &r1,
                                                    uint64_t &r2, uint64_t &r3) {
        u128 ll = mul64x64(a.lo, b.lo);
        u128 lh = mul64x64(a.lo, b.hi);
        u128 hl = mul64x64(a.hi, b.lo);
        u128 hh = mul64x64(a.hi, b.hi);
        r0 = ll.lo;
        /* r1 = ll.hi + lh.lo + hl.lo  (+carry) */
        uint64_t c1, c2;
        asm("add.cc.u64 %0,%2,%3; addc.u64 %1,0,0;" : "=l"(r1),"=l"(c1) : "l"(ll.hi),"l"(lh.lo));
        asm("add.cc.u64 %0,%0,%2; addc.u64 %1,%1,0;" : "+l"(r1),"+l"(c1) : "l"(hl.lo));
        /* r2 = hh.lo + lh.hi + hl.hi + c1 */
        asm("add.cc.u64 %0,%2,%3; addc.u64 %1,0,0;" : "=l"(r2),"=l"(c2) : "l"(hh.lo),"l"(lh.hi));
        asm("add.cc.u64 %0,%0,%2; addc.u64 %1,%1,0;" : "+l"(r2),"+l"(c2) : "l"(hl.hi));
        asm("add.cc.u64 %0,%0,%2; addc.u64 %1,%1,0;" : "+l"(r2),"+l"(c2) : "l"(c1));
        r3 = hh.hi + c2;
    }

    /* REDC для 128-битного модуля q < 2^127 */
    __device__ static __forceinline__ T mul(T a, T b, T q, T ni) {
        uint64_t t0,t1,t2,t3;
        mul_full(a, b, t0, t1, t2, t3);

        /* m = (T mod 2^128) * ni mod 2^128 */
        u128 tlow = mk128(t1, t0);
        u128 m = mul_low128(tlow, ni);

        /* T + m*q, затем >> 128. Цепочка переносов — аппаратная (add.cc/addc.cc):
         * младшие два лимба обнуляются по построению REDC, но полагаться на это
         * при ручной переброске переносов не стоит — см. историю Mont96. */
        uint64_t p0,p1,p2,p3;
        mul_full(m, q, p0, p1, p2, p3);

        uint64_t u0,u1,u2,u3;
        asm("add.cc.u64  %0, %4, %8;\n\t"
            "addc.cc.u64 %1, %5, %9;\n\t"
            "addc.cc.u64 %2, %6, %10;\n\t"
            "addc.u64    %3, %7, %11;"
            : "=l"(u0), "=l"(u1), "=l"(u2), "=l"(u3)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
              "l"(p0), "l"(p1), "l"(p2), "l"(p3));
        (void)u0; (void)u1;                    /* младшие 128 бит нулевые */

        u128 r = mk128(u3, u2);
        return ge128(r, q) ? sub128(r, q) : r;
    }

    /* a*R mod q через 128 удвоений (вне горячего цикла, 1 раз на кандидата) */
    __device__ static __forceinline__ T to_mont(uint64_t a, T q) {
        u128 r = mk128(0, a);
        if (ge128(r,q)) r = sub128(r,q);
        #pragma unroll 8
        for (int i=0;i<128;++i) {
            r = shl1_128(r);
            if (ge128(r,q)) r = sub128(r,q);
        }
        return r;
    }
    __device__ static __forceinline__ T one(T q)         { return to_mont(1ull,q); }
    __device__ static __forceinline__ bool eq(T a, T b)  { return eq128(a,b); }
    __device__ static __forceinline__ T from_u64(uint64_t x){ return mk128(0,x); }
    __device__ static __forceinline__ uint64_t lo64(T x)  { return x.lo; }
    __device__ static __forceinline__ uint64_t hi64(T x)  { return x.hi; }
};

/*──────────────────────────── 96-битная ────────────────────────────
 * Представление: 3 x 32-бит в двух uint64 (lo содержит limb0|limb1<<32,
 * hi содержит limb2). На Ampere 32-битные IMAD дешевле 64-битных,
 * поэтому 96-битный путь заметно быстрее общего 128-битного.
 * Для компактности используем ту же структуру u128 с hi < 2^32.
 *───────────────────────────────────────────────────────────────────*/
struct Mont96 {
    using T = u128;   /* hi < 2^32 */

    __device__ static __forceinline__ T ninv(T q) {
        /* -q^{-1} mod 2^96: считаем mod 2^128 и обрезаем (лишние биты не влияют,
           т.к. в REDC мы берём m mod 2^96 явно). */
        return Mont128::ninv(q);
    }
    __device__ static __forceinline__ T mask96(T x) { x.hi &= 0xFFFFFFFFull; return x; }

    /* REDC по модулю R = 2^96.
     *
     * ВНИМАНИЕ на два места, где 96-битная редукция ведёт себя не так, как
     * 128-битная (обе давали здесь ошибку, тест native/tests/test_mont.cu):
     *
     *  1) Перенос. В Mont128 младшие ДВА лимба суммы T+m*q обнуляются по
     *     построению, поэтому перенос можно тащить «через лимб». При R = 2^96
     *     обнуляются только 96 младших бит, старшие 32 бита лимба 1 остаются
     *     произвольными — цепочку переносов обязана вести сама аппаратура
     *     (add.cc -> addc.cc -> addc.cc -> addc).
     *
     *  2) Сдвиг. Бит 96 лежит в СЕРЕДИНЕ лимба 1, поэтому результат
     *     (T+m*q) >> 96 собирается из трёх лимбов u1,u2,u3, а не из двух
     *     старших: отбрасывать u1 значит потерять младшие 32 бита ответа.  */
    __device__ static __forceinline__ T mul(T a, T b, T q, T ni) {
        uint64_t t0,t1,t2,t3;
        Mont128::mul_full(a,b,t0,t1,t2,t3);           /* 192 значащих бит */
        /* m = (T mod 2^96)*ni mod 2^96 */
        u128 tlow = mask96(mk128(t1,t0));
        u128 m = mask96(Mont128::mul_low128(tlow, ni));

        uint64_t p0,p1,p2,p3;
        Mont128::mul_full(m,q,p0,p1,p2,p3);

        /* U = T + m*q — честная 4-лимбовая цепочка переносов */
        uint64_t u0,u1,u2,u3;
        asm("add.cc.u64  %0, %4, %8;\n\t"
            "addc.cc.u64 %1, %5, %9;\n\t"
            "addc.cc.u64 %2, %6, %10;\n\t"
            "addc.u64    %3, %7, %11;"
            : "=l"(u0), "=l"(u1), "=l"(u2), "=l"(u3)
            : "l"(t0), "l"(t1), "l"(t2), "l"(t3),
              "l"(p0), "l"(p1), "l"(p2), "l"(p3));
        (void)u0;                                     /* младшие 96 бит нулевые */

        /* r = U >> 96: биты берутся начиная с 32-го бита лимба u1 */
        u128 r = mk128((u2 >> 32) | (u3 << 32),
                       (u1 >> 32) | (u2 << 32));
        return ge128(r,q) ? sub128(r,q) : r;
    }

    __device__ static __forceinline__ T to_mont(uint64_t a, T q) {
        u128 r = mk128(0,a);
        if (ge128(r,q)) r = sub128(r,q);
        #pragma unroll 8
        for (int i=0;i<96;++i) { r = shl1_128(r); if (ge128(r,q)) r = sub128(r,q); }
        return r;
    }
    __device__ static __forceinline__ T one(T q)        { return to_mont(1ull,q); }
    __device__ static __forceinline__ bool eq(T a,T b)  { return eq128(a,b); }
    __device__ static __forceinline__ T from_u64(uint64_t x){ return mk128(0,x); }
    __device__ static __forceinline__ uint64_t lo64(T x) { return x.lo; }
    __device__ static __forceinline__ uint64_t hi64(T x) { return x.hi; }
};

#endif