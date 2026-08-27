/*==============================================================================
 *  tf_kernel.cu — GPU Trial Factoring для R_k(b) = (b^k - 1)/(b - 1)   [v2]
 *
 *  МАТЕМАТИКА
 *  ----------
 *  Пусть k простое и q — простой делитель R_k(b). Тогда b^k ≡ 1 (mod q),
 *  значит ord_q(b) | k, т.е. ord_q(b) ∈ {1, k}.
 *
 *   • ord_q(b) = k  =>  k | q-1, а q нечётно  =>  q = 2*m*k + 1.
 *     Это и есть форма кандидатов, которые перебирает ядро.
 *
 *   • ord_q(b) = 1  =>  q | (b-1).  КРАЕВОЙ СЛУЧАЙ:
 *     тогда b^k ≡ 1 (mod q) ВСЕГДА, и наивная проверка powmod дала бы
 *     ЛОЖНОЕ СРАБАТЫВАНИЕ. Истинный критерий здесь другой:
 *         R_k(b) = 1 + b + ... + b^(k-1) ≡ k (mod q)
 *         =>  q | R_k  <=>  q | k.
 *     Для k простого и q = 2mk+1 > k (при m >= 1) это невозможно,
 *     но ядро обрабатывает случай честно, а не по индукции.
 *
 *   • q | b  =>  R_k ≡ 1 (mod q)  =>  не делитель.
 *
 *  РАСКЛАДКА РАБОТЫ
 *  ----------------
 *  Один launch обрабатывает n_k показателей и диапазон
 *  m ∈ [m_start, m_start+m_span). Линейный индекс t раскладывается как
 *  ki = t / m_span, m = m_start + t % m_span, то есть k меняется МЕДЛЕННО:
 *  весь варп почти всегда работает с одним k, а значит с одинаковым kbits
 *  и одинаковой траекторией цикла powmod => дивергенции в горячем цикле нет.
 *
 *  АРИФМЕТИКА
 *  ----------
 *  Montgomery (REDC) трёх ширин — Mont64 / Mont96 / Mont128 из rh_mont.cuh.
 *  Ширину выбирает хост по величине q_max; кандидаты, не влезающие в
 *  выбранную ширину, пропускаются (предохранитель: хост не должен их подавать).
 *============================================================================*/

#include "rh_common.h"
#include "rh_mont.cuh"
#include "small_primes.h"
#include <cuda_runtime.h>
#include <stdint.h>

/*───────────── Малые простые для отсева составных q ─────────────
 * Составное q можно смело пропустить: его простые делители сами имеют
 * вид 2*m'*k+1 и будут проверены при своих m'. */
/* Сколько малых простых использовать в фильтре (<= RH_SMALL_PRIMES_MAX).
 *
 * ВЫБОР ЗНАЧЕНИЯ. Фильтр экономит powmod, но сам не бесплатен, и оптимум
 * лежит НЕ на максимуме. Замер (GTX 1650, n_k=256, мс на запуск):
 *
 *     простых:    54     32     24     16     12      8
 *     k~10^5:   12.5   10.9    ---    9.9    9.4    9.0
 *     k~10^6:   13.8   12.1   11.5   11.0   10.5   10.2
 *
 * Кривая плоская в диапазоне 8..16; берём 12 — практически оптимум по
 * времени при чуть лучшем отсеве. Переопределяется с -D для экспериментов
 * (см. scratchpad/profile_kernel.sh). */
#ifndef RH_SMALL_PRIMES_N
#define RH_SMALL_PRIMES_N 12
#endif

/* q = 2*m*k + 1 в 128 битах (переполнения нет: произведение m*k < 2^128). */
__device__ __forceinline__ u128 make_q(uint64_t m, uint64_t k) {
    u128 mk = mul64x64(m, k);
    u128 q  = shl1_128(mk);
    uint64_t lo = q.lo + 1ull;
    uint64_t hi = q.hi + (lo == 0ull ? 1ull : 0ull);
    return mk128(hi, lo);
}

/* Собственный ли делитель? Нужно q < R_k(b): иначе простой репьюнит был бы
 * «отсеян» самим собой (R_2(10) = 11 при q = 2·1·5+1 = 11), а хост засчитал бы
 * это ложным срабатыванием ядра.
 *
 * Дёшево: при (k-1)·floor(log2 b) >= 127 репьюнит заведомо больше любого q,
 * поэтому точный подсчёт запускается только для крошечных k. */
__device__ __forceinline__ bool repunit_exceeds(uint64_t b, uint64_t k, u128 q) {
    const int lb = 63 - __clzll(b);               /* floor(log2 b) >= 1 при b >= 2 */
    if (lb >= 1 && (k - 1) >= (uint64_t)(127 / lb) + 1) return true;

    /* Цикл выполняется не более ~128 раз (иначе сработало бы условие выше).
     * Каждое умножение проверяется на выход за 128 бит: как только r*b не
     * помещается в u128, оно заведомо больше q < 2^127. */
    u128 r = mk128(0, 1);                         /* R_1 = 1 */
    for (uint64_t i = 1; i < k; ++i) {
        if (__umul64hi(r.hi, b) != 0ull) return true;   /* r*b >= 2^128 */
        u128 t = mul64x64(r.lo, b);
        const uint64_t hi_add = r.hi * b;
        const uint64_t new_hi = t.hi + hi_add;
        if (new_hi < t.hi) return true;                 /* перенос за 128 бит */
        t.hi = new_hi;

        const uint64_t lo = t.lo + 1ull;                /* R_{i+1} = R_i*b + 1 */
        if (lo == 0ull) {
            if (t.hi == ~0ull) return true;
            t.hi += 1ull;
        }
        t.lo = lo;
        r = t;

        if (ge128(r, q) && !eq128(r, q)) return true;   /* r > q */
    }
    return false;                                       /* R_k(b) <= q */
}

/* Есть ли у q делитель среди малых простых?
 *
 * Деление на GPU дорогое (аппаратного целочисленного нет), поэтому вместо
 * `q % p` используется умножение на обратный по модулю 2^64:
 *      p | q  <=>  (q * p^-1) <= (2^64-1)/p
 * Константы считает native/tests/gen_small_primes.py (там же доказательство
 * и самопроверка). Замер на GTX 1650, 54 простых, n_k=256:
 *      деление   — 36.1 мс/launch
 *      умножение — см. README, раздел про производительность.
 */
__device__ __forceinline__ bool has_small_factor(u128 q) {
    if (q.hi == 0ull) {
        #pragma unroll 8
        for (int i = 0; i < RH_SMALL_PRIMES_N; ++i) {
            const uint64_t p = (uint64_t)d_sp_p[i];
            if (p * p > q.lo) break;      /* делителей <= sqrt(q) нет => q простое */
            if (q.lo * d_sp_inv[i] <= d_sp_lim[i]) return true;
        }
        return false;
    }
    /* q >= 2^64: трюк с обратным работает только для полного числа, поэтому
     * здесь остаётся честная редукция. Путь редкий — только широкие ядра. */
    #pragma unroll 4
    for (int i = 0; i < RH_SMALL_PRIMES_N; ++i) {
        if (mod128_64(q, (uint64_t)d_sp_p[i]) == 0ull) return true;
    }
    return false;
}

/*───────────── Проверка «влезает ли q в выбранную ширину» ─────────────
 * W=0: q < 2^63  (в REDC достаточно одного вычитания),
 * W=1: q < 2^95  (Mont96),
 * W=2: q < 2^127 (Mont128).                                            */
template<int W>
__device__ __forceinline__ bool q_fits(u128 q) {
    if (W == 0) return q.hi == 0ull && q.lo < (1ull << 63);
    if (W == 1) return q.hi < (1ull << 31);
    return q.hi < (1ull << 63);
}

/*───────────── Тело ядра, общее для всех трёх ширин ─────────────*/
template<int W, class M>
__device__ __forceinline__
void tf_body(uint64_t base,
             const uint64_t* __restrict__ ks,
             uint32_t  n_k,
             uint64_t  m_start,
             uint64_t  m_span,
             rh_tf_hit_t* __restrict__ hits,
             uint32_t* __restrict__ cnt,
             uint32_t  cap,
             unsigned long long* __restrict__ tested)
{
    const uint64_t total  = m_span * (uint64_t)n_k;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long local_tested = 0ull;

    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += stride)
    {
        const uint32_t ki = (uint32_t)(t / m_span);
        const uint64_t m  = m_start + (t % m_span);
        const uint64_t k  = __ldg(&ks[ki]);
        if (m == 0ull || k < 2ull) continue;

        const u128 q = make_q(m, k);
        if (!q_fits<W>(q)) continue;                 /* предохранитель */
        if (q.hi == 0ull && q.lo < 3ull) continue;

        /* (A) отсев составных q */
        if (has_small_factor(q)) continue;

        /* b mod q. При q >= 2^64 всегда base < q, редукция тривиальна. */
        const uint64_t b_mod = (q.hi == 0ull) ? (base % q.lo) : base;

        /* (B) q | b  =>  R_k ≡ 1 (mod q) — не делитель */
        if (b_mod == 0ull) continue;

        /* (C) q | (b-1): powmod дал бы ложное срабатывание.
         *     Истинный критерий: R_k ≡ k (mod q), т.е. q | R_k <=> q | k. */
        if (b_mod == 1ull) {
            const bool divides_k = (q.hi == 0ull) && (k % q.lo == 0ull);
            if (divides_k && repunit_exceeds(base, k, q)) {
                uint32_t slot = atomicAdd(cnt, 1u);
                if (slot < cap) {
                    hits[slot].q_lo = q.lo; hits[slot].q_hi = q.hi;
                    hits[slot].k_index = ki; hits[slot]._pad = 0u;
                }
            }
            continue;
        }

        /* (D) общий случай: b^k ≡ 1 (mod q) ? */
        typename M::T qm;
        if constexpr (W == 0) qm = q.lo;             /* Mont64: T = uint64_t */
        else                  qm = q;                /* Mont96/Mont128: T = u128 */

        const typename M::T ni  = M::ninv(qm);
        const typename M::T b_m = M::to_mont(b_mod, qm);

        /* Left-to-right бинарное возведение. Число итераций определяется
           только k, общим для всего варпа => цикл полностью синхронный. */
        const int kbits = 64 - __clzll(k);
        typename M::T r = b_m;                       /* старший бит k всегда 1 */
        for (int i = kbits - 2; i >= 0; --i) {
            r = M::mul(r, r, qm, ni);
            if ((k >> i) & 1ull) r = M::mul(r, b_m, qm, ni);
        }
        ++local_tested;

        /* Выход из Montgomery-домена: REDC(r·1) = r·R^{-1} = обычное значение.
           Это дешевле, чем считать R mod q через to_mont(1) ради сравнения:
           при k ~ 10^6 сам цикл powmod — всего ~20 умножений, а to_mont для
           128-битной ширины стоит 128 удвоений, то есть setup доминировал бы. */
        const typename M::T one = M::from_u64(1ull);
        const typename M::T r_out = M::mul(r, one, qm, ni);

        /* Делитель принимается только собственный: q < R_k(b). */
        if (M::eq(r_out, one) && repunit_exceeds(base, k, q)) {
            uint32_t slot = atomicAdd(cnt, 1u);
            if (slot < cap) {
                hits[slot].q_lo = q.lo; hits[slot].q_hi = q.hi;
                hits[slot].k_index = ki; hits[slot]._pad = 0u;
            }
        }
    }

    /* Один атомик на поток вместо одного на кандидата. */
    if (local_tested) atomicAdd(tested, local_tested);
}

/*───────────── Три специализации, вызываемые из tf_host.cu ─────────────*/
extern "C" __global__ __launch_bounds__(256, 4)
void rh_tf_k64(uint64_t base, const uint64_t* __restrict__ ks, uint32_t n_k,
               uint64_t m_start, uint64_t m_span,
               rh_tf_hit_t* __restrict__ hits, uint32_t* __restrict__ cnt,
               uint32_t cap, unsigned long long* __restrict__ tested)
{
    tf_body<0, Mont64>(base, ks, n_k, m_start, m_span, hits, cnt, cap, tested);
}

extern "C" __global__ __launch_bounds__(256, 2)
void rh_tf_k96(uint64_t base, const uint64_t* __restrict__ ks, uint32_t n_k,
               uint64_t m_start, uint64_t m_span,
               rh_tf_hit_t* __restrict__ hits, uint32_t* __restrict__ cnt,
               uint32_t cap, unsigned long long* __restrict__ tested)
{
    tf_body<1, Mont96>(base, ks, n_k, m_start, m_span, hits, cnt, cap, tested);
}

extern "C" __global__ __launch_bounds__(256, 2)
void rh_tf_k128(uint64_t base, const uint64_t* __restrict__ ks, uint32_t n_k,
                uint64_t m_start, uint64_t m_span,
                rh_tf_hit_t* __restrict__ hits, uint32_t* __restrict__ cnt,
                uint32_t cap, unsigned long long* __restrict__ tested)
{
    tf_body<2, Mont128>(base, ks, n_k, m_start, m_span, hits, cnt, cap, tested);
}
