/*==============================================================================
 *  prp_gmp.c — GMP-бэкенд PRP для R_k(b) = (b^k - 1)/(b - 1)   [v2]
 *
 *  АЛГОРИТМ
 *  --------
 *  1) N = (b^k - 1)/(b - 1) строится ТОЧНО: mpz_pow_ui + mpz_divexact.
 *     divexact использует exact division (Jebelean) и в разы быстрее tdiv.
 *     Результат кэшируется в арене: повторный вызов для той же пары (b,k)
 *     — например verify_factor сразу после PRP — не пересчитывает N.
 *
 *  2) Дешёвые отсевы:
 *       - чётность;
 *       - ОДИН gcd с primorial (простые 3..997) вместо 167 отдельных делений.
 *
 *  3) Сильный тест Ферма по базе 2 (Miller-Rabin, a=2):
 *         N-1 = d*2^s,  x = a^d mod N,  затем до s-1 возведений в квадрат.
 *     Стоимость ~ log2(N) умножений по модулю; отсекает всё, кроме
 *     псевдопростых по базе 2.
 *
 *  4) mr_rounds дополнительных раундов с фиксированными базами 3,5,7,...
 *
 *  ПАМЯТЬ
 *  ------
 *  Все mpz_t — слоты арены (rh_arena.c), инициализированные один раз на поток.
 *  В этом файле НЕТ ни одного mpz_init/mpz_clear. Внутренние временные буферы
 *  GMP идут через pool-аллокатор (rh_alloc.c).
 *============================================================================*/

#include "rh_prp.h"
#include <gmp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Импорт из rh_arena.c ─────────────────────────────────────── */
extern mpz_ptr rh_slot(rh_prp_arena_t* a, int i);
extern char*   rh_sbuf(rh_prp_arena_t* a, size_t need);
extern int     rh_cache_hit(rh_prp_arena_t* a, uint64_t b, uint64_t k);
extern void    rh_cache_set(rh_prp_arena_t* a, uint64_t b, uint64_t k);
extern void    rh_cache_clear(rh_prp_arena_t* a);

/* ── Карта слотов арены (RH_SLOTS = 16) ───────────────────────── */
enum {
    S_N    = 0,   /* кандидат N (кэшируется)      */
    S_D    = 1,   /* d из N-1 = d*2^s             */
    S_X    = 2,   /* текущее значение powm        */
    S_NM1  = 3,   /* N-1                          */
    S_TMP  = 4,   /* временное                    */
    S_BASE = 5,   /* база MR                      */
    S_PRIM = 6,   /* primorial (кэш на арену)     */
    S_G    = 7,   /* gcd                          */
    S_POW  = 8,   /* b^k                          */
    S_BM1  = 9,   /* b-1                          */
    S_Q    = 10,  /* делитель для verify / P-1    */
    S_R    = 11   /* остаток                      */
    /* 12,13 — резерв; 14 — S_TMP2 (используется pm1_ecm.c); 15 — резерв */
};

/* ── Портируемая установка 64-битного значения ─────────────────
 * unsigned long — 32 бита на LLP64 (Windows), поэтому не полагаемся на
 * mpz_set_ui для значений > 2^32. */
static void set_u64(mpz_ptr r, uint64_t v) {
    mpz_import(r, 1, -1, sizeof(uint64_t), 0, 0, &v);
    if (v == 0) mpz_set_ui(r, 0);
}

static void set_u128(mpz_ptr r, uint64_t lo, uint64_t hi) {
    uint64_t w[2] = { lo, hi };
    mpz_import(r, 2, -1, sizeof(uint64_t), 0, 0, w);
    if (lo == 0 && hi == 0) mpz_set_ui(r, 0);
}

/*==============================================================================
 * Построение N = (b^k - 1)/(b - 1) с кэшем в арене.
 * Возвращает указатель на слот S_N.
 *============================================================================*/
mpz_ptr rh_prp_build_n(rh_prp_arena_t* a, uint64_t base, uint64_t k) {
    mpz_ptr N = rh_slot(a, S_N);
    if (rh_cache_hit(a, base, k)) return N;

    mpz_ptr pw  = rh_slot(a, S_POW);
    mpz_ptr bm1 = rh_slot(a, S_BM1);

    set_u64(bm1, base);
    mpz_pow_ui(pw, bm1, (unsigned long)k);       /* b^k, бинарное возведение */
    mpz_sub_ui(pw, pw, 1);                       /* b^k - 1                  */
    set_u64(bm1, base - 1);                      /* b-1 (base >= 2)          */
    mpz_divexact(N, pw, bm1);                    /* деление точное           */

    rh_cache_set(a, base, k);
    return N;
}

/*==============================================================================
 * primorial: произведение простых 3..997. Один gcd вместо 167 делений —
 * для N > ~10k бит это в 5-10 раз дешевле.
 *============================================================================*/
static const unsigned SMALL_P[] = {
    3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,
    101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,
    191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,
    281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,
    389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,
    491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,
    607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,
    719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,
    829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,
    953,967,971,977,983,991,997
};
#define SMALL_P_N (sizeof(SMALL_P)/sizeof(SMALL_P[0]))

static void ensure_primorial(rh_prp_arena_t* a) {
    mpz_ptr P = rh_slot(a, S_PRIM);
    if (mpz_sgn(P) != 0) return;                 /* уже построен для этой арены */
    mpz_set_ui(P, 1);
    for (size_t i = 0; i < SMALL_P_N; ++i)
        mpz_mul_ui(P, P, SMALL_P[i]);
}

/*==============================================================================
 * Один раунд Miller-Rabin: n-1 = d*2^s, d нечётно.
 * 1 = прошёл (вероятно простое), 0 = составное. Аллокаций нет.
 *============================================================================*/
static int mr_round(rh_prp_arena_t* a, mpz_srcptr n, mpz_srcptr d,
                    unsigned long s, mpz_srcptr base_val)
{
    mpz_ptr x   = rh_slot(a, S_X);
    mpz_ptr nm1 = rh_slot(a, S_NM1);
    mpz_ptr t   = rh_slot(a, S_TMP);

    /* Самая дорогая часть: ~log2(d) умножений по модулю.
       mpz_powm внутри сам выбирает Montgomery/REDC. */
    mpz_powm(x, base_val, d, n);

    if (mpz_cmp_ui(x, 1) == 0) return 1;
    if (mpz_cmp(x, nm1) == 0)  return 1;

    for (unsigned long r = 1; r < s; ++r) {
        /* x = x^2 mod n через mul+mod (дешевле, чем mpz_powm_ui(...,2,...)) */
        mpz_mul(t, x, x);
        mpz_mod(x, t, n);
        if (mpz_cmp(x, nm1) == 0)  return 1;
        if (mpz_cmp_ui(x, 1) == 0) return 0;     /* нетривиальный корень из 1 */
    }
    return 0;
}

/*==============================================================================
 * Основной вход GMP-бэкенда (вызывается из prp_dispatch.c).
 * Возврат: 1 = PRP, 0 = составное, <0 = rh_status_t.
 *============================================================================*/
int rh_prp_gmp_impl(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                    unsigned mr_rounds, uint32_t gerbicz_L, rh_prp_stat_t* st)
{
    (void)gerbicz_L;                              /* Gerbicz — только в GWNUM-пути */
    if (!a || base < 2 || k < 2) return RH_ERR_INVALID_ARG;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (st) memset(st, 0, sizeof(*st));

    /* Резервируем лимбы под ожидаемый размер: bits(N) ≈ k*log2(b). */
    {
        double lg = 0.0; uint64_t bb = base;
        while (bb > 1) { lg += 1.0; bb >>= 1; }   /* floor(log2 b) */
        rh_prp_arena_reserve(a, (uint64_t)((double)k * (lg + 1.0)) + 64);
    }

    mpz_ptr N   = rh_prp_build_n(a, base, k);
    mpz_ptr d   = rh_slot(a, S_D);
    mpz_ptr nm1 = rh_slot(a, S_NM1);
    mpz_ptr bv  = rh_slot(a, S_BASE);
    mpz_ptr g   = rh_slot(a, S_G);

    const uint64_t bits = (uint64_t)mpz_sizeinbase(N, 2);
    if (st) { st->bits = bits; st->backend_used = RH_BACKEND_GMP; }

    int verdict;

    /* ── Тривиальные случаи ──
     * Порог именно 4: для N < 4 не существует базы из допустимого диапазона
     * [2, N-2], и любой раунд Miller-Rabin был бы бессмысленным. */
    if (mpz_cmp_ui(N, 4) < 0) {
        verdict = (mpz_cmp_ui(N, 2) == 0) || (mpz_cmp_ui(N, 3) == 0);
        goto done;
    }
    if (mpz_even_p(N))        { verdict = 0; goto done; }

    /* ── Отсев малыми простыми одним gcd ── */
    ensure_primorial(a);
    mpz_gcd(g, N, rh_slot(a, S_PRIM));
    if (mpz_cmp_ui(g, 1) != 0 && mpz_cmp(g, N) != 0) { verdict = 0; goto done; }
    /* g == N возможно только для малого N — его добьёт Miller-Rabin. */

    /* ── N-1 = d * 2^s ── */
    mpz_sub_ui(nm1, N, 1);
    {
        unsigned long s = mpz_scan1(nm1, 0);       /* младшие нулевые биты */
        mpz_tdiv_q_2exp(d, nm1, s);

        /* Раунд 1: база 2 — сильный тест Ферма. */
        mpz_set_ui(bv, 2);
        if (!mr_round(a, N, d, s, bv)) { verdict = 0; goto done; }

        /* Дополнительные фиксированные базы. Набор выбран так, чтобы
           совместные псевдопростые были известны/исключены (Feitsma, Galway).

           База обязана лежать в [2, N-2]: при a ≡ 0, 1, -1 (mod N) раунд не
           несёт информации, а a = N (например N = 3, a = 3) объявил бы
           простое число составным — этим теряются малые репьюниты
           вроде R_2(2) = 3 и R_3(2) = 7. */
        {
            static const unsigned long extra[] = {3,5,7,11,13,17,19,23,29,31,37};
            unsigned n_extra = mr_rounds;
            if (n_extra > sizeof(extra)/sizeof(extra[0]))
                n_extra = (unsigned)(sizeof(extra)/sizeof(extra[0]));
            for (unsigned i = 0; i < n_extra; ++i) {
                if (mpz_cmp_ui(N, extra[i] + 2) < 0) break;   /* база > N-2 */
                mpz_set_ui(bv, extra[i]);
                if (!mr_round(a, N, d, s, bv)) { verdict = 0; goto done; }
            }
        }
    }
    verdict = 1;

done:
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (st) {
        st->elapsed_sec = (double)(t1.tv_sec - t0.tv_sec)
                        + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
        /* Каждый раунд MR — примерно log2(N) возведений в квадрат. */
        st->squarings = bits * (uint64_t)(1 + (verdict ? mr_rounds : 0));
    }
    return verdict;
}

/*==============================================================================
 * Верификация делителя, найденного на GPU (страховка от багов ядра).
 * q = q_hi*2^64 + q_lo.
 *
 * Требуется СОБСТВЕННЫЙ делитель: q < N. Иначе простой репьюнит «отсеивался»
 * бы самим собой — например R_2(10) = 11 при q = 11 = 2·1·5+1.
 *============================================================================*/
int rh_prp_verify_factor(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                         uint64_t q_lo, uint64_t q_hi)
{
    if (!a) return RH_ERR_INVALID_ARG;
    if (q_hi == 0 && q_lo < 2) return RH_ERR_INVALID_ARG;

    mpz_ptr N = rh_prp_build_n(a, base, k);
    mpz_ptr Q = rh_slot(a, S_Q);
    set_u128(Q, q_lo, q_hi);
    if (mpz_cmp(Q, N) >= 0) return 0;             /* сам кандидат — не делитель */
    return mpz_divisible_p(N, Q) ? 1 : 0;
}

/*==============================================================================
 * Десятичное представление N (для отчёта о находке).
 * buf == NULL  =>  вернуть требуемую длину без учёта '\0'.
 *============================================================================*/
size_t rh_prp_decimal(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                      char* buf, size_t buflen)
{
    if (!a) return 0;
    mpz_ptr N = rh_prp_build_n(a, base, k);
    size_t need = mpz_sizeinbase(N, 10) + 2;      /* запас на знак и '\0' */
    if (!buf || buflen < need) return need - 1;

    char* tmp = rh_sbuf(a, need);
    if (!tmp) return 0;
    mpz_get_str(tmp, 10, N);
    size_t len = strlen(tmp);
    memcpy(buf, tmp, len + 1);
    return len;
}
