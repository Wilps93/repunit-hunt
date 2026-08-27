/*==============================================================================
 * prp_gwnum.c — PRP через GWNUM (Woltman, Prime95 SDK).
 *
 * ЗАЧЕМ: GWNUM реализует IBDWT (Irrational Base Discrete Weighted Transform)
 * с AVX-512/FMA. Для чисел > 10 000 бит он в 20-60 раз быстрее mpz_powm,
 * а для > 1 Мбит — более чем в 100 раз.
 *
 * ФОРМА ЧИСЛА: R_k(b) не имеет вида k·b^n+c, поэтому используем
 * gwsetup_general_mod — GWNUM применяет IBDWT-умножение + редукцию
 * Барретта по произвольному модулю.
 *
 * КОНТРОЛЬ ОШИБОК: только roundoff (см. подробный разбор перед gw_prp_base3 —
 * проверка Гербица к произвольному показателю E = N-1 неприменима задёшево).
 * Положительный вердикт дополнительно верифицируется GMP в prp_dispatch.c.
 *
 * ROUNDOFF CHECK: GWNUM даёт gw_get_maxerr(). Порог 0.40 — консервативный;
 * при превышении переключаемся на больший FFT-размер (gwsetup с
 * увеличенным safety margin) и перезапускаем.
 *============================================================================*/

#include "rh_prp.h"

#ifndef RH_HAVE_GWNUM
int rh_gwnum_available(void) { return 0; }
int rh_prp_gwnum(const void* N, uint64_t base, uint64_t k,
                 unsigned r, uint32_t L, rh_prp_stat_t* s) {
    (void)N;(void)base;(void)k;(void)r;(void)L;(void)s; return RH_ERR_NO_BACKEND;
}
#else

#include <gmp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "gwnum.h"
#include "gwcommon.h"
#include "gwutil.h"

int rh_gwnum_available(void) { return 1; }

#define ROUNDOFF_LIMIT 0.40

/*──────────── Вспомогательное: gwnum -> mpz ────────────
 * Размер буфера берётся так же, как это делает сам Prime95 (gwtest.c):
 *   allocgiant (((unsigned long) gwdata.bit_length >> 5) + 10)
 * Запас обязателен: gwtogiant использует giant как рабочую память. */
static void gw_to_mpz(gwhandle* gw, gwnum x, mpz_t out) {
    const int len = (int)(((unsigned long)gw->bit_length >> 5) + 10);
    giant g = allocgiant(len);
    if (!g) { mpz_set_ui(out, 0); return; }
    if (gwtogiant(gw, x, g) < 0) {          /* 0 = успех, <0 = ошибка */
        mpz_set_ui(out, 0);
    } else {
        mpz_import(out, (size_t)abs(g->sign), -1, sizeof(uint32_t), 0, 0, g->n);
        if (g->sign < 0) mpz_neg(out, out);
    }
    free(g);
}

/*==============================================================================
 * Основная процедура: Fermat-PRP по базе 3 через IBDWT.
 *
 * ПОЧЕМУ ЗДЕСЬ НЕТ ПРОВЕРКИ GERBICZ–LI
 * ------------------------------------
 * Проверка Гербица применима, когда показатель даёт ДЛИННУЮ ЦЕПОЧКУ КВАДРАТОВ.
 * Сам автор GWNUM формулирует это ограничение прямо (Prime95, commonb.c):
 * «We can do Gerbicz error checking if b=2 and there are a long string of
 * squarings». Для чисел вида k·2^n показатель специально подбирают так, чтобы
 * умножений на малые константы почти не было.
 *
 * У нас же N = R_k(b), показатель E = N-1 — произвольная битовая строка, и
 * цепочка имеет вид x <- x^2 · 3^{бит}. Для неё инвариант Гербица принимает вид
 *     d_{j+1} = d_j^{2^L} · 3^{S_j},   S_j — сумма L-битных кусков E,
 * и его проверка стоит ~2L операций на каждые L итераций, то есть больше 100%
 * накладных расходов. Дешёвого варианта для произвольного показателя нет.
 *
 * Предыдущая реализация в этом файле сравнивала bk^(2^L) с текущим x, между
 * которыми были умножения на 3, — то есть проверяла неверное равенство. При
 * этом она вообще никогда не запускалась: условие blocks_done % L == 0
 * наступает лишь после L² ≈ 4·10^6 итераций, а практические числа здесь на
 * два порядка меньше. Замеры это подтверждали: gerbicz_checks = 0 всегда.
 *
 * ЧТО ЗАЩИЩАЕТ ВЫЧИСЛЕНИЕ ВМЕСТО НЕЁ
 *   1. Контроль ошибки округления FFT (gw_get_maxerr) каждые 128 итераций:
 *      при превышении порога вызывающий перезапускает счёт с большим FFT.
 *   2. Верификация ПОЛОЖИТЕЛЬНОГО вердикта независимым GMP-путём
 *      (см. prp_dispatch.c): PRP-кандидаты редки, поэтому полная перепроверка
 *      почти ничего не стоит и полностью исключает ложное «PRP».
 *   3. От ложного «составное» (потерянная находка) защищает только повторный
 *      счёт — это стандартная практика GIMPS (double-check), см. README.
 *
 * Возврат: 1 = PRP, 0 = составное, RH_ERR_FFT_ERROR при недопустимом roundoff.
 *============================================================================*/
static int gw_prp_base3(mpz_srcptr N, uint64_t base, uint64_t k,
                        uint32_t L, rh_prp_stat_t* st, int retry_margin)
{
    (void)L;                                      /* длина блока больше не нужна */

    gwhandle gw;
    gwinit(&gw);
    gwset_num_threads(&gw, 1);                    /* по 1 потоку на кандидата:
                                                     параллелизм даёт rayon    */
    gwset_larger_fftlen_count(&gw, (char)retry_margin);  /* запас при roundoff */

    /* ── Выбор модуля ─────────────────────────────────────────────────
     * Считаем не по R_k(b), а по b^k - 1, которое на него делится:
     *
     *     x ≡ 3^E (mod b^k - 1)  =>  x mod R_k = 3^E mod R_k.
     *
     * Причина — форма числа. Для произвольного модуля GWNUM применяет
     * gwsetup_general_mod, то есть IBDWT плюс редукция Барретта; для
     * 1·b^k + (-1) редукция почти бесплатна. Замер на GTX-машине
     * (native/tests/bench_gwform.c, b = 10):
     *     k =  5003  general 0.008 мс/итер, специальная 0.003  (2.81x)
     *     k = 10007  general 0.015,          специальная 0.005  (2.73x)
     *     k = 20011  general 0.042,          специальная 0.015  (2.79x)
     * Плата — модуль длиннее на log2(b-1) бит (для b = 10 это 3 бита).
     *
     * Если форма не поддержана (слишком большое основание, экзотические k),
     * откатываемся на прежний general_mod: корректность важнее скорости. */
    giant gN = NULL;
    int special = 0;

    if (base >= 2 && k >= 2 && base <= 0xFFFFFFFFull && k <= 0x7FFFFFFFull) {
        special = (gwsetup(&gw, 1.0, (unsigned long)base, (unsigned long)k, -1) == 0);
    }

    if (!special) {
        size_t nlimbs32 = (mpz_sizeinbase(N,2) + 31) / 32 + 2;
        gN = allocgiant((int)nlimbs32);
        if (!gN) { gwdone(&gw); return RH_ERR_NOMEM; }
        {
            size_t written = 0;
            mpz_export(gN->n, &written, -1, sizeof(uint32_t), 0, 0, N);
            gN->sign = (int)written;
        }
        if (gwsetup_general_mod_giant(&gw, gN)) {
            free(gN); gwdone(&gw); return RH_ERR_INTERNAL;
        }
    }

    const unsigned long ebits = (unsigned long)mpz_sizeinbase(N,2);   /* E = N-1 */

    gwnum x = gwalloc(&gw);
    if (!x) { free(gN); gwdone(&gw); return RH_ERR_NOMEM; }

    /* E = N-1; идём от старшего бита. Старший бит E — всегда 1. */
    mpz_t E; mpz_init(E); mpz_sub_ui(E, N, 1);
    long i = (long)mpz_sizeinbase(E,2) - 2;       /* старший бит учтён в x = 3 */

    dbltogw(&gw, 3.0, x);
    gw_clear_maxerr(&gw);
    gwsetnormroutine(&gw, 0, 1 /*error checking on*/, 0);

    /* Умножение на базу делаем ШТАТНЫМ механизмом GWNUM: константа задаётся
     * один раз, а применяется прямо внутри возведения в квадрат по флагу
     * GWMUL_MULBYCONST.
     *
     * Прежний gwsmallmul отдельной операцией молча портил результат на
     * больших числах. Замер (b = 10, сверка с GMP на 12 итерациях):
     *     66 476 бит — совпадает, roundoff 0.0002
     *    163 044 бит — РАСХОЖДЕНИЕ, roundoff 0.5
     *    332 203 бит — РАСХОЖДЕНИЕ, roundoff 0.5
     * при том, что чистые квадраты на тех же размерах считались верно.
     * С GWMUL_MULBYCONST всё сходится вплоть до 664 396 бит (roundoff 0.0015),
     * и одна операция вместо двух заодно экономит время. */
    gwsetmulbyconst(&gw, 3);

    unsigned long iters = 0;
    double maxerr = 0.0;
    int rc = 0;

    while (i >= 0) {
        int opts = GWMUL_STARTNEXTFFT;
        if (mpz_tstbit(E, (mp_bitcnt_t)i)) opts |= GWMUL_MULBYCONST;
        gwsquare2(&gw, x, x, opts);
        --i; ++iters;

        /* Контроль roundoff каждые 128 итераций */
        if ((iters & 127) == 0) {
            double e = gw_get_maxerr(&gw);
            if (e > maxerr) maxerr = e;
            if (e > ROUNDOFF_LIMIT) { rc = RH_ERR_FFT_ERROR; break; }
        }
    }

    if (rc == 0) {
        double e = gw_get_maxerr(&gw);
        if (e > maxerr) maxerr = e;

        /* ФИНАЛЬНАЯ ПРОВЕРКА ROUNDOFF — ОБЯЗАТЕЛЬНА.
         *
         * Периодическая проверка внутри цикла срабатывает раз в 128 итераций,
         * поэтому скачок ошибки на одной из последних 127 итераций остаётся
         * незамеченным, и вердикт «составное» возвращается по заведомо
         * испорченному остатку. Это ПОТЕРЯННАЯ НАХОДКА: ложное «PRP» ловится
         * перепроверкой в prp_dispatch.c, а ложное «составное» — ничем.
         *
         * Реальный случай: b = 23, k = 3181 (14 385 бит). Ошибка впервые
         * превышает порог на итерации 14383 из 14384, то есть после
         * последней периодической проверки на 14336-й. maxerr = 0.5000
         * (максимум из возможных — остаток целиком мусор), GWNUM возвращал
         * «составное», тогда как R_3181(23) — PRP (член A204940). С margin = 1
         * FFT растёт с 768 до 1024, maxerr падает до 0.0015 и вердикт верен.
         *
         * Возврат RH_ERR_FFT_ERROR заставляет rh_prp_gwnum пересчитать с
         * увеличенным FFT (до 4 попыток), как и при срабатывании внутри цикла. */
        if (maxerr > ROUNDOFF_LIMIT) {
            if (st) st->max_roundoff = maxerr;
            rc = RH_ERR_FFT_ERROR;
            goto done;
        }

        mpz_t res; mpz_init(res);
        gw_to_mpz(&gw, x, res);
        /* Здесь и происходит переход от большего модуля к нужному:
           x взято по модулю b^k-1, а R_k его делит. */
        mpz_mod(res, res, N);
        rc = (mpz_cmp_ui(res, 1) == 0) ? 1 : 0;
        mpz_clear(res);

        if (st) {
            st->bits           = (uint64_t)mpz_sizeinbase(N,2);
            st->squarings      = (uint64_t)ebits;
            st->backend_used   = RH_BACKEND_GWNUM;
            st->gerbicz_checks = 0;               /* см. комментарий выше */
            st->max_roundoff   = maxerr;
        }
    }

done:
    mpz_clear(E);
    gwfree(&gw, x);
    free(gN);
    gwdone(&gw);
    return rc;
}

int rh_prp_gwnum(const void* N_mpz, uint64_t base, uint64_t k,
                 unsigned mr_rounds, uint32_t L, rh_prp_stat_t* st)
{
    (void)mr_rounds;   /* GWNUM-путь: только Fermat base 3; довешивание баз
                          делается GMP-бэкендом на прошедших кандидатах  */
    mpz_srcptr N = (mpz_srcptr)N_mpz;
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);

    int rc = RH_ERR_FFT_ERROR;
    for (int margin = 0; margin <= 3 && rc == RH_ERR_FFT_ERROR; ++margin) {
        rc = gw_prp_base3(N, base, k, L, st, margin);
    }

    clock_gettime(CLOCK_MONOTONIC,&t1);
    if (st) st->elapsed_sec = (t1.tv_sec-t0.tv_sec)+1e-9*(t1.tv_nsec-t0.tv_nsec);
    return rc;
}
#endif
