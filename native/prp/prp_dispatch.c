/*==============================================================================
 * prp_dispatch.c — выбор PRP-бэкенда по размеру числа.
 *
 * СТРАТЕГИЯ:
 *   bits <  GW_THRESHOLD (по умолчанию 10 000)  -> GMP (низкий overhead setup)
 *   bits >= GW_THRESHOLD                        -> GWNUM (IBDWT)
 *
 * Двухступенчатая схема:
 *   1) GWNUM делает быстрый Fermat-PRP base 3 (IBDWT + контроль roundoff).
 *   2) Если PRP — GMP полностью пересчитывает кандидата (база 2 + mr_rounds баз)
 *      (это происходит крайне редко, поэтому медленный GMP не мешает).
 *
 * Положительный вердикт GWNUM ВСЕГДА перепроверяется на GMP: это дёшево
 * (PRP-кандидаты редки) и полностью исключает ложное «PRP» из-за ошибки FFT.
 *============================================================================*/

#include "rh_prp.h"
#include <gmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern int     rh_prp_gmp_impl(rh_prp_arena_t*,uint64_t,uint64_t,unsigned,uint32_t,rh_prp_stat_t*);
extern mpz_ptr rh_prp_build_n(rh_prp_arena_t*,uint64_t,uint64_t);

static uint64_t gw_threshold(void) {
    const char* e = getenv("RH_GWNUM_THRESHOLD_BITS");
    return e ? strtoull(e,NULL,10) : 10000ull;
}

int rh_prp_test(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                unsigned mr_rounds, uint32_t backend,
                uint32_t gerbicz_L, rh_prp_stat_t* st)
{
    if(!a || base<2 || k<2) return RH_ERR_INVALID_ARG;

    /* Оценка размера без построения N.
     *
     * R_b(k) = 1 + b + ... + b^(k-1), то есть примерно b^(k-1), откуда
     * bits ≈ (k-1)·log2(b). Прежняя оценка брала lg2b = floor(log2 b) и
     * прибавляла единицу «на округление вверх»: для b = 2 это давало
     * (k-1)·2 вместо (k-1)·1, то есть ЗАВЫШЕНИЕ РОВНО ВДВОЕ, и измеренный
     * порог переключения на GWNUM (2500 бит) фактически применялся как 1250.
     * Считаем честный log2 и округляем вверх на один бит, а не на разряд. */
    double lg2b = log2((double)base);
    uint64_t est_bits = (uint64_t)((double)(k-1)*lg2b) + 1;

    int use_gw = 0;
    if (backend == RH_BACKEND_GWNUM) use_gw = 1;
    else if (backend == RH_BACKEND_AUTO)
        use_gw = rh_gwnum_available() && (est_bits >= gw_threshold());

    /* Явно запрошен GWNUM, но он не собран: молча терять кандидата нельзя,
     * поэтому откатываемся на GMP и предупреждаем ОДИН раз. */
    if (use_gw && !rh_gwnum_available()) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[rh_prp] backend=gwnum запрошен, но бэкенд не собран "
                            "(нужен GWNUM_DIR) — считаем на GMP.\n");
        }
        use_gw = 0;
    }

    if (!use_gw)
        return rh_prp_gmp_impl(a, base, k, mr_rounds, gerbicz_L, st);

    /* ── GWNUM-путь ───────────────────────────────────────────── */
    rh_prp_arena_reserve(a, est_bits + 256);
    mpz_ptr N = rh_prp_build_n(a, base, k);

    /* Дешёвые отсевы до запуска тяжёлого FFT */
    if (mpz_even_p(N)) return 0;

    /* Внутри rh_prp_gwnum уже есть повторы с увеличенным FFT при roundoff. */
    int rc = rh_prp_gwnum((const void*)N, base, k, 0, gerbicz_L, st);
    if (rc == RH_ERR_FFT_ERROR) {
        /* Все четыре попытки с увеличенным FFT упёрлись в roundoff. Вернуть
         * ошибку нельзя: вызывающий (src/pipeline.rs) на ошибке лишь пишет
         * лог и НЕ создаёт записи в журнале — кандидат исчезает из результатов
         * молча. Считаем такой случай точной арифметикой: медленно, но верно. */
        fprintf(stderr, "[rh_prp] GWNUM не уложился в допустимый roundoff "
                        "(b=%llu k=%llu, roundoff=%.4f) — считаем на GMP.\n",
                (unsigned long long)base, (unsigned long long)k,
                st ? st->max_roundoff : 0.0);
        return rh_prp_gmp_impl(a, base, k, mr_rounds, gerbicz_L, st);
    }
    if (rc < 0) return rc;
    if (rc == 0) return 0;

    /* ── Верификация положительного вердикта ──────────────────────────
     * GWNUM даёт Fermat-PRP по базе 3 на быстром FFT, но защищён только
     * контролем roundoff: проверка Гербица к произвольному показателю
     * неприменима задёшево (разбор — в prp_gwnum.c). Поэтому КАЖДЫЙ
     * положительный результат пересчитывается независимым GMP-путём.
     * Статистически это бесплатно: PRP-кандидатов единицы на миллионы
     * проверок, зато ложное «PRP» из-за ошибки FFT исключается полностью.
     * Обратный случай (ложное «составное») ловится только повторным
     * прогоном — как double-check в GIMPS.                              */
    {
        rh_prp_stat_t tmp;
        int r2 = rh_prp_gmp_impl(a, base, k, mr_rounds, gerbicz_L, &tmp);
        if (r2 < 0) return r2;
        if (r2 == 0) {
            fprintf(stderr,
                "[rh_prp] РАСХОЖДЕНИЕ БЭКЕНДОВ: GWNUM сказал PRP, GMP — составное "
                "(b=%llu k=%llu, roundoff=%.4f). Доверяем GMP; проверьте FFT/железо.\n",
                (unsigned long long)base, (unsigned long long)k,
                st ? st->max_roundoff : 0.0);
            return 0;
        }
    }
    return 1;
}