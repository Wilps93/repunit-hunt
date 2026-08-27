/*==============================================================================
 * pm1_ecm.c — этап P-1 (Полларда) между trial factoring и PRP.
 *
 * МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ ДЛЯ РЕПЬЮНИТОВ:
 * -------------------------------------------
 * Каждый простой делитель q числа R_k(b) имеет вид q = 2·m·k + 1.
 * Значит q - 1 = 2·m·k УЖЕ содержит известный крупный множитель k.
 * Это даёт огромное преимущество для P-1:
 *   • Стандартный P-1 считает a^(lcm(1..B1)) mod N.
 *   • Мы стартуем с a^(2k) — то есть «бесплатно» получаем множитель k
 *     в показателе, и B1 нужен только чтобы покрыть гладкую часть m.
 *   • Эффективно: делители с гладким m находятся при B1 в ~k раз меньшем.
 *
 * Реализация: libecm (GMP-ECM) с ECM_PM1, начальное значение x0 = 2^(2k).
 *
 * ЭКОНОМИКА: при B1=10^5, B2=10^7 стоимость ≈ 1% от PRP,
 * вероятность найти множитель ≈ 3-6% => ожидаемая экономия 2-5%.
 *============================================================================*/

#include "rh_prp.h"

#ifndef RH_HAVE_ECM
int rh_pm1_available(void) { return 0; }
int rh_pm1_factor(rh_prp_arena_t* a,uint64_t b,uint64_t k,
                  const rh_pm1_params_t* p,char* out,size_t n) {
    (void)a;(void)b;(void)k;(void)p;(void)out;(void)n; return RH_ERR_NO_BACKEND;
}
#else

#include <gmp.h>
#include <ecm.h>
#include <string.h>
#include <stdio.h>

extern mpz_ptr rh_prp_build_n(rh_prp_arena_t*,uint64_t,uint64_t);
extern mpz_ptr rh_slot(rh_prp_arena_t*,int);

int rh_pm1_available(void) { return 1; }

int rh_pm1_factor(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                  const rh_pm1_params_t* prm, char* out, size_t outlen)
{
    if(!a||!prm) return RH_ERR_INVALID_ARG;

    mpz_ptr N = rh_prp_build_n(a, base, k);

    /* Слоты арены под факторы (не аллоцируем новые mpz_t) */
    mpz_ptr f  = rh_slot(a, 10);   /* S_Q переиспользуем */
    mpz_ptr x0 = rh_slot(a, 14);   /* S_TMP2 */
    mpz_set_ui(f, 0);

    ecm_params p;
    ecm_init(p);
    p->method  = ECM_PM1;
    p->verbose = 0;
    mpz_set_ui(p->B2, (unsigned long)prm->b2);

    /* ── Ключевая оптимизация: сид x0 = 2^(2k) mod N ──────────────
     * Так мы «бесплатно» вносим известный множитель 2k в показатель,
     * поскольку для любого делителя q справедливо q-1 = 2·m·k.        */
    if (prm->k_known >= 2) {
        mpz_set_ui(x0, 2);
        mpz_powm_ui(x0, x0, (unsigned long)(2*prm->k_known), N);
        mpz_set(p->x, x0);
    }

    int res = ecm_factor(f, N, (double)prm->b1, p);
    ecm_clear(p);

    if (res > 0 && mpz_cmp_ui(f,1) > 0 && mpz_cmp(f,N) != 0) {
        if (out && outlen > mpz_sizeinbase(f,10)+1) mpz_get_str(out,10,f);
        return 1;
    }
    if (res < 0) return RH_ERR_INTERNAL;
    return 0;
}
#endif