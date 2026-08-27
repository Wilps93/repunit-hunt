/*==============================================================================
 * bench_gwform.c — стоит ли считать PRP по модулю b^k - 1 вместо R_k(b)?
 *
 * ИДЕЯ. Сейчас GWNUM работает через gwsetup_general_mod: универсальная
 * редукция Барретта для произвольного модуля. Но R_k(b) делит b^k - 1, а это
 * число вида 1·b^k + (-1) — «родная» форма для gwsetup, где редукция почти
 * бесплатна. Считать можно по большему модулю и привести в самом конце:
 *
 *     x ≡ 3^E (mod b^k - 1)   =>   x mod R_k = 3^E mod R_k,
 *
 * поскольку R_k | b^k - 1. Плата — число длиннее на log2(b-1) бит.
 *
 * Тест меряет ЧИСТУЮ стоимость итерации (gwsquare2) в обоих режимах плюс
 * время самого gwsetup.
 *
 * Сборка:
 *   gcc -O2 -o bench_gwform native/tests/bench_gwform.c \
 *       -I native/include -I $GWNUM_DIR/gwnum \
 *       $GWNUM_DIR/gwnum/gwnum.a -lgmp -lm -lpthread -lstdc++ -no-pie
 *============================================================================*/

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gwnum.h"
#include "gwcommon.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Сколько секунд занимает одна итерация «квадрат + умножение на 3». */
static double bench_iters(gwhandle *gw, int iters) {
    gwnum x = gwalloc(gw);
    if (!x) return -1.0;
    dbltogw(gw, 3.0, x);
    /* прогрев */
    for (int i = 0; i < 20; ++i) gwsquare2(gw, x, x, GWMUL_STARTNEXTFFT);

    double t0 = now();
    for (int i = 0; i < iters; ++i) {
        gwsquare2(gw, x, x, GWMUL_STARTNEXTFFT);
        gwsmallmul(gw, 3.0, x);
    }
    double dt = now() - t0;
    gwfree(gw, x);
    return dt / iters;
}

int main(int argc, char **argv) {
    const unsigned long base = (argc > 1) ? strtoul(argv[1], NULL, 10) : 10;
    const unsigned long k    = (argc > 2) ? strtoul(argv[2], NULL, 10) : 5003;
    const int iters          = (argc > 3) ? atoi(argv[3]) : 300;

    /* N = (b^k - 1)/(b - 1) */
    mpz_t N, t;
    mpz_init(N); mpz_init(t);
    mpz_ui_pow_ui(t, base, k);
    mpz_sub_ui(t, t, 1);
    mpz_divexact_ui(N, t, base - 1);
    printf("b=%lu k=%lu | R_k = %zu бит, b^k-1 = %zu бит | итераций %d\n\n",
           base, k, mpz_sizeinbase(N, 2), mpz_sizeinbase(t, 2), iters);

    /* ── Режим 1: general mod по R_k (как сейчас) ── */
    double setup1, per1;
    {
        gwhandle gw;
        gwinit(&gw);
        gwset_num_threads(&gw, 1);
        size_t words = (mpz_sizeinbase(N, 2) + 31) / 32 + 2;
        giant gN = allocgiant((int)words);
        size_t written = 0;
        mpz_export(gN->n, &written, -1, sizeof(uint32_t), 0, 0, N);
        gN->sign = (int)written;

        double s = now();
        int err = gwsetup_general_mod_giant(&gw, gN);
        setup1 = now() - s;
        if (err) { printf("general_mod: ошибка setup %d\n", err); return 1; }
        printf("general mod (R_k):     setup %6.3f с", setup1);
        per1 = bench_iters(&gw, iters);
        printf(" | %8.3f мс/итерацию\n", per1 * 1e3);
        free(gN);
        gwdone(&gw);
    }

    /* ── Режим 2: специальная форма 1·b^k - 1 ── */
    double setup2, per2;
    {
        gwhandle gw;
        gwinit(&gw);
        gwset_num_threads(&gw, 1);
        double s = now();
        int err = gwsetup(&gw, 1.0, base, k, -1);
        setup2 = now() - s;
        if (err) { printf("special form: ошибка setup %d (форма не поддержана)\n", err); return 1; }
        printf("специальная (b^k - 1): setup %6.3f с", setup2);
        per2 = bench_iters(&gw, iters);
        printf(" | %8.3f мс/итерацию\n", per2 * 1e3);
        gwdone(&gw);
    }

    printf("\nускорение на итерации: %.2fx\n", per1 / per2);
    printf("итераций в тесте ≈ %zu, экономия ≈ %.1f с на число\n",
           mpz_sizeinbase(N, 2), (per1 - per2) * mpz_sizeinbase(N, 2));

    mpz_clear(N); mpz_clear(t);
    return 0;
}
