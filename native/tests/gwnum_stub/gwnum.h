/*==============================================================================
 * gwnum.h — ЗАГЛУШКА, только для компиляционной проверки prp_gwnum.c.
 *
 * ЭТО НЕ GWNUM. Сигнатуры восстановлены по вызовам в native/prp/prp_gwnum.c,
 * а не по официальному SDK, поэтому проверка ловит синтаксис, типы и логику
 * внутри нашего файла — но НЕ соответствие реальному API Woltman'а.
 * Для настоящей сборки нужен Prime95 SDK и GWNUM_DIR.
 *
 * Использование:
 *   gcc -fsyntax-only -DRH_HAVE_GWNUM -I native/include \
 *       -I native/tests/gwnum_stub native/prp/prp_gwnum.c
 *============================================================================*/
#ifndef GWNUM_STUB_H
#define GWNUM_STUB_H

#include <stdint.h>
#include <stdlib.h>

/* ── giant: целое произвольной длины в 32-битных лимбах ── */
typedef struct giantstruct {
    int       sign;     /* число значащих лимбов, знак — знак числа */
    uint32_t* n;        /* лимбы, младший первым                    */
} giantstruct;
typedef struct giantstruct* giant;

static inline giant allocgiant(int len) {
    giant g = (giant)malloc(sizeof(giantstruct));
    if (!g) return 0;
    g->n = (uint32_t*)malloc((size_t)len * sizeof(uint32_t));
    g->sign = 0;
    return g;
}

/* ── основные типы ── */
typedef double* gwnum;

typedef struct {
    int    thread_count;
    int    larger_fftlen_count;
    double maxerr;
    unsigned long bit_length;
} gwhandle;

/* ── опции умножения ── */
#define GWMUL_STARTNEXTFFT 0x0400
#define GWMUL_MULBYCONST   0x0200

/* ── жизненный цикл ── */
static inline void gwinit(gwhandle* gw) { (void)gw; }
static inline void gwdone(gwhandle* gw) { (void)gw; }
static inline void gwset_num_threads(gwhandle* gw, int n) { (void)gw; (void)n; }
static inline void gwset_larger_fftlen_count(gwhandle* gw, char n) { (void)gw; (void)n; }
static inline int  gwsetup_general_mod_giant(gwhandle* gw, giant g) { (void)gw; (void)g; return 0; }
/* Быстрый путь для чисел вида k*b^n+c (у нас b^k - 1) */
static inline int  gwsetup(gwhandle* gw, double k, unsigned long b,
                           unsigned long n, signed long c)
                           { (void)gw; (void)k; (void)b; (void)n; (void)c; return 0; }
static inline void gwsetmulbyconst(gwhandle* gw, long s) { (void)gw; (void)s; }

/* ── память ── */
static inline gwnum gwalloc(gwhandle* gw) { (void)gw; return (gwnum)malloc(sizeof(double)); }
static inline void  gwfree(gwhandle* gw, gwnum x) { (void)gw; free(x); }

/* ── арифметика ── */
static inline void gwsquare2(gwhandle* gw, gwnum s, gwnum d, int opts)
                             { (void)gw; (void)s; (void)d; (void)opts; }
static inline void gwmul3(gwhandle* gw, gwnum a, gwnum b, gwnum d, int opts)
                          { (void)gw; (void)a; (void)b; (void)d; (void)opts; }
static inline void gwsmallmul(gwhandle* gw, double m, gwnum x) { (void)gw; (void)m; (void)x; }
static inline void gwcopy(gwhandle* gw, gwnum s, gwnum d) { (void)gw; (void)s; (void)d; }
static inline int  gwequal(gwhandle* gw, gwnum a, gwnum b) { (void)gw; (void)a; (void)b; return 1; }

/* ── конвертация ── */
static inline void dbltogw(gwhandle* gw, double v, gwnum x) { (void)gw; (void)v; (void)x; }
static inline int  gwtogiant(gwhandle* gw, gwnum x, giant g) { (void)gw; (void)x; (void)g; return 0; }

/* ── контроль ошибок округления ── */
static inline double gw_get_maxerr(gwhandle* gw) { return gw ? gw->maxerr : 0.0; }
static inline void   gw_clear_maxerr(gwhandle* gw) { if (gw) gw->maxerr = 0.0; }
static inline void   gwsetnormroutine(gwhandle* gw, int z, int e, int c)
                                      { (void)gw; (void)z; (void)e; (void)c; }

/* ── прочее ── */
static inline unsigned long gwdata_bits(gwhandle* gw) { return gw ? gw->bit_length : 0; }

#endif /* GWNUM_STUB_H */
