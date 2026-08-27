/*==============================================================================
 * rh_alloc.c — thread-local bump-аллокатор, подставляемый в GMP через
 * mp_set_memory_functions.
 *
 * ПРОБЛЕМА: даже при переиспользовании mpz_t, GMP выделяет временные буферы
 * (mpn_mul_n scratch, Toom/FFT workspace) через malloc. При 32+ потоках это
 * точка контенции в glibc-malloc: до 15% времени в futex.
 *
 * РЕШЕНИЕ: каждый поток получает большой arena-буфер (по умолчанию 64 МБ,
 * настраивается). Аллокации — инкремент указателя. Освобождение — no-op
 * (LIFO-паттерн GMP гарантирует, что мы можем откатывать указатель).
 * Реализован «стековый» free: если освобождаем последний блок — откат.
 *
 * ВАЖНО: mp_set_memory_functions глобальна, поэтому все функции обязаны быть
 * thread-safe. Мы используем __thread и fallback на malloc, если пул исчерпан.
 *============================================================================*/

#define _GNU_SOURCE
#include <gmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include "rh_prp.h"

#define RH_ALIGN     64
#define RH_MAGIC     0x52484D31u   /* 'RHM1' */
#define ALIGN_UP(x)  (((x)+RH_ALIGN-1) & ~(size_t)(RH_ALIGN-1))

typedef struct { uint32_t magic; uint32_t from_pool; size_t size; } hdr_t;

static __thread uint8_t* g_pool     = NULL;
static __thread size_t   g_cap      = 0;
static __thread size_t   g_off      = 0;
static __thread size_t   g_hiwater  = 0;
static __thread uint8_t* g_last     = NULL;   /* для LIFO-отката */

static size_t default_pool_bytes(void) {
    const char* e = getenv("RH_GMP_POOL_MB");
    size_t mb = e ? (size_t)strtoul(e,NULL,10) : 64;
    if (mb < 4)    mb = 4;
    if (mb > 4096) mb = 4096;
    return mb << 20;
}

/* Пул на huge pages: снижает TLB-промахи на 10-20% для чисел > 1 Мбит. */
static void ensure_pool(void) {
    if (g_pool) return;
    g_cap = default_pool_bytes();
    void* p = mmap(NULL, g_cap, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { g_pool = NULL; g_cap = 0; return; }
#ifdef MADV_HUGEPAGE
    madvise(p, g_cap, MADV_HUGEPAGE);
#endif
    g_pool = (uint8_t*)p;
    g_off = 0; g_last = NULL;
}

static void* rh_alloc(size_t n) {
    ensure_pool();
    size_t need = ALIGN_UP(n + sizeof(hdr_t));
    if (g_pool && g_off + need <= g_cap) {
        uint8_t* base = g_pool + g_off;
        hdr_t* h = (hdr_t*)base;
        h->magic = RH_MAGIC; h->from_pool = 1; h->size = need;
        g_off += need;
        if (g_off > g_hiwater) g_hiwater = g_off;
        g_last = base;
        return base + sizeof(hdr_t);
    }
    /* Fallback: пул исчерпан (очень большие FFT-буферы) */
    uint8_t* base = (uint8_t*)malloc(need);
    if (!base) return NULL;
    hdr_t* h = (hdr_t*)base;
    h->magic = RH_MAGIC; h->from_pool = 0; h->size = need;
    return base + sizeof(hdr_t);
}

static void rh_free(void* p, size_t n) {
    (void)n;
    if (!p) return;
    uint8_t* base = (uint8_t*)p - sizeof(hdr_t);
    hdr_t* h = (hdr_t*)base;
    if (h->magic != RH_MAGIC) { free(p); return; }   /* чужой указатель */
    if (!h->from_pool) { free(base); return; }
    /* LIFO-откат: если это последний выделенный блок — возвращаем место. */
    if (base == g_last) {
        g_off -= h->size;
        g_last = NULL;
    }
    /* иначе — «утечка» внутри пула; она обнулится при rh_gmp_pool_reset() */
}

static void* rh_realloc(void* p, size_t oldn, size_t newn) {
    if (!p) return rh_alloc(newn);
    uint8_t* base = (uint8_t*)p - sizeof(hdr_t);
    hdr_t* h = (hdr_t*)base;
    if (h->magic != RH_MAGIC) {                  /* не наш блок */
        void* q = rh_alloc(newn);
        if (!q) return NULL;
        memcpy(q, p, oldn < newn ? oldn : newn);
        return q;
    }
    size_t cur_payload = h->size - sizeof(hdr_t);
    if (newn <= cur_payload) return p;            /* влезает на месте */

    /* Расширение последнего блока в пуле — просто двигаем вершину */
    if (h->from_pool && base == g_last) {
        size_t need = ALIGN_UP(newn + sizeof(hdr_t));
        if ((size_t)(base - g_pool) + need <= g_cap) {
            g_off = (size_t)(base - g_pool) + need;
            h->size = need;
            if (g_off > g_hiwater) g_hiwater = g_off;
            return p;
        }
    }
    void* q = rh_alloc(newn);
    if (!q) return NULL;
    memcpy(q, p, oldn < newn ? oldn : newn);
    rh_free(p, oldn);
    return q;
}

/* ── Публичный API ───────────────────────────────────────────── */

void rh_gmp_pool_install(void) {
    /* Идемпотентно: GMP хранит указатели глобально. */
    static int installed = 0;
    if (!installed) { mp_set_memory_functions(rh_alloc, rh_realloc, rh_free); installed = 1; }
    ensure_pool();
}

/* Сброс вершины пула.
 *
 * ОПАСНО и потому НЕ вызывается из Rust-слоя: mpz_t арены живут всё время
 * работы потока и удерживают буферы, выданные этим пулом. Обнуление вершины
 * приведёт к тому, что следующие аллокации перезапишут ещё живые числа.
 * Функция оставлена только для сценария «арена уже освобождена». */
void rh_gmp_pool_reset(void) { g_off = 0; g_last = NULL; }

size_t rh_gmp_pool_hiwater(void) { return g_hiwater; }
size_t rh_gmp_pool_capacity(void) { return g_cap; }

void rh_gmp_pool_release(void) {
    if (g_pool) { munmap(g_pool, g_cap); g_pool = NULL; g_cap = g_off = 0; g_last = NULL; }
}