#ifndef RH_PRP_H
#define RH_PRP_H
#include "rh_common.h"
#ifdef __cplusplus
extern "C" {
#endif

/* ── Pool allocator (rh_alloc.c) ────────────────────────────── */
void   rh_gmp_pool_install(void);
void   rh_gmp_pool_reset(void);
void   rh_gmp_pool_release(void);
size_t rh_gmp_pool_hiwater(void);
size_t rh_gmp_pool_capacity(void);

/* ── Арена mpz_t ────────────────────────────────────────────── */
typedef struct rh_prp_arena rh_prp_arena_t;
rh_prp_arena_t* rh_prp_arena_new(void);
void            rh_prp_arena_free(rh_prp_arena_t*);
void            rh_prp_arena_reserve(rh_prp_arena_t*, uint64_t bits);

/* ── Диспетчер PRP ──────────────────────────────────────────── */
/* Возврат: 1 = PRP, 0 = составное, <0 = rh_status_t.
 * backend: RH_BACKEND_AUTO выбирает GWNUM при bits >= gwnum_threshold. */
int rh_prp_test(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                unsigned mr_rounds, uint32_t backend,
                uint32_t gerbicz_L, rh_prp_stat_t* stat);

/* GMP-бэкенд (prp_gmp.c). Возврат: 1 = PRP, 0 = составное, <0 = rh_status_t. */
int rh_prp_gmp_impl(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                    unsigned mr_rounds, uint32_t gerbicz_L, rh_prp_stat_t* stat);

/* Верификация делителя, найденного GPU (128-битный q). */
int rh_prp_verify_factor(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                         uint64_t q_lo, uint64_t q_hi);

/* Десятичное представление N. */
size_t rh_prp_decimal(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                      char* buf, size_t buflen);

/* Внутреннее: построить N в слоте арены и вернуть mpz_ptr.
 * НЕ объявляем здесь: gmp.h типизирует mpz_ptr как указатель на анонимную
 * структуру, поэтому объявление без gmp.h конфликтует с определением.
 * Потребители (prp_dispatch.c, pm1_ecm.c) объявляют его локально после
 * #include <gmp.h>:   extern mpz_ptr rh_prp_build_n(rh_prp_arena_t*,uint64_t,uint64_t);
 */

/* ── GWNUM-бэкенд (prp_gwnum.c), доступен при RH_HAVE_GWNUM ─── */
/* base и k нужны, чтобы считать по модулю b^k-1 — числа специальной формы,
 * для которой GWNUM обходится без редукции Барретта (см. prp_gwnum.c). */
int rh_prp_gwnum(const void* N_mpz, uint64_t base, uint64_t k,
                 unsigned mr_rounds, uint32_t gerbicz_L, rh_prp_stat_t* stat);
int rh_gwnum_available(void);

/* ── P-1 (pm1_ecm.c) ────────────────────────────────────────── */
/* Возврат: 1 = найден множитель (записан в factor_dec, десятичная строка),
 *          0 = не найден, <0 = ошибка. */
int rh_pm1_factor(rh_prp_arena_t* a, uint64_t base, uint64_t k,
                  const rh_pm1_params_t* p, char* factor_dec, size_t len);
int rh_pm1_available(void);

#ifdef __cplusplus
}
#endif
#endif