/*==============================================================================
 * rh_common.h — общие ABI-типы между Rust, C и CUDA (v2).
 *
 * ВАЖНО: все структуры зеркалируются на стороне Rust как #[repr(C)].
 * Поля упорядочены по убыванию размера, чтобы padding был предсказуем
 * и одинаков для gcc/clang/nvcc/rustc на всех поддерживаемых ABI.
 *============================================================================*/
#ifndef RH_COMMON_H
#define RH_COMMON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*───────────────────────── Коды возврата ─────────────────────────*/
/* Отрицательные значения зеркалируются в src/ffi/mod.rs::check(). */
typedef enum {
    RH_OK                 =  0,
    RH_ERR_NOMEM          = -1,
    RH_ERR_INVALID_ARG    = -2,
    RH_ERR_CUDA           = -3,
    RH_ERR_NO_DEVICE      = -4,
    RH_ERR_OVERFLOW       = -5,
    RH_ERR_INTERNAL       = -6,
    RH_ERR_FFT_ERROR      = -7,   /* roundoff превысил порог — нужен больший FFT */
    RH_ERR_GERBICZ        = -8,   /* Gerbicz-Li check не сошёлся (сбой железа)  */
    RH_ERR_NO_BACKEND     = -9    /* бэкенд не собран (нет GWNUM / libecm)      */
} rh_status_t;

/*───────────────────────── Выбор PRP-бэкенда ─────────────────────*/
typedef enum {
    RH_BACKEND_AUTO  = 0,
    RH_BACKEND_GMP   = 1,
    RH_BACKEND_GWNUM = 2
} rh_backend_t;

/*───────────────────────── GPU trial factoring ───────────────────*/
/* Ширина модульной арифметики на устройстве. Выбирается хостом по q_max:
 *   q < 2^63  -> 64,  q < 2^95 -> 96,  q < 2^127 -> 128.             */
typedef enum {
    RH_W64  = 0,
    RH_W96  = 1,
    RH_W128 = 2
} rh_width_t;

/* Найденный делитель. q = q_hi*2^64 + q_lo, k = ks[k_index] того батча,
 * который был загружен через rh_gpu_upload_ks(). */
typedef struct {
    uint64_t q_lo;
    uint64_t q_hi;
    uint32_t k_index;
    uint32_t _pad;      /* явный padding: размер 24 байта на всех ABI */
} rh_tf_hit_t;

/* Ёмкость выходного буфера одного запуска.
 *
 * 16 записей оказалось мало: в начале диапазона m (где q лишь немного больше
 * границы CPU-сита) плотность делителей высокая, и батч из сотен показателей
 * даёт десятки попаданий за запуск. На длинном прогоне это привело к 8
 * переполнениям — потерянные делители уходят в PRP, то есть вместо мгновенного
 * отсева тратится полноценный тест на сотню тысяч бит.
 *
 * 256 записей — 6 КиБ на буфер, ничтожно по памяти и с большим запасом
 * относительно наблюдаемого максимума. Остаток потерь виден в поле `lost`. */
#define RH_TF_MAX_FACTORS 256

typedef struct {
    rh_tf_hit_t hits[RH_TF_MAX_FACTORS];
    uint64_t    candidates_tested;  /* сколько q реально прошло powmod —
                                       вход для tuner::observe_gpu()        */
    uint32_t    count;              /* сколько записей в hits[] валидно     */
    uint32_t    lost;               /* сколько попаданий НЕ влезло в буфер   */
} rh_tf_result_t;

/*───────────────────────── Статистика PRP ────────────────────────*/
typedef struct {
    uint64_t bits;            /* размер N в битах                        */
    uint64_t squarings;       /* число возведений в квадрат по модулю    */
    double   elapsed_sec;
    uint32_t backend_used;    /* rh_backend_t, фактически применённый    */
    uint32_t gerbicz_checks;  /* сколько Gerbicz-проверок прошло успешно */
    double   max_roundoff;    /* максимум gw_get_maxerr() (0 для GMP)    */
} rh_prp_stat_t;

/*───────────────────────── Параметры P-1 ─────────────────────────*/
typedef struct {
    uint64_t b1;
    uint64_t b2;
    /* Известный множитель показателя: q-1 = 2*m*k для любого делителя q,
     * поэтому сид берётся как 2^(2k) — см. native/prp/pm1_ecm.c.        */
    uint64_t k_known;
} rh_pm1_params_t;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* RH_COMMON_H */
