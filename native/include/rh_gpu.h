#ifndef RH_GPU_H
#define RH_GPU_H
#include "rh_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rh_gpu_ctx rh_gpu_ctx_t;

typedef struct {
    int      device_id, sm_count, max_threads_per_sm, cc_major, cc_minor;
    int      clock_khz;
    uint64_t global_mem_bytes;
    char     name[128];
} rh_gpu_info_t;

int         rh_gpu_device_count(void);
rh_status_t rh_gpu_query(int device_id, rh_gpu_info_t* info);
rh_status_t rh_gpu_init(int device_id, rh_gpu_ctx_t** out);
void        rh_gpu_destroy(rh_gpu_ctx_t* ctx);

/* Загрузить батч показателей k (до 8192). Инвалидирует кэш CUDA-графов. */
rh_status_t rh_gpu_upload_ks(rh_gpu_ctx_t* ctx, const uint64_t* ks, uint32_t n);

/* Прогнать диапазон m ∈ [m_start, m_start+m_span) для всех загруженных k.
 * width: 0=64bit, 1=96bit, 2=128bit. */
rh_status_t rh_gpu_tf_batch(rh_gpu_ctx_t* ctx, uint64_t base, uint64_t m_start, uint64_t m_span, uint32_t width, rh_tf_result_t* res);

uint64_t    rh_gpu_suggest_span(const rh_gpu_ctx_t* ctx, uint32_t n_k, uint32_t width);

/* Диагностика: сколько раз пересобирался CUDA-граф за время жизни контекста.
 * Ожидаемо — по разу на каждую использованную ширину арифметики; рост этого
 * счётчика с числом запусков означает, что граф не переиспользуется. */
uint64_t    rh_gpu_graph_builds(const rh_gpu_ctx_t* ctx);

#ifdef __cplusplus
}
#endif
#endif