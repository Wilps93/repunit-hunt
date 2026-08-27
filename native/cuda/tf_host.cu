/*==============================================================================
 * tf_host.cu — host-обвязка GPU trial factoring.
 *
 * УСТРОЙСТВО
 *  1. Persistent device buffers: массив ks (до MAX_K) и буферы попаданий
 *     аллоцируются один раз на контекст — в горячем цикле аллокаций нет.
 *  2. Occupancy-driven grid sizing отдельно для каждой ширины арифметики.
 *  3. CUDA Graph на запуск: memset ×2 → kernel → memcpy ×3 уходят на
 *     устройство ОДНИМ сабмитом.
 *
 * ПОЧЕМУ ГРАФ ЗДЕСЬ ВАЖЕН (и почему он строится вручную)
 * -----------------------------------------------------
 * Замер на GTX 1650 под WSL2 (native/tests/bench_gpu.cu, n_k=256):
 *     граф (даже с пересозданием) — 37.5 мс/launch
 *     обычные async-вызовы        — 51.5 мс/launch
 * То есть +37% времени. Причина в среде: под WSL2 GPU паравиртуализован, и
 * каждый вызов драйвера идёт через границу VM. Пять отдельных вызовов на
 * запуск стоят заметно дороже одного сабмита графа.
 *
 * Раньше граф захватывался через stream capture, и его сигнатура включала
 * m_start. Конвейер увеличивает m на каждом запуске, поэтому граф
 * пересоздавался КАЖДЫЙ раз (захват + инстанцирование ≈ 0.1 мс). Теперь граф
 * строится вручную (cudaGraphAdd*Node), хэндл узла ядра сохраняется, а между
 * запусками меняются только параметры — cudaGraphExecKernelNodeSetParams.
 * Пересборка нужна лишь при смене ширины арифметики или размера батча.
 *
 * Аргументы ядра живут в полях контекста: cudaGraphExecKernelNodeSetParams
 * читает значения по указателям в момент вызова, поэтому они обязаны
 * пережить построение графа.
 *============================================================================*/

#include "rh_gpu.h"
#include "rh_common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

extern "C" __global__ void rh_tf_k64 (uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);
extern "C" __global__ void rh_tf_k96 (uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);
extern "C" __global__ void rh_tf_k128(uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);

#define CK(e) do { cudaError_t _x=(e); if(_x!=cudaSuccess){                       \
    std::fprintf(stderr,"[rh_gpu] %s @%s:%d : %s\n",cudaGetErrorName(_x),         \
                 __FILE__,__LINE__,cudaGetErrorString(_x)); return RH_ERR_CUDA;}} while(0)

#define MAX_K 8192

struct rh_gpu_ctx {
    int          device_id;
    int          sm_count;
    int          block[3];              /* blockDim по ширине */
    int          grid[3];
    uint64_t*    d_ks;
    uint64_t*    h_ks;                  /* pinned staging */
    uint32_t     n_k_cur;
    uint32_t     cap;

    cudaStream_t stream;
    rh_tf_hit_t*        d_hits;
    uint32_t*           d_cnt;
    unsigned long long* d_tested;
    rh_tf_hit_t*        h_hits;         /* pinned */
    uint32_t*           h_cnt;
    unsigned long long* h_tested;

    /* Граф: строится под конкретную (ширину, n_k), переиспользуется дальше */
    cudaGraph_t     graph;
    cudaGraphExec_t graph_exec;
    cudaGraphNode_t kernel_node;
    int             graph_width;        /* -1 = графа нет */
    uint32_t        graph_n_k;

    /* Аргументы ядра — адресуются графом, поэтому живут здесь */
    uint64_t arg_base, arg_m_start, arg_m_span;

    /* Диагностика: сколько раз граф пришлось построить заново.
     * В норме — по одному разу на каждую использованную ширину. */
    uint64_t graph_builds;
};

/*───────────────────────── device query ─────────────────────────*/
extern "C" int rh_gpu_device_count(void) {
    int n=0; if (cudaGetDeviceCount(&n)!=cudaSuccess) return 0; return n;
}

extern "C" rh_status_t rh_gpu_query(int dev, rh_gpu_info_t* info) {
    if(!info) return RH_ERR_INVALID_ARG;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    info->device_id=dev; info->sm_count=p.multiProcessorCount;
    info->max_threads_per_sm=p.maxThreadsPerMultiProcessor;
    info->cc_major=p.major; info->cc_minor=p.minor;
    info->global_mem_bytes=(uint64_t)p.totalGlobalMem;
    info->clock_khz = p.clockRate;
    std::snprintf(info->name,sizeof(info->name),"%s",p.name);
    return RH_OK;
}

static int pick_best(void) {
    int n=0; if(cudaGetDeviceCount(&n)!=cudaSuccess||n==0) return -1;
    int best=0,bs=-1;
    for(int i=0;i<n;++i){ cudaDeviceProp p;
        if(cudaGetDeviceProperties(&p,i)!=cudaSuccess) continue;
        if(p.multiProcessorCount>bs){bs=p.multiProcessorCount;best=i;} }
    return best;
}

static const void* kern_ptr(int w) {
    switch(w){ case 0: return (const void*)rh_tf_k64;
               case 1: return (const void*)rh_tf_k96;
               default:return (const void*)rh_tf_k128; }
}

/*───────────────────────── init / destroy ─────────────────────────*/
extern "C" rh_status_t rh_gpu_init(int dev, rh_gpu_ctx_t** out) {
    if(!out) return RH_ERR_INVALID_ARG;
    *out=nullptr;
    if(dev<0){ dev=pick_best(); if(dev<0) return RH_ERR_NO_DEVICE; }

    rh_gpu_ctx_t* c=(rh_gpu_ctx_t*)std::calloc(1,sizeof(rh_gpu_ctx_t));
    if(!c) return RH_ERR_NOMEM;
    c->device_id=dev; c->cap=RH_TF_MAX_FACTORS; c->graph_width=-1;

    if(cudaSetDevice(dev)!=cudaSuccess){ std::free(c); return RH_ERR_CUDA; }

    /* Синхронизация — БЛОКИРУЮЩАЯ, а не spin-wait.
     *
     * По умолчанию CUDA крутит cudaStreamSynchronize в активном ожидании, и
     * поток-владелец контекста занимает целое ядро вхолостую. Здесь это прямо
     * бьёт по делу: PRP-стадия загружает все ядра, и лишний крутящийся поток
     * отбирает у неё время. Замер на k = 9000..18000 (числа ~45 000 бит):
     * при активной GPU-стадии тесты шли 30.5 мкс/бит против 22.9 мкс/бит,
     * когда GPU простаивал, — то есть весь выигрыш от отсева съедался.
     * BlockingSync усыпляет поток на время счёта и возвращает ядро счёту.
     *
     * Ошибку не проверяем: если контекст уже создан, флаг просто не применится
     * (cudaErrorSetOnActiveProcess) — это не повод отказываться от работы. */
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaDeviceProp p;
    if(cudaGetDeviceProperties(&p,dev)!=cudaSuccess){ std::free(c); return RH_ERR_CUDA; }
    c->sm_count=p.multiProcessorCount;

    /* Occupancy-driven config для каждой ширины */
    for(int w=0;w<3;++w){
        int mg=0, bs=0;
        if(cudaOccupancyMaxPotentialBlockSize(&mg,&bs,kern_ptr(w),0,0)!=cudaSuccess) bs=256;
        if(bs>256) bs=256;                       /* согласовано с launch_bounds */
        c->block[w]=bs;
        int bpsm=0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm,kern_ptr(w),bs,0);
        if(bpsm<=0) bpsm=4;
        /* 4x oversubscription сглаживает неравномерность из-за early-continue */
        c->grid[w]=bpsm*c->sm_count*4;
    }

    if(cudaMalloc(&c->d_ks,sizeof(uint64_t)*MAX_K)!=cudaSuccess ||
       cudaHostAlloc(&c->h_ks,sizeof(uint64_t)*MAX_K,cudaHostAllocDefault)!=cudaSuccess){
        rh_gpu_destroy(c); return RH_ERR_NOMEM; }

    if(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking)!=cudaSuccess ||
       cudaMalloc(&c->d_hits,sizeof(rh_tf_hit_t)*c->cap)!=cudaSuccess ||
       cudaMalloc(&c->d_cnt,sizeof(uint32_t))!=cudaSuccess ||
       cudaMalloc(&c->d_tested,sizeof(unsigned long long))!=cudaSuccess ||
       cudaHostAlloc(&c->h_hits,sizeof(rh_tf_hit_t)*c->cap,cudaHostAllocDefault)!=cudaSuccess ||
       cudaHostAlloc(&c->h_cnt,sizeof(uint32_t),cudaHostAllocDefault)!=cudaSuccess ||
       cudaHostAlloc(&c->h_tested,sizeof(unsigned long long),cudaHostAllocDefault)!=cudaSuccess)
    { rh_gpu_destroy(c); return RH_ERR_NOMEM; }

    *out=c; return RH_OK;
}

static void destroy_graph(rh_gpu_ctx_t* c) {
    if(c->graph_exec){ cudaGraphExecDestroy(c->graph_exec); c->graph_exec=nullptr; }
    if(c->graph){ cudaGraphDestroy(c->graph); c->graph=nullptr; }
    c->kernel_node=nullptr;
    c->graph_width=-1;
}

extern "C" void rh_gpu_destroy(rh_gpu_ctx_t* c) {
    if(!c) return;
    cudaSetDevice(c->device_id);
    destroy_graph(c);
    if(c->d_hits)   cudaFree(c->d_hits);
    if(c->d_cnt)    cudaFree(c->d_cnt);
    if(c->d_tested) cudaFree(c->d_tested);
    if(c->h_hits)   cudaFreeHost(c->h_hits);
    if(c->h_cnt)    cudaFreeHost(c->h_cnt);
    if(c->h_tested) cudaFreeHost(c->h_tested);
    if(c->stream)   cudaStreamDestroy(c->stream);
    if(c->d_ks)     cudaFree(c->d_ks);
    if(c->h_ks)     cudaFreeHost(c->h_ks);
    std::free(c);
}

extern "C" uint64_t rh_gpu_suggest_span(const rh_gpu_ctx_t* c, uint32_t n_k, uint32_t width) {
    if(!c) return 1u<<16;
    if(width>2) width=2;
    /* Целимся в ~190 задач на поток => десятки мс на запуск: накладные расходы
       сабмита размазываются, но отклик остаётся приемлемым. */
    uint64_t threads=(uint64_t)c->grid[width]*(uint64_t)c->block[width];
    uint64_t total_work=threads*192ull;
    uint64_t span = n_k ? total_work/n_k : total_work;
    if(span<1024) span=1024;
    return span;
}

/*───────────────────────── загрузка массива k ─────────────────────────*/
extern "C" rh_status_t rh_gpu_upload_ks(rh_gpu_ctx_t* c, const uint64_t* ks, uint32_t n) {
    if(!c||!ks||n==0||n>MAX_K) return RH_ERR_INVALID_ARG;
    CK(cudaSetDevice(c->device_id));
    std::memcpy(c->h_ks,ks,sizeof(uint64_t)*n);
    CK(cudaMemcpy(c->d_ks,c->h_ks,sizeof(uint64_t)*n,cudaMemcpyHostToDevice));
    c->n_k_cur=n;
    /* n_k зашит в параметры узла ядра — граф под старый батч больше не годится */
    if(c->graph_n_k != n) destroy_graph(c);
    return RH_OK;
}

/*───────────────────────── построение графа ─────────────────────────*/
static rh_status_t build_graph(rh_gpu_ctx_t* c, int w)
{
    destroy_graph(c);
    CK(cudaGraphCreate(&c->graph, 0));

    /* два узла обнуления счётчиков */
    cudaGraphNode_t n_cnt, n_tested;
    cudaMemsetParams mp{};
    mp.dst = c->d_cnt; mp.value = 0; mp.elementSize = 4; mp.width = 1; mp.height = 1;
    CK(cudaGraphAddMemsetNode(&n_cnt, c->graph, nullptr, 0, &mp));
    mp.dst = c->d_tested; mp.width = 2;          /* 8 байт = 2 слова по 4 */
    CK(cudaGraphAddMemsetNode(&n_tested, c->graph, nullptr, 0, &mp));

    /* узел ядра; указатели на аргументы обязаны пережить построение графа */
    void* args[] = { &c->arg_base, (void*)&c->d_ks, &c->n_k_cur,
                     &c->arg_m_start, &c->arg_m_span,
                     &c->d_hits, &c->d_cnt, &c->cap, &c->d_tested };
    cudaKernelNodeParams kp{};
    kp.func           = (void*)kern_ptr(w);
    kp.gridDim        = dim3(c->grid[w]);
    kp.blockDim       = dim3(c->block[w]);
    kp.sharedMemBytes = 0;
    kp.kernelParams   = args;
    kp.extra          = nullptr;

    cudaGraphNode_t deps[2] = { n_cnt, n_tested };
    CK(cudaGraphAddKernelNode(&c->kernel_node, c->graph, deps, 2, &kp));

    /* выгрузка результатов */
    cudaGraphNode_t n_c1, n_c2, n_c3;
    CK(cudaGraphAddMemcpyNode1D(&n_c1, c->graph, &c->kernel_node, 1,
                                c->h_cnt, c->d_cnt, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
    CK(cudaGraphAddMemcpyNode1D(&n_c2, c->graph, &c->kernel_node, 1,
                                c->h_tested, c->d_tested, sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost));
    CK(cudaGraphAddMemcpyNode1D(&n_c3, c->graph, &c->kernel_node, 1,
                                c->h_hits, c->d_hits, sizeof(rh_tf_hit_t)*c->cap,
                                cudaMemcpyDeviceToHost));

    CK(cudaGraphInstantiate(&c->graph_exec, c->graph, nullptr, nullptr, 0));
    ++c->graph_builds;
    c->graph_width = w;
    c->graph_n_k   = c->n_k_cur;
    return RH_OK;
}

/*───────────────────────── основной запуск ─────────────────────────*/
extern "C" rh_status_t rh_gpu_tf_batch(rh_gpu_ctx_t* c,
                                       uint64_t base,
                                       uint64_t m_start,
                                       uint64_t m_span,
                                       uint32_t width,
                                       rh_tf_result_t* res)
{
    if(!c||!res||m_span==0||c->n_k_cur==0) return RH_ERR_INVALID_ARG;
    if(width>2) return RH_ERR_INVALID_ARG;

    CK(cudaSetDevice(c->device_id));

    /* Параметры запуска — в поля, на которые смотрит узел ядра */
    c->arg_base    = base;
    c->arg_m_start = m_start;
    c->arg_m_span  = m_span;

    if(c->graph_width != (int)width || c->graph_n_k != c->n_k_cur) {
        rh_status_t st = build_graph(c, (int)width);
        if(st != RH_OK) return st;
    } else {
        /* Граф уже есть: обновляем только параметры узла ядра.
           Это единственная причина, по которой граф вообще имеет смысл —
           иначе пересборка съедала бы выигрыш от единого сабмита. */
        void* args[] = { &c->arg_base, (void*)&c->d_ks, &c->n_k_cur,
                         &c->arg_m_start, &c->arg_m_span,
                         &c->d_hits, &c->d_cnt, &c->cap, &c->d_tested };
        cudaKernelNodeParams kp{};
        kp.func           = (void*)kern_ptr((int)width);
        kp.gridDim        = dim3(c->grid[width]);
        kp.blockDim       = dim3(c->block[width]);
        kp.sharedMemBytes = 0;
        kp.kernelParams   = args;
        kp.extra          = nullptr;
        CK(cudaGraphExecKernelNodeSetParams(c->graph_exec, c->kernel_node, &kp));
    }

    CK(cudaGraphLaunch(c->graph_exec, c->stream));
    CK(cudaStreamSynchronize(c->stream));
    CK(cudaGetLastError());

    uint32_t n=*c->h_cnt;                 /* сколько ядро НАШЛО (может быть > cap) */
    res->lost = (n>c->cap) ? (n - c->cap) : 0u;
    if(n>c->cap) n=c->cap;
    res->count=n;
    res->candidates_tested = *c->h_tested;
    if(n) std::memcpy(res->hits,c->h_hits,sizeof(rh_tf_hit_t)*n);
    if(n<RH_TF_MAX_FACTORS)
        std::memset(res->hits+n,0,sizeof(rh_tf_hit_t)*(RH_TF_MAX_FACTORS-n));
    return RH_OK;
}

extern "C" uint64_t rh_gpu_graph_builds(const rh_gpu_ctx_t* c) {
    return c ? c->graph_builds : 0;
}
