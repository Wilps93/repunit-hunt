/*==============================================================================
 * test_mont.cu — автономная проверка GPU-части.
 *
 * ЧАСТЬ 1. Montgomery-арифметика трёх ширин (Mont64 / Mont96 / Mont128):
 *   для каждого вектора считаем b^k mod q ровно так, как это делает ядро
 *   (ninv -> to_mont -> цикл mul -> выход из домена через REDC(r·1)) и
 *   сверяем с эталоном, посчитанным независимо на Python (произвольная точность).
 *
 * ЧАСТЬ 2. Сами ядра rh_tf_k64/k96/k128 на НАСТОЯЩИХ делителях репьюнитов:
 *   узкий диапазон m вокруг известного m => ядро обязано найти ровно этот q.
 *
 * Сборка:
 *   nvcc -std=c++17 -arch=sm_75 -I native/include -I native/cuda \
 *        native/tests/test_mont.cu native/cuda/tf_kernel.cu -o test_mont
 *============================================================================*/

#include "rh_common.h"
#include "rh_mont.cuh"
#include "mont_vectors.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#define CK(e) do { cudaError_t _x=(e); if(_x!=cudaSuccess){                    \
    std::printf("CUDA %s @%d: %s\n", cudaGetErrorName(_x), __LINE__,           \
                cudaGetErrorString(_x)); return 1; } } while(0)

/* Ядра из tf_kernel.cu */
extern "C" __global__ void rh_tf_k64 (uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);
extern "C" __global__ void rh_tf_k96 (uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);
extern "C" __global__ void rh_tf_k128(uint64_t,const uint64_t*,uint32_t,uint64_t,uint64_t,
                                      rh_tf_hit_t*,uint32_t*,uint32_t,unsigned long long*);

/*───────────────── Часть 1: powmod в трёх ширинах ─────────────────*/
template<int W, class M>
__device__ __forceinline__ u128 powmod_as_kernel(uint64_t b, uint64_t k, u128 q)
{
    typename M::T qm;
    if constexpr (W == 0) qm = q.lo;
    else                  qm = q;

    const uint64_t b_mod = (q.hi == 0ull) ? (b % q.lo) : b;
    if (b_mod == 0ull) return mk128(0, 0);

    const typename M::T ni  = M::ninv(qm);
    const typename M::T b_m = M::to_mont(b_mod, qm);

    const int kbits = 64 - __clzll(k);
    typename M::T r = b_m;
    for (int i = kbits - 2; i >= 0; --i) {
        r = M::mul(r, r, qm, ni);
        if ((k >> i) & 1ull) r = M::mul(r, b_m, qm, ni);
    }
    const typename M::T one = M::from_u64(1ull);
    const typename M::T out = M::mul(r, one, qm, ni);
    return mk128(M::hi64(out), M::lo64(out));
}

__global__ void run_vectors(const mont_vec_t* __restrict__ v, int n,
                            uint64_t* __restrict__ out_lo,
                            uint64_t* __restrict__ out_hi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u128 q = mk128(v[i].q_hi, v[i].q_lo);
    u128 r;
    switch (v[i].width) {
        case 0:  r = powmod_as_kernel<0, Mont64 >(v[i].b, v[i].k, q); break;
        case 1:  r = powmod_as_kernel<1, Mont96 >(v[i].b, v[i].k, q); break;
        default: r = powmod_as_kernel<2, Mont128>(v[i].b, v[i].k, q); break;
    }
    out_lo[i] = r.lo;
    out_hi[i] = r.hi;
}

static int test_arithmetic(int* failed)
{
    mont_vec_t* d_v; uint64_t *d_lo, *d_hi;
    CK(cudaMalloc(&d_v, sizeof(g_mont_vecs)));
    CK(cudaMalloc(&d_lo, sizeof(uint64_t) * MONT_VEC_N));
    CK(cudaMalloc(&d_hi, sizeof(uint64_t) * MONT_VEC_N));
    CK(cudaMemcpy(d_v, g_mont_vecs, sizeof(g_mont_vecs), cudaMemcpyHostToDevice));

    run_vectors<<<(MONT_VEC_N + 63) / 64, 64>>>(d_v, MONT_VEC_N, d_lo, d_hi);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    uint64_t lo[MONT_VEC_N], hi[MONT_VEC_N];
    CK(cudaMemcpy(lo, d_lo, sizeof(lo), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hi, d_hi, sizeof(hi), cudaMemcpyDeviceToHost));

    int bad = 0, per_w[3] = {0, 0, 0}, cnt_w[3] = {0, 0, 0};
    for (int i = 0; i < MONT_VEC_N; ++i) {
        const int w = (int)g_mont_vecs[i].width;
        ++cnt_w[w];
        if (lo[i] != g_mont_vecs[i].e_lo || hi[i] != g_mont_vecs[i].e_hi) {
            ++bad; ++per_w[w];
            if (bad <= 6) {
                std::printf("  FAIL w=%d b=%llu k=%llu q=%llu:%llu -> получено %llu:%llu, "
                            "ожидалось %llu:%llu\n", w,
                    (unsigned long long)g_mont_vecs[i].b, (unsigned long long)g_mont_vecs[i].k,
                    (unsigned long long)g_mont_vecs[i].q_hi, (unsigned long long)g_mont_vecs[i].q_lo,
                    (unsigned long long)hi[i], (unsigned long long)lo[i],
                    (unsigned long long)g_mont_vecs[i].e_hi, (unsigned long long)g_mont_vecs[i].e_lo);
            }
        }
    }
    std::printf("  Mont64 : %d/%d ок\n", cnt_w[0] - per_w[0], cnt_w[0]);
    std::printf("  Mont96 : %d/%d ок\n", cnt_w[1] - per_w[1], cnt_w[1]);
    std::printf("  Mont128: %d/%d ок\n", cnt_w[2] - per_w[2], cnt_w[2]);
    cudaFree(d_v); cudaFree(d_lo); cudaFree(d_hi);
    *failed += bad;
    return 0;
}

/*───────────────── Часть 2: настоящие ядра ─────────────────*/
struct KernelCase {
    const char* name;
    int         width;
    uint64_t    base, k, m;      /* известно: q = 2*m*k+1 делит R_k(base) */
    uint64_t    q_lo, q_hi;
};

static int test_kernel(const KernelCase& c, int* failed)
{
    uint64_t* d_ks; rh_tf_hit_t* d_hits; uint32_t* d_cnt; unsigned long long* d_tested;
    CK(cudaMalloc(&d_ks, sizeof(uint64_t)));
    CK(cudaMalloc(&d_hits, sizeof(rh_tf_hit_t) * RH_TF_MAX_FACTORS));
    CK(cudaMalloc(&d_cnt, sizeof(uint32_t)));
    CK(cudaMalloc(&d_tested, sizeof(unsigned long long)));
    CK(cudaMemcpy(d_ks, &c.k, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cnt, 0, sizeof(uint32_t)));
    CK(cudaMemset(d_tested, 0, sizeof(unsigned long long)));

    /* Узкое окно вокруг известного m: ядро обязано найти ровно один делитель */
    const uint64_t m_start = c.m - 3;
    const uint64_t m_span  = 7;

    switch (c.width) {
        case 0: rh_tf_k64 <<<4, 64>>>(c.base, d_ks, 1, m_start, m_span,
                                      d_hits, d_cnt, RH_TF_MAX_FACTORS, d_tested); break;
        case 1: rh_tf_k96 <<<4, 64>>>(c.base, d_ks, 1, m_start, m_span,
                                      d_hits, d_cnt, RH_TF_MAX_FACTORS, d_tested); break;
        default:rh_tf_k128<<<4, 64>>>(c.base, d_ks, 1, m_start, m_span,
                                      d_hits, d_cnt, RH_TF_MAX_FACTORS, d_tested); break;
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    uint32_t n = 0; rh_tf_hit_t hits[RH_TF_MAX_FACTORS]; unsigned long long tested = 0;
    CK(cudaMemcpy(&n, d_cnt, sizeof(n), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hits, d_hits, sizeof(hits), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&tested, d_tested, sizeof(tested), cudaMemcpyDeviceToHost));

    bool found = false;
    for (uint32_t i = 0; i < n && i < RH_TF_MAX_FACTORS; ++i)
        if (hits[i].q_lo == c.q_lo && hits[i].q_hi == c.q_hi && hits[i].k_index == 0)
            found = true;

    std::printf("  %-6s b=%llu k=%llu: попаданий=%u, протестировано=%llu -> %s\n",
                c.name, (unsigned long long)c.base, (unsigned long long)c.k,
                n, tested, found ? "делитель НАЙДЕН" : "ДЕЛИТЕЛЬ ПОТЕРЯН");
    if (!found) ++*failed;

    cudaFree(d_ks); cudaFree(d_hits); cudaFree(d_cnt); cudaFree(d_tested);
    return 0;
}

int main()
{
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("устройство: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    int failed = 0;

    std::printf("=== Часть 1: Montgomery-арифметика против эталона (%d векторов) ===\n",
                MONT_VEC_N);
    if (test_arithmetic(&failed)) return 1;

    std::printf("\n=== Часть 2: ядра на настоящих делителях репьюнитов ===\n");
    /* q = 46922399 = 2*20173*1163 + 1, делит R_1163(10)                       */
    KernelCase c64  { "k64",  0, 10, 1163, 20173ull, 46922399ull, 0ull };
    /* q = 18446744073709551629 = 2*1317624576693539402*7 + 1 (65 бит), простое */
    KernelCase c96  { "k96",  1, 9285164788987252851ull, 7, 1317624576693539402ull,
                      13ull, 1ull };
    /* Тот же 65-битный делитель, но прогнанный через 128-битное ядро: ширина
     * задаёт лишь ВЕРХНЮЮ границу, меньшие q обязаны обрабатываться тоже.
     *
     * Настоящего делителя в «родном» диапазоне Mont128 (q >= 2^95) здесь быть
     * не может: множитель m передаётся как uint64_t, поэтому достижимо только
     * q < 2^65·k, и для q >= 2^95 требуется k > 2^30. Собственно арифметика
     * Mont128 на числах до 2^126 проверяется частью 1.                        */
    KernelCase c128 { "k128", 2, 9285164788987252851ull, 7, 1317624576693539402ull,
                      13ull, 1ull };

    if (test_kernel(c64,  &failed)) return 1;
    if (test_kernel(c96,  &failed)) return 1;
    if (test_kernel(c128, &failed)) return 1;

    std::printf("\n%s (несовпадений: %d)\n",
                failed ? "ЕСТЬ ОШИБКИ" : "ВСЁ СОШЛОСЬ", failed);
    return failed ? 1 : 0;
}
