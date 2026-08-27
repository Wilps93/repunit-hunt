/*==============================================================================
 * bench_gpu.cu — измерение накладных расходов GPU-пути.
 *
 * Вопрос, на который отвечает тест: переиспользуется ли CUDA-граф между
 * запусками? Конвейер увеличивает m на каждом запуске, и если m_start входит
 * в сигнатуру графа, тот пересоздаётся каждый раз.
 *
 * Меряем два режима при прочих равных:
 *   A. m_start меняется (как в реальном конвейере);
 *   B. m_start фиксирован (граф заведомо переиспользуется).
 * Разница = стоимость пересоздания.
 *
 * Частоты GPU под WSL заметно плавают, поэтому берём МИНИМУМ по сериям:
 * среднее по одной серии шумит на десятки процентов.
 *
 * Сборка:
 *   nvcc -std=c++17 -arch=sm_75 -O2 -I native/include -I native/cuda \
 *        native/tests/bench_gpu.cu native/cuda/tf_host.cu native/cuda/tf_kernel.cu \
 *        -o bench_gpu
 *============================================================================*/

#include "rh_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 20;
    const uint32_t n_k = (argc > 2) ? (uint32_t)atoi(argv[2]) : 256;
    const uint64_t k0  = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 100003;

    rh_gpu_ctx_t* ctx = nullptr;
    if (rh_gpu_init(0, &ctx) != RH_OK) { std::printf("init failed\n"); return 1; }

    rh_gpu_info_t info;
    rh_gpu_query(0, &info);
    std::printf("устройство: %s, SM=%d\n", info.name, info.sm_count);

    /* Батч показателей, похожий на реальный: простые около 100000 */
    std::vector<uint64_t> ks;
    for (uint64_t k = k0 | 1; ks.size() < n_k; k += 2) {
        bool prime = true;
        for (uint64_t d = 3; d * d <= k; d += 2) if (k % d == 0) { prime = false; break; }
        if (prime) ks.push_back(k);
    }
    if (rh_gpu_upload_ks(ctx, ks.data(), (uint32_t)ks.size()) != RH_OK) {
        std::printf("upload failed\n"); return 1;
    }

    const uint64_t span = rh_gpu_suggest_span(ctx, (uint32_t)ks.size(), RH_W64);
    std::printf("n_k=%zu, span=%llu, итераций в серии=%d\n\n",
                ks.size(), (unsigned long long)span, iters);

    rh_tf_result_t res;
    const uint64_t base = 10;
    const uint64_t m0 = 1000000;

    /* прогрев: частоты карты должны выйти на рабочие */
    for (int i = 0; i < 8; ++i) rh_gpu_tf_batch(ctx, base, m0 + i * span, span, RH_W64, &res);

    const int series = 5;
    double best_a = 1e9, best_b = 1e9;
    unsigned long long tested = 0;

    for (int s_i = 0; s_i < series; ++s_i) {
        double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            rh_gpu_tf_batch(ctx, base, m0 + (uint64_t)i * span, span, RH_W64, &res);
            tested = res.candidates_tested;
        }
        double ta = (now_ms() - t0) / iters;
        if (ta < best_a) best_a = ta;

        double t1 = now_ms();
        for (int i = 0; i < iters; ++i)
            rh_gpu_tf_batch(ctx, base, m0, span, RH_W64, &res);
        double tb = (now_ms() - t1) / iters;
        if (tb < best_b) best_b = tb;

        std::printf("  серия %d: A=%7.3f  B=%7.3f мс\n", s_i + 1, ta, tb);
    }

    /* Прямое доказательство переиспользования графа: счётчик пересборок.
     * Косвенный замер времени тут бесполезен — разброс частот под WSL
     * (37..46 мс между сериями) перекрывает любой эффект. */
    const uint64_t builds = rh_gpu_graph_builds(ctx);
    const int total_launches = 8 + series * iters * 2;
    std::printf("\nпересборок графа: %llu на %d запусков\n",
                (unsigned long long)builds, total_launches);
    std::printf("%s\n", builds <= 2
        ? "=> граф строится один раз и переиспользуется"
        : "=> ГРАФ ПЕРЕСОБИРАЕТСЯ, оптимизация не работает");

    const double diff_pct = best_b > 0 ? (best_a - best_b) / best_b * 100.0 : 0.0;
    std::printf("\nлучшее из %d серий по %d запусков:\n", series, iters);
    std::printf("A. m_start меняется  : %8.3f мс/launch\n", best_a);
    std::printf("B. m_start фиксирован: %8.3f мс/launch\n", best_b);
    std::printf("разница: %+.3f мс (%.1f%%)\n", best_a - best_b, diff_pct);
    std::printf("(разброс между сериями под WSL достигает 20%%, поэтому вывод "
                "делается по счётчику пересборок, а не по времени)\n");
    std::printf("пропускная способность: %.1f млн кандидатов/с (после отсева малыми простыми)\n",
                tested / (best_a / 1000.0) / 1e6);

    rh_gpu_destroy(ctx);
    return 0;
}
