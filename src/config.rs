use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    // ── Задача ──────────────────────────────────────────────
    pub base: u64,
    pub k_min: u64,
    pub k_max: u64,

    // ── Сито ────────────────────────────────────────────────
    /// Верхняя граница малых простых q для битового сита
    pub bitsieve_q_limit: u64,

    // ── Trial factoring ─────────────────────────────────────
    pub tf_enabled: bool,
    pub tf_q_min: u64,
    /// Жёсткий потолок; фактическая граница подбирается тюнером
    pub tf_q_hard_max_bits: u32,
    pub tf_adaptive: bool,
    /// Сколько k подавать на GPU одним батчем
    pub tf_k_batch: usize,

    // ── P-1 ─────────────────────────────────────────────────
    pub pm1_enabled: bool,
    pub pm1_b1: u64,
    pub pm1_b2: u64,
    pub pm1_adaptive: bool,

    // ── PRP ─────────────────────────────────────────────────
    pub mr_rounds: u32,
    /// Доля вердиктов «составное», пересчитываемых вторым, независимым
    /// бэкендом (0.0 = выключено, 1.0 = каждый). Защита от ложного
    /// «составное» — потерянной находки; см. README, раздел 1.4.
    pub double_check_ratio: f64,
    /// "auto" | "gmp" | "gwnum"
    pub prp_backend: String,
    pub gwnum_threshold_bits: u64,
    /// Длина блока Gerbicz (0 = авто ≈ sqrt(bits))
    pub gerbicz_l: u32,

    // ── Ресурсы ─────────────────────────────────────────────
    pub threads: usize,
    pub pin_threads: bool,
    pub gmp_pool_mb: usize,
    pub queue_depth: usize,
    pub gpu_devices: Vec<i32>,   // пусто = все

    // ── I/O ─────────────────────────────────────────────────
    pub worklog: String,
    pub results: String,
    pub checkpoint_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            base: 10, k_min: 3, k_max: 1_000_000,
            // 2^27. Замер (k=5000..15000): предел 2^24 снимает 30.1% за 0.2 с,
            // 2^27 — 33.1% за 2.25 с и 275 МиБ. Три лишних процентных пункта
            // стоят двух секунд, а каждый снятый кандидат экономит PRP-тест
            // в секунды. Дальше (2^28) отдача падает, а память удваивается.
            bitsieve_q_limit: 1 << 27,
            tf_enabled: true,
            // Нижняя граница TF: всё, что ниже, уже закрыто ситом (см. q_min()).
            tf_q_min: 1 << 27,
            tf_q_hard_max_bits: 80,              // до 2^80 — 128-битная ветка
            tf_adaptive: true,
            tf_k_batch: 4096,
            // Выключен по умолчанию: при размерах до нескольких миллионов бит
            // этап дороже PRP-теста, который должен экономить (см. Tuner::
            // pm1_worth_it). Включать имеет смысл на очень больших k.
            pm1_enabled: false,
            pm1_b1: 100_000, pm1_b2: 10_000_000, pm1_adaptive: true,
            mr_rounds: 2,
            double_check_ratio: 0.0,
            prp_backend: "auto".into(),
            // Точка, где GWNUM обгоняет GMP, измерена (bench kernels):
            //   1017 бит — 0.10x (GWNUM медленнее из-за gwsetup)
            //   1994 бит — 1.06x (паритет)
            //   3010 бит — 1.95x, 5017 — 3.21x, 9966 — 4.31x
            // Прежние 10 000 оставляли весь диапазон 2000..10000 бит на GMP,
            // теряя там от двух до четырёх раз. Берём 2500 — сразу за точкой
            // пересечения, с небольшим запасом на разброс.
            gwnum_threshold_bits: 2_500,
            gerbicz_l: 0,
            threads: 0, pin_threads: true, gmp_pool_mb: 64,
            queue_depth: 8192, gpu_devices: vec![],
            worklog: "worklog.jsonl".into(),
            results: "results.json".into(),
            checkpoint_secs: 60,
        }
    }
}

impl Config {
    pub fn load(p: &Path) -> anyhow::Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(p)?)?)
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.base >= 2, "base должно быть >= 2");
        anyhow::ensure!(
            self.k_min >= 2 && self.k_max > self.k_min,
            "нужен непустой диапазон k: 2 <= k_min < k_max"
        );
        // 2^127 — потолок 128-битной Montgomery-арифметики на устройстве.
        anyhow::ensure!(
            (24..=127).contains(&self.tf_q_hard_max_bits),
            "tf_q_hard_max_bits должно быть в 24..=127"
        );
        // 8192 — MAX_K в native/cuda/tf_host.cu.
        anyhow::ensure!(
            (1..=8192).contains(&self.tf_k_batch),
            "tf_k_batch должно быть в 1..=8192"
        );
        anyhow::ensure!(self.bitsieve_q_limit >= 2, "bitsieve_q_limit >= 2");
        anyhow::ensure!(
            matches!(self.prp_backend.as_str(), "auto" | "gmp" | "gwnum"),
            "prp_backend: auto | gmp | gwnum"
        );
        anyhow::ensure!(self.queue_depth >= 1, "queue_depth >= 1");
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.double_check_ratio),
            "double_check_ratio должно быть в 0.0..=1.0"
        );
        anyhow::ensure!(
            !self.pm1_enabled || self.pm1_b2 >= self.pm1_b1,
            "P-1: требуется B2 >= B1"
        );
        Ok(())
    }

    /// Экспорт настроек, которые нативный слой читает из окружения.
    /// Вызывать ОДИН РАЗ до старта рабочих потоков.
    pub fn export_env(&self) {
        std::env::set_var("RH_GMP_POOL_MB", self.gmp_pool_mb.to_string());
        std::env::set_var("RH_GWNUM_THRESHOLD_BITS", self.gwnum_threshold_bits.to_string());
    }
    pub fn backend(&self) -> crate::ffi::prp::Backend {
        match self.prp_backend.as_str() {
            "gmp" => crate::ffi::prp::Backend::Gmp,
            "gwnum" => crate::ffi::prp::Backend::Gwnum,
            _ => crate::ffi::prp::Backend::Auto,
        }
    }
    /// Предел CPU-сита с поправкой на диапазон задачи.
    ///
    /// Сито окупается тем сильнее, чем дороже PRP-тест, который оно экономит,
    /// а стоимость самого сита от k не зависит вовсе. На коротком прогоне
    /// (k < 5000, числа в единицы килобит) построение сита до 2^27 занимает
    /// 2.25 с — больше, чем весь остальной поиск. Поэтому предел
    /// масштабируется по k_max и лишь затем ограничивается конфигом.
    ///
    /// Коэффициент 2000 подобран так, чтобы на k_max ≈ 60000 получалось
    /// ~2^27 (измеренный оптимум), а на k_max = 5000 — примерно 2^23.
    pub fn sieve_limit(&self) -> u64 {
        let scaled = self.k_max.saturating_mul(2000).max(1 << 16);
        scaled.min(self.bitsieve_q_limit)
    }

    /// Нижняя граница trial factoring: ниже уже отработало малое сито.
    pub fn q_min(&self) -> u128 {
        (self.tf_q_min as u128).max(self.sieve_limit() as u128)
    }
    pub fn q_hard_max(&self) -> u128 { 1u128 << self.tf_q_hard_max_bits }
    pub fn bits_of(&self, k: u64) -> u64 {
        ((k - 1) as f64 * (self.base as f64).log2()) as u64 + 1
    }
}