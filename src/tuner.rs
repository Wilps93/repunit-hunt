//! Адаптивный подбор глубины trial factoring.
//!
//! ТЕОРИЯ.
//! Стоимость расширения TF на очередную «декаду» [Q, e·Q]:
//!     C(Q) = (e-1)·Q / (2k) / rate_gpu    [секунд]
//! где делитель 2k учитывает, что проверяем только q ≡ 1 (mod 2k).
//!
//! Выгода: вероятность найти делитель в этой декаде ≈ 1/ln Q
//! (эвристика Mertens с поправкой на форму q=2mk+1: делители
//! распределены как обычные простые в арифметической прогрессии).
//!     G(Q) = T_prp / ln Q
//!
//! Оптимум: расширяем, пока C(Q) < G(Q).
//!
//! Стоимость P-1 оценивается отдельно (см. `pm1_worth_it`): раньше здесь
//! утверждалось, что этап «покрывает диапазон, эквивалентный TF до 2^75»,
//! но подсчёт в модульных умножениях этого не подтверждает — на числах до
//! нескольких миллионов бит P-1 дороже PRP-теста, который он экономит.

use crate::ffi::gpu::Width;

/// Показатель степени в модели стоимости PRP: t ≈ coef · bits^PRP_EXP.
///
/// Здесь стояло 1.15 — как если бы тест стоил примерно линейно от размера.
/// Подгонка по 2756 реальным замерам (журнал прогона k = 5000…60000,
/// 20 000…199 310 бит) даёт показатель 2.24, и это не случайность:
/// R_k(b) не имеет специальной формы, поэтому GWNUM работает через
/// gwsetup_general_mod — IBDWT-умножение плюс редукция Барретта. Одна
/// итерация стоит O(n log n), а итераций n, откуда ~n²·log n.
///
/// Цена ошибки была вполне ощутимой: на 163 041 бит старая модель обещала
/// 9.7 с против фактических 16.6, то есть тюнер вдвое занижал выгоду от
/// trial factoring и выбирал слишком мелкую глубину. Средняя ошибка модели
/// упала с 6.11 с до 1.03 с.
const PRP_EXP: f64 = 2.239;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Tuner {
    /// Измеренная скорость GPU: кандидатов q в секунду (по ширинам).
    rate: [RwLock<f64>; 3],
    /// Измеренное время одного PRP-теста: t = prp_coef * bits^PRP_EXP.
    prp_coef: RwLock<f64>,
    samples_gpu: [AtomicU64; 3],
    samples_prp: AtomicU64,
    /// Жёсткие границы из конфига
    q_min: u128,
    q_hard_max: u128,
}

impl Tuner {
    pub fn new(q_min: u128, q_hard_max: u128) -> Self {
        Self {
            // Стартовые оценки НАМЕРЕННО занижены. Раньше здесь стояло 8e9
            // q/s — «как на топовой карте», и до накопления статистики тюнер
            // планировал глубину TF в десятки раз оптимистичнее реальности
            // (замер на GTX 1650 дал 0.30e9). Заниженный старт безопаснее:
            // он ведёт к более мелкому TF, а не к бесполезно глубокому.
            // Настоящее значение приходит из calibrate() при старте GPU-потока.
            rate: [RwLock::new(0.3e9), RwLock::new(0.15e9), RwLock::new(0.1e9)],
            // Стартовое значение из подгонки по реальным замерам (см. PRP_EXP);
            // дальше уточняется на лету по фактическим временам.
            prp_coef: RwLock::new(3.79e-11),
            samples_gpu: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            samples_prp: AtomicU64::new(0),
            q_min, q_hard_max,
        }
    }

    /// Обновить измеренную скорость GPU (экспоненциальное сглаживание).
    pub fn observe_gpu(&self, width: Width, candidates: u64, secs: f64) {
        if secs <= 0.0 || candidates == 0 { return; }
        let r = candidates as f64 / secs;
        let idx = width as usize;
        let mut w = self.rate[idx].write();
        let n = self.samples_gpu[idx].fetch_add(1, Ordering::Relaxed);
        let alpha = if n < 8 { 0.5 } else { 0.05 };
        *w = (1.0 - alpha) * *w + alpha * r;
    }

    /// Задать скорость GPU напрямую (результат калибровочного запуска).
    /// В отличие от `observe_gpu`, не смешивает со стартовой константой.
    pub fn set_gpu_rate(&self, width: Width, candidates: u64, secs: f64) {
        if secs <= 0.0 || candidates == 0 { return; }
        *self.rate[width as usize].write() = candidates as f64 / secs;
        self.samples_gpu[width as usize].store(1, Ordering::Relaxed);
    }

    /// Обновить модель стоимости PRP.
    pub fn observe_prp(&self, bits: u64, secs: f64) {
        if secs <= 0.0 || bits == 0 { return; }
        let coef = secs / (bits as f64).powf(PRP_EXP);
        let mut w = self.prp_coef.write();
        let n = self.samples_prp.fetch_add(1, Ordering::Relaxed);
        let alpha = if n < 8 { 0.5 } else { 0.05 };
        *w = (1.0 - alpha) * *w + alpha * coef;
    }

    pub fn prp_seconds(&self, bits: u64) -> f64 {
        *self.prp_coef.read() * (bits as f64).powf(PRP_EXP)
    }

    pub fn gpu_rate(&self, width: Width) -> f64 { *self.rate[width as usize].read() }

    /// Оптимальная верхняя граница q для данного k.
    pub fn optimal_q_max(&self, k: u64, bits: u64) -> u128 {
        let t_prp = self.prp_seconds(bits);
        let mut q: u128 = self.q_min.max(1 << 20);

        loop {
            // saturating: q может подойти вплотную к 2^127 — переполнение здесь
            // означало бы выбор заниженной ширины арифметики.
            let width = Width::for_qmax(q.saturating_mul(3));
            let rate = self.gpu_rate(width);
            // Кандидатов в декаде [q, e*q]: (e-1)*q / (2k)
            let cands = (std::f64::consts::E - 1.0) * (q as f64) / (2.0 * k as f64);
            let cost = cands / rate;
            let gain = t_prp / (q as f64).ln();

            if cost > gain { return q; }
            let next_f = q as f64 * std::f64::consts::E;
            if !next_f.is_finite() || next_f >= self.q_hard_max as f64 {
                return self.q_hard_max;
            }
            let next = next_f as u128;
            if next <= q { return self.q_hard_max; }   // защита от залипания
            q = next;
        }
    }

    /// Значения функции Дикмана ρ(u) — доля чисел величины x, все простые
    /// делители которых меньше x^(1/u). Таблица, а не асимптотика: последняя
    /// врёт на порядки при u < 5, а нас интересует именно этот участок.
    fn dickman_rho(u: f64) -> f64 {
        const TAB: [(f64, f64); 9] = [
            (1.0, 1.0),      (1.5, 5.9e-1), (2.0, 3.07e-1), (2.5, 1.30e-1),
            (3.0, 4.86e-2),  (4.0, 4.91e-3), (5.0, 3.5e-4),  (6.0, 1.96e-5),
            (8.0, 3.2e-8),
        ];
        if u <= 1.0 { return 1.0; }
        if u >= 8.0 { return 3.2e-8; }
        for w in TAB.windows(2) {
            let ((u0, r0), (u1, r1)) = (w[0], w[1]);
            if u <= u1 {
                // интерполяция по логарифму: ρ падает экспоненциально
                let t = (u - u0) / (u1 - u0);
                return (r0.ln() + t * (r1.ln() - r0.ln())).exp();
            }
        }
        3.2e-8
    }

    /// Стоит ли запускать P-1 перед PRP?
    ///
    /// ЧЕСТНАЯ ЭКОНОМИКА (прежняя версия брала вероятность успеха «≈4%» с
    /// потолка, из-за чего решение не имело отношения к реальности).
    ///
    /// * Стоимость. Stage 1 требует ≈ 2.9·B1 модульных умножений, stage 2 —
    ///   ещё примерно B2/50. Весь PRP-тест стоит ≈ `bits` умножений. То есть
    ///   P-1 окупается, только если B1 много меньше размера числа.
    /// * Вероятность. P-1 найдёт делитель q, если m = (q−1)/(2k) окажется
    ///   B1-гладким. Для интересных нам q (тех, что лежат глубже пройденного TF)
    ///   m ≈ q/(2k), и вероятность — это ρ(ln m / ln B1).
    ///
    /// Численно для нашего диапазона: bits = 163 041, B1 = 10⁵, q ≈ 2⁶⁰,
    /// k ≈ 5·10⁴ дают стоимость 2.9·10⁵ умножений против 1.6·10⁵ у всего
    /// PRP-теста — то есть этап дороже того, что экономит. Поэтому на сотнях
    /// тысяч бит P-1 отключается, а смысл появляется на миллионах.
    pub fn pm1_worth_it(&self, bits: u64, b1: u64, b2: u64, q_typical: u128, k: u64) -> bool {
        if b1 < 100 || k == 0 {
            return false;
        }
        // Всё считаем в модульных умножениях — единица, общая для обеих частей.
        let prp_mulmods = bits as f64;
        let cost_mulmods = 2.9 * b1 as f64 + b2 as f64 / 50.0;

        // Типичный множитель m у делителя, до которого TF не достаёт.
        let m = (q_typical as f64 / (2.0 * k as f64)).max(2.0);
        let u = m.ln() / (b1 as f64).ln();
        let p = Self::dickman_rho(u);

        cost_mulmods < p * prp_mulmods
    }

    pub fn report(&self) -> String {
        format!(
            "Tuner: GPU rate 64={:.2}G/s 96={:.2}G/s 128={:.2}G/s | PRP coef={:.3e} \
             (пример: 100k бит -> {:.1}s)",
            self.gpu_rate(Width::W64) / 1e9,
            self.gpu_rate(Width::W96) / 1e9,
            self.gpu_rate(Width::W128) / 1e9,
            *self.prp_coef.read(),
            self.prp_seconds(100_000),
        )
    }
}