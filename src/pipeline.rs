//! Четырёхступенчатый конвейер:
//!
//!   [k-gen] → [CPU small sieve, rayon] → [GPU trial factoring] → [P-1] → [PRP]
//!
//! Ключевые решения:
//!
//! * Каждая стадия отделена ОГРАНИЧЕННЫМ каналом: это даёт естественный
//!   backpressure, память не растёт, а медленная стадия просто притормаживает
//!   предыдущую.
//! * GPU крутится в собственном потоке на устройство (контекст CUDA привязан
//!   к потоку), PRP — в rayon-пуле с thread-local GMP-аренами.
//! * Показатели идут на GPU БАТЧАМИ: один launch обсчитывает `tf_k_batch`
//!   значений k, иначе при больших k диапазон m слишком мал, чтобы загрузить
//!   устройство.
//! * КАЖДЫЙ найденный на GPU делитель перепроверяется на CPU через GMP.
//!   Ложное срабатывание — сигнал о баге в ядре (или о разогнанной карте),
//!   и мы обязаны его увидеть, а не молча выбросить кандидата.

use crate::config::Config;
use crate::ffi::gpu::{GpuContext, Width};
use crate::ffi::pm1::{self, Pm1Params};
use crate::ffi::prp;
use crate::sieve::{PrimeIter, SmallSieve};
use crate::tuner::Tuner;
use crate::worklog::WorkLog;
use anyhow::Result;
use crossbeam_channel as chan;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum Outcome {
    SmallFactor { k: u64, q: u64 },
    GpuFactor { k: u64, q: u128 },
    Pm1Factor { k: u64, q: String },
    Composite { k: u64, bits: u64, secs: f64, backend: u32 },
    Prp { k: u64, bits: u64, secs: f64, backend: u32 },
    /// Показатель не удалось посчитать ни одним бэкендом. Пишется в журнал
    /// явно: молча пропавший кандидат неотличим от непроверенного, а такая
    /// запись видна и человеку, и `--verify`.
    Failed { k: u64, bits: u64, reason: String },
}

#[derive(Default)]
pub struct Stats {
    pub generated: AtomicU64,
    pub small_sieved: AtomicU64,
    pub gpu_sieved: AtomicU64,
    pub pm1_sieved: AtomicU64,
    pub prp_tested: AtomicU64,
    pub prp_found: AtomicU64,
    pub false_positives: AtomicU64,
    pub double_checked: AtomicU64,
    pub double_check_mismatches: AtomicU64,
}

pub fn run(cfg: Arc<Config>, log: Arc<WorkLog>) -> Result<Vec<u64>> {
    let stats = Arc::new(Stats::default());
    let tuner = Arc::new(Tuner::new(cfg.q_min(), cfg.q_hard_max()));
    let t0 = Instant::now();

    // ── Стадия 0: генерация k и отсев малыми делителями ───────────────
    let sieve_limit = cfg.sieve_limit();
    log::info!("Строим малое сито до q ≤ {sieve_limit} ...");
    let small = SmallSieve::build(cfg.base, sieve_limit, cfg.k_min, cfg.k_max);
    log::info!(
        "Малое сито готово за {:.2}s: отсеяно {} показателей (q ≤ {})",
        t0.elapsed().as_secs_f64(),
        small.eliminated(),
        small.q_limit()
    );

    // Глубина очереди задана в ШТУКАХ КАНДИДАТОВ, а на GPU идут батчи,
    // поэтому пересчитываем, иначе в канале осело бы queue_depth*tf_k_batch чисел.
    let gpu_queue = (cfg.queue_depth / cfg.tf_k_batch).max(2);
    let (tx_gpu, rx_gpu) = chan::bounded::<Vec<u64>>(gpu_queue);
    let (tx_prp, rx_prp) = chan::bounded::<u64>(cfg.queue_depth.max(2));
    let (tx_out, rx_out) = chan::unbounded::<Outcome>();

    // ── Стадия 1: k-generator + small sieve ───────────────────────────
    let gen_handle = {
        let cfg = cfg.clone();
        let stats = stats.clone();
        let log_ref = log.clone();
        let tx_gpu = tx_gpu.clone();
        let tx_out = tx_out.clone();
        std::thread::Builder::new()
            .name("k-gen+sieve".into())
            .spawn(move || {
                let ks: Vec<u64> = PrimeIter::new(cfg.k_min, cfg.k_max)
                    .filter(|&k| !log_ref.is_done(k))
                    .collect();
                stats.generated.store(ks.len() as u64, Ordering::Relaxed);
                log::info!("К проверке: {} простых показателей", ks.len());

                // Чистый CPU-фильтр, идеально масштабируется по ядрам.
                let survivors: Vec<u64> = ks
                    .par_iter()
                    .filter_map(|&k| match small.find_factor(k) {
                        Some(q) => {
                            stats.small_sieved.fetch_add(1, Ordering::Relaxed);
                            let _ = tx_out.send(Outcome::SmallFactor { k, q });
                            None
                        }
                        None => Some(k),
                    })
                    .collect();

                // Батчи одинакового порядка величины k: у соседних простых
                // почти совпадают границы по m, значит один launch не тратит
                // время на «пустые» участки диапазона.
                for chunk in survivors.chunks(cfg.tf_k_batch) {
                    if tx_gpu.send(chunk.to_vec()).is_err() {
                        break;
                    }
                }
                drop(tx_gpu);
                drop(tx_out);
            })?
    };
    drop(tx_gpu);

    // ── Стадия 2: GPU trial factoring ─────────────────────────────────
    let devices = select_devices(&cfg);
    let mut gpu_handles = Vec::new();

    if devices.is_empty() {
        if cfg.tf_enabled {
            log::warn!("GPU не найден — стадия TF пропускается, все кандидаты идут в PRP.");
        }
        let rx = rx_gpu.clone();
        let tx = tx_prp.clone();
        gpu_handles.push(std::thread::Builder::new()
            .name("tf-bypass".into())
            .spawn(move || {
                for batch in rx {
                    for k in batch {
                        if tx.send(k).is_err() {
                            return;
                        }
                    }
                }
            })?);
    } else {
        for dev in devices {
            let rx = rx_gpu.clone();
            let tx = tx_prp.clone();
            let tx_out = tx_out.clone();
            let cfg = cfg.clone();
            let stats = stats.clone();
            let tuner = tuner.clone();
            gpu_handles.push(
                std::thread::Builder::new()
                    .name(format!("gpu-tf-{dev}"))
                    .spawn(move || {
                        gpu_worker(dev, cfg, stats, tuner, rx, tx, tx_out);
                    })?,
            );
        }
    }
    drop(rx_gpu);
    drop(tx_prp);

    // ── Стадия 3: P-1 + PRP ───────────────────────────────────────────
    //
    // ЗДЕСЬ НАМЕРЕННО НЕ rayon. `par_bridge` тянул бы задачи из канала прямо
    // изнутри рабочих потоков пула: поток, ждущий следующий элемент, блокируется,
    // удерживая внутренний мьютекс итератора. Стоит всем потокам пула застрять
    // на пустом канале — и параллельный фильтр стадии 1 (тот же глобальный пул)
    // не получит ни одного потока, а значит кандидаты не появятся никогда.
    // Выделенные потоки-потребители на MPMC-канале от этого свободны.
    let n_prp = if cfg.threads == 0 { num_cpus::get() } else { cfg.threads };
    let mut prp_handles = Vec::with_capacity(n_prp);
    for i in 0..n_prp {
        let cfg = cfg.clone();
        let stats = stats.clone();
        let tuner = tuner.clone();
        let tx_out = tx_out.clone();
        let rx = rx_prp.clone();
        prp_handles.push(
            std::thread::Builder::new()
                .name(format!("prp-{i}"))
                .spawn(move || {
                    for k in rx {
                        prp_stage(&cfg, &stats, &tuner, &tx_out, k);
                    }
                })?,
        );
    }
    drop(rx_prp);
    drop(tx_out);

    // ── Периодический сброс журнала на диск ───────────────────────────
    // Записи о простых делителях буферизуются (их тысячи в секунду), поэтому
    // без чекпоинтов падение процесса стоило бы часов повторной работы.
    // PRP-находки флашатся отдельно и немедленно — см. WorkLog::record.
    let stop_flusher = Arc::new(AtomicBool::new(false));
    let flusher = {
        let log = log.clone();
        let stop = stop_flusher.clone();
        let period = Duration::from_secs(cfg.checkpoint_secs.max(1));
        std::thread::Builder::new()
            .name("worklog-flush".into())
            .spawn(move || {
                let mut last = Instant::now();
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(200));
                    if last.elapsed() >= period {
                        if let Err(e) = log.flush() {
                            log::warn!("сброс журнала не удался: {e}");
                        }
                        last = Instant::now();
                    }
                }
            })?
    };

    // ── Сбор результатов ──────────────────────────────────────────────
    let mut found = Vec::new();
    for outcome in rx_out {
        match &outcome {
            Outcome::Prp { k, bits, secs, .. } => {
                log::info!("*** PRP: R_{k}({}) — {bits} бит ({secs:.1}s) ***", cfg.base);
                log::info!("    {}", decimal_preview(cfg.base, *k));
                found.push(*k);
            }
            Outcome::Composite { k, bits, secs, .. } => {
                log::debug!("составное k={k} ({bits} бит, {secs:.2}s)")
            }
            Outcome::SmallFactor { k, q } => log::trace!("k={k}: малый делитель {q}"),
            Outcome::GpuFactor { k, q } => log::debug!("k={k}: делитель с GPU {q}"),
            Outcome::Pm1Factor { k, q } => log::info!("k={k}: делитель P-1 {q}"),
            Outcome::Failed { k, bits, reason } =>
                log::error!("k={k} ({bits} бит) НЕ ПОСЧИТАН: {reason}"),
        }
        log.record(&outcome)?;
    }

    let _ = gen_handle.join();
    for h in gpu_handles {
        let _ = h.join();
    }
    for h in prp_handles {
        let _ = h.join();
    }
    stop_flusher.store(true, Ordering::Relaxed);
    let _ = flusher.join();

    log::info!(
        "Готово за {:.1}s | сгенерировано={} малое_сито={} gpu={} pm1={} prp={} найдено={}",
        t0.elapsed().as_secs_f64(),
        stats.generated.load(Ordering::Relaxed),
        stats.small_sieved.load(Ordering::Relaxed),
        stats.gpu_sieved.load(Ordering::Relaxed),
        stats.pm1_sieved.load(Ordering::Relaxed),
        stats.prp_tested.load(Ordering::Relaxed),
        stats.prp_found.load(Ordering::Relaxed),
    );
    log::info!("{}", tuner.report());
    {
        // Пул считается per-thread; здесь видно значение потока-сборщика,
        // но и его достаточно, чтобы заметить упирание в потолок gmp_pool_mb.
        let (hi, cap) = prp::PrpArena::pool_usage();
        if cap > 0 {
            log::info!(
                "Пул GMP: пик {:.1} МиБ из {:.1} МиБ{}",
                hi as f64 / (1 << 20) as f64,
                cap as f64 / (1 << 20) as f64,
                if hi * 10 > cap * 9 { " — стоит поднять gmp_pool_mb" } else { "" }
            );
        }
    }

    let dc = stats.double_checked.load(Ordering::Relaxed);
    if dc > 0 {
        let mm = stats.double_check_mismatches.load(Ordering::Relaxed);
        log::info!(
            "Double-check: пересчитано {dc}, расхождений {mm}{}",
            if mm > 0 { " — СМОТРИТЕ ОШИБКИ ВЫШЕ" } else { "" }
        );
    }

    let fp = stats.false_positives.load(Ordering::Relaxed);
    if fp > 0 {
        log::error!(
            "GPU выдал {fp} непроверившихся делителей — результаты TF ненадёжны, \
             проверьте ядро и стабильность устройства."
        );
    }

    found.sort_unstable();
    Ok(found)
}

/// Какие устройства использовать: явный список из конфига или все найденные.
fn select_devices(cfg: &Config) -> Vec<i32> {
    if !cfg.tf_enabled {
        return Vec::new();
    }
    let n = GpuContext::device_count();
    if n == 0 {
        return Vec::new();
    }
    if cfg.gpu_devices.is_empty() {
        (0..n as i32).collect()
    } else {
        cfg.gpu_devices
            .iter()
            .copied()
            .filter(|&d| d >= 0 && (d as usize) < n)
            .collect()
    }
}

/// Поток одного GPU: батч k → диапазоны m → проверка попаданий → PRP.
fn gpu_worker(
    dev: i32,
    cfg: Arc<Config>,
    stats: Arc<Stats>,
    tuner: Arc<Tuner>,
    rx: chan::Receiver<Vec<u64>>,
    tx: chan::Sender<u64>,
    tx_out: chan::Sender<Outcome>,
) {
    let mut ctx = match GpuContext::new(dev) {
        Ok(c) => c,
        Err(e) => {
            log::error!("GPU {dev}: инициализация не удалась: {e} — батчи пойдут в PRP напрямую");
            for batch in rx {
                for k in batch {
                    if tx.send(k).is_err() {
                        return;
                    }
                }
            }
            return;
        }
    };
    if let Ok(info) = GpuContext::query(dev) {
        log::info!(
            "GPU {dev}: {} | SM={} | CC {}.{} | {} МиБ",
            info.name_str(),
            info.sm_count,
            info.cc_major,
            info.cc_minor,
            info.global_mem_bytes >> 20
        );
    }

    calibrate_gpu(&mut ctx, &cfg, &tuner, dev);

    for batch in rx {
        if batch.is_empty() {
            continue;
        }
        // Границы по m считаем по САМОМУ БОЛЬШОМУ k батча: для меньших k
        // фактическая глубина по q окажется чуть ниже — это консервативно
        // и не даёт выйти за пределы выбранной ширины арифметики.
        let k_hi = *batch.last().unwrap();
        let bits = cfg.bits_of(k_hi);

        // Потолок 128-битной арифметики ядра: q обязано быть < 2^127,
        // иначе кандидаты у верхней границы молча отбрасывались бы в q_fits.
        let q_cap = (1u128 << 127) - 1;
        let q_max = if cfg.tf_adaptive {
            tuner.optimal_q_max(k_hi, bits).min(cfg.q_hard_max())
        } else {
            cfg.q_hard_max()
        }
        .min(q_cap);
        let q_start = cfg.q_min();

        // Считаем в u128 и насыщаем: при малом k и большом q_max частное
        // не влезает в u64, а тихое усечение дало бы неверный диапазон.
        let two_k = 2u128 * k_hi as u128;
        let clamp = |x: u128| x.min(u64::MAX as u128) as u64;
        let m_start = clamp(q_start / two_k).max(1);
        let m_end = clamp(q_max / two_k);

        if m_end <= m_start {
            for &k in &batch {
                if tx.send(k).is_err() {
                    return;
                }
            }
            continue;
        }

        // Ширина по фактически достижимому q = 2*m_end*k_hi + 1.
        let q_top = two_k * m_end as u128 + 1;
        let width = Width::for_qmax(q_top);
        // Кандидаты, не влезающие в ширину, ядро молча пропускает,
        // поэтому убеждаемся, что верхняя граница действительно покрыта.
        debug_assert!(q_top < width.q_limit(), "ширина {width:?} не покрывает q={q_top}");
        if let Err(e) = ctx.upload_ks(&batch) {
            log::error!("GPU {dev}: upload_ks: {e}");
            for &k in &batch {
                if tx.send(k).is_err() {
                    return;
                }
            }
            continue;
        }

        let span = ctx.suggest_span(batch.len(), width).max(1024);
        let mut alive = vec![true; batch.len()];
        let mut m = m_start;

        while m < m_end {
            let this_span = span.min(m_end - m);
            let t_launch = Instant::now();
            let res = match ctx.tf_batch(cfg.base, m, this_span, width) {
                Ok(r) => r,
                Err(e) => {
                    log::error!("GPU {dev}: tf_batch (m={m}): {e}");
                    break;
                }
            };
            tuner.observe_gpu(width, res.candidates_tested, t_launch.elapsed().as_secs_f64());

            for hit in res.hits() {
                let idx = hit.k_index as usize;
                if idx >= batch.len() || !alive[idx] {
                    continue;
                }
                let k = batch[idx];
                let q = hit.q();

                // Независимая проверка на CPU: делит ли q число R_k(b)?
                let verified = prp::with_arena(|a| a.verify_factor(cfg.base, k, q))
                    .unwrap_or(Ok(false))
                    .unwrap_or(false);

                if verified {
                    alive[idx] = false;
                    stats.gpu_sieved.fetch_add(1, Ordering::Relaxed);
                    let _ = tx_out.send(Outcome::GpuFactor { k, q });
                } else {
                    stats.false_positives.fetch_add(1, Ordering::Relaxed);
                    log::error!(
                        "ЛОЖНОЕ СРАБАТЫВАНИЕ GPU: b={} k={k} q={q} не делит R_k — \
                         проверьте краевые случаи ядра (q|b, q|b-1) и стабильность карты",
                        cfg.base
                    );
                }
            }

            if alive.iter().all(|&a| !a) {
                break; // весь батч отсеян
            }
            m += this_span;
        }

        for (i, &k) in batch.iter().enumerate() {
            if alive[i] && tx.send(k).is_err() {
                return;
            }
        }
    }
}

/// Пробный запуск, чтобы тюнер знал реальную скорость ЭТОЙ карты.
///
/// Без него первые батчи планируются по вшитым константам, и глубина trial
/// factoring выбирается наугад: на GTX 1650 реальная скорость оказалась в 27
/// раз ниже прежней стартовой оценки. Стоит один запуск (~10 мс).
fn calibrate_gpu(ctx: &mut GpuContext, cfg: &Config, tuner: &Tuner, dev: i32) {
    let k = 1_000_003u64; // типичный по величине показатель
    if ctx.upload_ks(&[k]).is_err() {
        return;
    }
    let width = Width::W64;
    let span = ctx.suggest_span(1, width);
    let m0 = (cfg.q_min() / (2 * k as u128)).max(1) as u64;

    // Первый запуск прогревает (сборка графа, выход карты на частоты),
    // засчитываем только второй.
    let _ = ctx.tf_batch(cfg.base, m0, span, width);
    let t = Instant::now();
    match ctx.tf_batch(cfg.base, m0 + span, span, width) {
        Ok(res) => {
            let secs = t.elapsed().as_secs_f64();
            tuner.set_gpu_rate(width, res.candidates_tested, secs);
            log::info!(
                "GPU {dev}: калибровка — {:.2} млрд кандидатов/с",
                tuner.gpu_rate(width) / 1e9
            );
        }
        Err(e) => log::warn!("GPU {dev}: калибровка не удалась: {e}"),
    }
}

/// P-1 (по решению тюнера) и затем PRP.
fn prp_stage(
    cfg: &Config,
    stats: &Stats,
    tuner: &Tuner,
    tx_out: &chan::Sender<Outcome>,
    k: u64,
) {
    let bits = cfg.bits_of(k);

    // ── P-1: дешёвая попытка снять кандидата до дорогого PRP ──
    // q_typical — граница, до которой TF уже дошёл: P-1 имеет смысл только
    // для делителей ЗА ней, иначе их и так нашёл бы перебор.
    let q_typical = tuner.optimal_q_max(k, bits).min(cfg.q_hard_max());
    if cfg.pm1_enabled
        && pm1::available()
        && (!cfg.pm1_adaptive
            || tuner.pm1_worth_it(bits, cfg.pm1_b1, cfg.pm1_b2, q_typical, k))
    {
        let params = Pm1Params { b1: cfg.pm1_b1, b2: cfg.pm1_b2, k_known: k };
        let got = prp::with_arena(|a| pm1::factor(a, cfg.base, k, &params));
        match got {
            Ok(Ok(Some(q))) => {
                stats.pm1_sieved.fetch_add(1, Ordering::Relaxed);
                let _ = tx_out.send(Outcome::Pm1Factor { k, q });
                return;
            }
            Ok(Ok(None)) => {}
            Ok(Err(e)) | Err(e) => log::debug!("P-1 недоступен/ошибка (k={k}): {e}"),
        }
    }

    // ── PRP ──
    let r = prp::with_arena(|a| {
        a.reserve_bits(bits + 128);
        a.is_prp(cfg.base, k, cfg.mr_rounds, cfg.backend(), cfg.gerbicz_l)
    });
    stats.prp_tested.fetch_add(1, Ordering::Relaxed);

    match r {
        Ok(Ok((is_prp, st))) => {
            tuner.observe_prp(st.bits.max(1), st.elapsed_sec);
            let out = if is_prp {
                stats.prp_found.fetch_add(1, Ordering::Relaxed);
                Outcome::Prp { k, bits: st.bits, secs: st.elapsed_sec, backend: st.backend_used }
            } else {
                // Выборочный double-check: ложное «PRP» уже исключено
                // верификацией в диспетчере, а вот ложное «составное» —
                // это ПОТЕРЯННАЯ находка, и заметить её можно только
                // повторным счётом. Пересчитываем долю кандидатов вторым,
                // независимым бэкендом.
                if should_double_check(cfg, k, st.backend_used) {
                    double_check(cfg, stats, tx_out, k, st.backend_used);
                }
                Outcome::Composite { k, bits: st.bits, secs: st.elapsed_sec, backend: st.backend_used }
            };
            let _ = tx_out.send(out);
        }
        // ОШИБКА ОСНОВНОГО ПУТИ — НЕ ПОВОД ТЕРЯТЬ КАНДИДАТА.
        //
        // Прежде здесь стояла только запись в лог. Кандидат при этом не попадал
        // в журнал вовсе: он не «составной» и не «PRP», он просто исчезал, и
        // обнаружить пропажу можно было лишь сверкой с внешним списком.
        // Реальный случай: b=2, k=1279 (M_1279 — известное простое Мерсенна) в
        // многопоточном прогоне дал «internal error» при настройке GWNUM и
        // пропал из результатов; тот же k, посчитанный отдельно, проходит без
        // ошибок всеми бэкендами, то есть сбой непостоянный.
        //
        // Теперь при любой ошибке кандидат досчитывается точной арифметикой
        // GMP: она не зависит ни от FFT, ни от состояния GWNUM. Если и это не
        // удалось — пишем в журнал явную запись об ошибке, чтобы показатель
        // остался виден и `--verify` мог к нему вернуться.
        Ok(Err(e)) | Err(e) => {
            log::warn!("PRP k={k}: {e} — пересчитываю на GMP");
            let again = prp::with_arena(|a| {
                a.reserve_bits(bits + 128);
                a.is_prp(cfg.base, k, cfg.mr_rounds, prp::Backend::Gmp, cfg.gerbicz_l)
            });
            match again {
                Ok(Ok((is_prp, st))) => {
                    tuner.observe_prp(st.bits.max(1), st.elapsed_sec);
                    let out = if is_prp {
                        stats.prp_found.fetch_add(1, Ordering::Relaxed);
                        log::warn!("PRP k={k}: GMP подтвердил находку, потерянную основным путём");
                        Outcome::Prp { k, bits: st.bits, secs: st.elapsed_sec,
                                       backend: st.backend_used }
                    } else {
                        Outcome::Composite { k, bits: st.bits, secs: st.elapsed_sec,
                                             backend: st.backend_used }
                    };
                    let _ = tx_out.send(out);
                }
                Ok(Err(e2)) | Err(e2) => {
                    log::error!("PRP k={k}: не удалось посчитать ни одним бэкендом \
                                 ({e} / {e2}) — показатель помечен как нерешённый");
                    let _ = tx_out.send(Outcome::Failed { k, bits, reason: e2.to_string() });
                }
            }
        }
    }
}

/// Попадает ли кандидат в выборку перепроверки?
///
/// Решение ДЕТЕРМИНИРОВАНО (хеш от base и k), а не случайно: один и тот же
/// прогон перепроверяет одни и те же числа, и результат воспроизводим.
fn should_double_check(cfg: &Config, k: u64, backend_used: u32) -> bool {
    if cfg.double_check_ratio <= 0.0 {
        return false;
    }
    // Пересчитывать GMP через GMP бессмысленно — это тот же самый код.
    if backend_used != prp::Backend::Gwnum as u32 {
        return false;
    }
    // FNV-1a от (base, k)
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in cfg.base.to_le_bytes().iter().chain(k.to_le_bytes().iter()) {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    ((h % 10_000) as f64) < cfg.double_check_ratio * 10_000.0
}

/// Пересчёт вердикта «составное» независимым бэкендом.
fn double_check(
    cfg: &Config,
    stats: &Stats,
    tx_out: &chan::Sender<Outcome>,
    k: u64,
    backend_used: u32,
) {
    // GWNUM считал на IBDWT — перепроверяем точной целочисленной арифметикой.
    let second = prp::Backend::Gmp;
    let r = prp::with_arena(|a| a.is_prp(cfg.base, k, cfg.mr_rounds, second, 0));
    stats.double_checked.fetch_add(1, Ordering::Relaxed);

    match r {
        Ok(Ok((true, st))) => {
            // Первый бэкенд потерял находку — это серьёзно.
            stats.double_check_mismatches.fetch_add(1, Ordering::Relaxed);
            log::error!(
                "DOUBLE-CHECK: k={k} был объявлен составным (бэкенд {backend_used}),                  но независимый пересчёт даёт PRP! Доверяем пересчёту.",
            );
            stats.prp_found.fetch_add(1, Ordering::Relaxed);
            let _ = tx_out.send(Outcome::Prp {
                k,
                bits: st.bits,
                secs: st.elapsed_sec,
                backend: st.backend_used,
            });
        }
        Ok(Ok((false, _))) => log::debug!("double-check k={k}: подтверждено составное"),
        Ok(Err(e)) | Err(e) => log::error!("double-check k={k}: {e}"),
    }
}

/// Короткое представление найденного числа: первые и последние цифры.
/// Печатать целиком бессмысленно — у R_49081(10) это 49 081 знак.
fn decimal_preview(base: u64, k: u64) -> String {
    match prp::with_arena(|a| a.decimal(base, k)) {
        Ok(Ok(s)) => {
            let n = s.len();
            if n <= 60 {
                format!("N = {s} ({n} знаков)")
            } else {
                format!("N = {}...{} ({n} знаков)", &s[..25], &s[n - 25..])
            }
        }
        _ => String::from("N = <не удалось построить десятичную запись>"),
    }
}
