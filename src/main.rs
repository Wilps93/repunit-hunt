//! repunit-hunt — гибридный CPU/GPU поиск простых обобщённых репьюнитов
//! R_k(b) = (b^k - 1)/(b - 1).
//!
//! Результат работы — PRP-кандидаты (сильный тест Ферма + раунды Miller-Rabin).
//! Строгое доказательство простоты выполняется отдельно (ECPP/Primo):
//! у R_k(b) нет удобной формы N±1 с известной факторизацией.

mod affinity;
mod config;
mod ffi;
mod pipeline;
mod report;
mod sieve;
mod tuner;
mod verify;
mod worklog;

use anyhow::Result;
use clap::Parser;
use config::Config;
use std::path::PathBuf;
use std::sync::Arc;

/// mimalloc: у GMP/rayon очень много мелких аллокаций из многих потоков,
/// системный glibc-malloc на 32+ потоках заметно упирается в блокировки.
/// (Крупные временные буферы самой GMP идут мимо — через пул в rh_alloc.c.)
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(
    name = "repunit-hunt",
    version,
    about = "GPU/CPU поиск простых обобщённых репьюнитов R_k(b) = (b^k-1)/(b-1)"
)]
struct Cli {
    /// Путь к TOML-конфигу (флаги ниже имеют приоритет)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Основание b
    #[arg(short, long)]
    base: Option<u64>,
    /// Нижняя граница показателя k (включительно)
    #[arg(long)]
    kmin: Option<u64>,
    /// Верхняя граница показателя k (исключительно)
    #[arg(long)]
    kmax: Option<u64>,
    /// Число рабочих потоков (0 = по числу ядер)
    #[arg(long)]
    threads: Option<usize>,
    /// Отключить стадию GPU trial factoring
    #[arg(long)]
    no_gpu: bool,
    /// Отключить стадию P-1
    #[arg(long)]
    no_pm1: bool,

    /// Показать информацию об устройствах и выйти
    #[arg(long)]
    devices: bool,

    /// Перепроверить журнал вторым бэкендом и выйти (аналог double-check GIMPS)
    #[arg(long)]
    verify: bool,

    /// Какую долю «составных» пересчитывать при --verify (1.0 = все)
    #[arg(long, default_value = "0.02")]
    verify_ratio: f64,

    /// Доля «составных», пересчитываемых прямо во время поиска
    #[arg(long)]
    double_check: Option<f64>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();

    if cli.devices {
        return show_devices();
    }

    // ── Конфигурация ──────────────────────────────────────────────────
    let mut cfg = match &cli.config {
        Some(p) => Config::load(p)?,
        None => Config::default(),
    };
    if let Some(b) = cli.base {
        cfg.base = b;
    }
    if let Some(k) = cli.kmin {
        cfg.k_min = k;
    }
    if let Some(k) = cli.kmax {
        cfg.k_max = k;
    }
    if let Some(t) = cli.threads {
        cfg.threads = t;
    }
    if cli.no_gpu {
        cfg.tf_enabled = false;
    }
    if cli.no_pm1 {
        cfg.pm1_enabled = false;
    }
    if let Some(r) = cli.double_check {
        cfg.double_check_ratio = r;
    }
    cfg.validate()?;
    cfg.export_env(); // до старта потоков: нативный слой читает окружение

    // ── Режим перепроверки журнала ────────────────────────────────────
    if cli.verify {
        rayon::ThreadPoolBuilder::new()
            .num_threads(if cfg.threads == 0 { num_cpus::get() } else { cfg.threads })
            .build_global()?;
        let bad = verify::run(&cfg, cli.verify_ratio)?;
        std::process::exit(if bad == 0 { 0 } else { 1 });
    }

    // ── Пул потоков ───────────────────────────────────────────────────
    let threads = if cfg.threads == 0 { num_cpus::get() } else { cfg.threads };
    let pin = cfg.pin_threads;
    let order = Arc::new(affinity::cpu_order());
    if pin {
        affinity::enable_thp();
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("rh-worker-{i}"))
        .start_handler(move |i| {
            // Прибиваем поток ДО первых аллокаций арены: first-touch положит
            // страницы на локальный NUMA-узел (иначе до 2x потери на remote access).
            if pin {
                let cpu = order[i % order.len()];
                if !affinity::pin_current_thread(cpu) {
                    log::debug!("pin потока {i} к CPU {cpu} не поддержан платформой");
                }
            }
        })
        .build_global()?;

    let topo = affinity::topology();
    log::info!(
        "Пул rayon: {threads} потоков | CPU={} NUMA-узлов={} | pin={}",
        topo.cpus,
        topo.nodes,
        pin
    );
    log::info!(
        "Ищем R_k({}) для простых k ∈ [{}, {}) | бэкенд PRP: {} | GWNUM: {}",
        cfg.base,
        cfg.k_min,
        cfg.k_max,
        cfg.prp_backend,
        if ffi::prp::gwnum_available() { "есть" } else { "нет" }
    );
    if cfg.pm1_enabled && !ffi::pm1::available() {
        log::warn!("P-1 включён в конфиге, но libecm не собран — стадия будет пропущена.");
    }

    // ── Запуск ────────────────────────────────────────────────────────
    let log = Arc::new(worklog::WorkLog::open(&cfg.worklog)?);
    let cfg = Arc::new(cfg);

    let found = pipeline::run(cfg.clone(), log.clone())?;
    log.flush()?;

    report::write_json(
        std::path::Path::new(&cfg.results),
        &report::Report {
            base: cfg.base,
            k_min: cfg.k_min,
            k_max: cfg.k_max,
            prp_exponents: &found,
            note: "PRP, не доказательство простоты — требуется ECPP/Primo",
        },
    )?;

    println!("\n=== РЕЗУЛЬТАТ ===");
    if found.is_empty() {
        println!("PRP-кандидатов в диапазоне не найдено.");
    } else {
        for k in &found {
            println!("R_{k}({}) — PRP", cfg.base);
        }
    }
    println!("Журнал: {} | отчёт: {}", cfg.worklog, cfg.results);
    Ok(())
}

fn show_devices() -> Result<()> {
    let n = ffi::gpu::GpuContext::device_count();
    println!("CUDA-устройств: {n}");
    for d in 0..n as i32 {
        match ffi::gpu::GpuContext::query(d) {
            Ok(i) => println!(
                "  [{}] {} | SM={} | CC {}.{} | {} МиБ | {} МГц",
                i.device_id,
                i.name_str(),
                i.sm_count,
                i.cc_major,
                i.cc_minor,
                i.global_mem_bytes >> 20,
                i.clock_khz / 1000
            ),
            Err(e) => println!("  [{d}] запрос не удался: {e}"),
        }
    }
    println!(
        "PRP-бэкенды: GMP=есть, GWNUM={} | P-1 (libecm)={}",
        if ffi::prp::gwnum_available() { "есть" } else { "нет" },
        if ffi::pm1::available() { "есть" } else { "нет" }
    );
    Ok(())
}
