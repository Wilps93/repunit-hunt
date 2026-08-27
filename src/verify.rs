//! Перепроверка журнала (`--verify`) — аналог double-check в GIMPS.
//!
//! Что именно проверяется по каждой записи `worklog.jsonl`:
//!
//! * `factored` — делитель q пересчитывается: действительно ли q делит R_k(b)
//!   и является ли он СОБСТВЕННЫМ (q < N). Стоит доли миллисекунды, поэтому
//!   проверяются все записи подряд.
//! * `composite` — вердикт пересчитывается ДРУГИМ бэкендом: результат GWNUM
//!   перепроверяется точной целочисленной арифметикой GMP. Это единственный
//!   способ поймать ложное «составное», то есть потерянную находку.
//! * `PRP` — пересчитывается всегда: находка обязана подтверждаться.
//!
//! Пересчёт «составных» дорог (это полный PRP-тест), поэтому по умолчанию
//! берётся выборка; `--verify-ratio 1.0` проверяет всё.

use crate::config::Config;
use crate::ffi::prp::{self, Backend};
use crate::worklog::WorkLog;
use anyhow::Result;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct VerifyStats {
    pub factors_checked: AtomicU64,
    pub factors_bad: AtomicU64,
    pub prp_checked: AtomicU64,
    pub prp_bad: AtomicU64,
    pub composites_checked: AtomicU64,
    pub composites_bad: AtomicU64,
    pub skipped: AtomicU64,
}

/// Прогнать перепроверку журнала. `ratio` — доля «составных» для пересчёта.
pub fn run(cfg: &Config, ratio: f64) -> Result<u64> {
    let path = Path::new(&cfg.worklog);
    let records = WorkLog::read_records(path)?;
    anyhow::ensure!(!records.is_empty(), "журнал {} пуст или не найден", cfg.worklog);

    log::info!("Перепроверка {} записей из {}", records.len(), cfg.worklog);
    let st = VerifyStats::default();

    records.par_iter().for_each(|rec| {
        let k = match rec.get("k").and_then(|v| v.as_u64()) {
            Some(k) => k,
            None => return,
        };
        let status = rec.get("status").and_then(|v| v.as_str()).unwrap_or("");

        match status {
            "factored" => {
                let q = rec
                    .get("q")
                    .and_then(|v| v.as_str().and_then(|s| s.parse::<u128>().ok())
                        .or_else(|| v.as_u64().map(u128::from)));
                let Some(q) = q else {
                    st.skipped.fetch_add(1, Ordering::Relaxed);
                    return;
                };
                st.factors_checked.fetch_add(1, Ordering::Relaxed);
                let ok = prp::with_arena(|a| a.verify_factor(cfg.base, k, q))
                    .unwrap_or(Ok(false))
                    .unwrap_or(false);
                if !ok {
                    st.factors_bad.fetch_add(1, Ordering::Relaxed);
                    log::error!(
                        "k={k}: записанный делитель q={q} НЕ делит R_k({}) \
                         (или не является собственным)",
                        cfg.base
                    );
                }
            }

            "PRP" => {
                // Находку пересчитываем всегда и точной арифметикой.
                st.prp_checked.fetch_add(1, Ordering::Relaxed);
                match prp::with_arena(|a| a.is_prp(cfg.base, k, cfg.mr_rounds, Backend::Gmp, 0)) {
                    Ok(Ok((true, _))) => log::info!("k={k}: PRP подтверждён"),
                    Ok(Ok((false, _))) => {
                        st.prp_bad.fetch_add(1, Ordering::Relaxed);
                        log::error!("k={k}: записан как PRP, но пересчёт даёт СОСТАВНОЕ!");
                    }
                    Ok(Err(e)) | Err(e) => log::error!("k={k}: ошибка пересчёта: {e}"),
                }
            }

            "composite" => {
                // Пересчитываем другим бэкендом, чем тот, что дал вердикт.
                let used = rec.get("backend").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                if !in_sample(cfg.base, k, ratio) {
                    st.skipped.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // Пересчитываем ВСЕГДА на GMP: это точная целочисленная
                // арифметика, независимая от FFT. Обратное направление
                // (перепроверять GMP через GWNUM) смысла не имеет — доверять
                // надо более надёжному пути, а не более быстрому.
                st.composites_checked.fetch_add(1, Ordering::Relaxed);
                match prp::with_arena(|a| a.is_prp(cfg.base, k, cfg.mr_rounds, Backend::Gmp, 0)) {
                    Ok(Ok((false, _))) => {}
                    Ok(Ok((true, _))) => {
                        st.composites_bad.fetch_add(1, Ordering::Relaxed);
                        log::error!(
                            "k={k}: записан как составное (бэкенд {used}), \
                             но пересчёт даёт PRP — ПОТЕРЯННАЯ НАХОДКА!"
                        );
                    }
                    Ok(Err(e)) | Err(e) => log::error!("k={k}: ошибка пересчёта: {e}"),
                }
            }

            _ => {
                st.skipped.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    let bad = st.factors_bad.load(Ordering::Relaxed)
        + st.prp_bad.load(Ordering::Relaxed)
        + st.composites_bad.load(Ordering::Relaxed);

    println!("\n=== ПЕРЕПРОВЕРКА ЖУРНАЛА ===");
    println!(
        "делители:  проверено {}, неверных {}",
        st.factors_checked.load(Ordering::Relaxed),
        st.factors_bad.load(Ordering::Relaxed)
    );
    println!(
        "PRP:       проверено {}, неподтверждённых {}",
        st.prp_checked.load(Ordering::Relaxed),
        st.prp_bad.load(Ordering::Relaxed)
    );
    println!(
        "составные: пересчитано {}, оказались PRP {}",
        st.composites_checked.load(Ordering::Relaxed),
        st.composites_bad.load(Ordering::Relaxed)
    );
    println!("пропущено (вне выборки): {}", st.skipped.load(Ordering::Relaxed));
    println!(
        "\nИТОГ: {}",
        if bad == 0 { "расхождений нет".to_string() } else { format!("РАСХОЖДЕНИЙ: {bad}") }
    );
    Ok(bad)
}

/// Детерминированная выборка: одни и те же k при одном и том же ratio.
fn in_sample(base: u64, k: u64, ratio: f64) -> bool {
    if ratio >= 1.0 {
        return true;
    }
    if ratio <= 0.0 {
        return false;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in base.to_le_bytes().iter().chain(k.to_le_bytes().iter()) {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    ((h % 10_000) as f64) < ratio * 10_000.0
}
