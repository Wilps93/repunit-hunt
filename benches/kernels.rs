//! Микробенчмарк и кросс-проверка PRP-бэкендов (harness = false, без criterion).
//!
//! Бенчмарк намеренно ходит напрямую в C-ABI: у пакета нет lib-таргета,
//! а флаги линковки из build.rs распространяются на все таргеты, поэтому
//! `rh_*`-символы здесь доступны так же, как в основном бинаре.
//!
//! Что делает:
//!   1. гоняет PRP по набору k и печатает время;
//!   2. если собран GWNUM — прогоняет КАЖДОЕ число обоими бэкендами и сверяет
//!      вердикты. Расхождение означает ошибку в одном из путей (FFT, Gerbicz,
//!      конвертация gwnum->mpz) и валит бенчмарк.
//!
//! Запуск:  cargo bench --bench kernels
//!          RH_BENCH_BASE=10 RH_BENCH_KS=1279,2203,4253 cargo bench --bench kernels

use std::os::raw::{c_int, c_uint};
use std::time::Instant;

const BACKEND_GMP: u32 = 1;
const BACKEND_GWNUM: u32 = 2;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct PrpStat {
    bits: u64,
    squarings: u64,
    elapsed_sec: f64,
    backend_used: u32,
    gerbicz_checks: u32,
    max_roundoff: f64,
}

#[repr(C)]
struct RhArena {
    _p: [u8; 0],
}

extern "C" {
    fn rh_prp_arena_new() -> *mut RhArena;
    fn rh_prp_arena_free(a: *mut RhArena);
    fn rh_prp_arena_reserve(a: *mut RhArena, bits: u64);
    fn rh_prp_test(
        a: *mut RhArena,
        base: u64,
        k: u64,
        mr: c_uint,
        backend: u32,
        gerbicz_l: u32,
        st: *mut PrpStat,
    ) -> c_int;
    fn rh_prp_verify_factor(a: *mut RhArena, base: u64, k: u64, q_lo: u64, q_hi: u64) -> c_int;
    fn rh_gmp_pool_hiwater() -> usize;
    fn rh_gwnum_available() -> c_int;
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn verdict(r: c_int) -> &'static str {
    match r {
        1 => "PRP",
        0 => "составное",
        -7 => "ошибка FFT",
        -8 => "СБОЙ Gerbicz",
        -9 => "нет бэкенда",
        _ => "ошибка",
    }
}

fn main() {
    let base = env_u64("RH_BENCH_BASE", 10);
    let ks: Vec<u64> = std::env::var("RH_BENCH_KS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![317, 1031, 2003, 5003, 10007, 21001]);

    // SAFETY: арена создаётся и освобождается ровно один раз в этом потоке.
    let arena = unsafe { rh_prp_arena_new() };
    assert!(!arena.is_null(), "не удалось создать арену");

    // SAFETY: чистая функция без аргументов.
    let has_gw = unsafe { rh_gwnum_available() } != 0;
    println!("repunit-hunt bench | base={base} | GWNUM={}", if has_gw { "есть" } else { "нет" });

    if has_gw {
        println!(
            "\n{:>7} {:>9} {:>10} {:>10} {:>7} {:>12} {:>10} {:>8}",
            "k", "бит", "GMP,с", "GWNUM,с", "ускор", "вердикт", "roundoff", "Gerbicz"
        );
    } else {
        println!("\n{:>7} {:>9} {:>10} {:>12}", "k", "бит", "GMP,с", "вердикт");
    }

    let mut mismatches = 0;

    for &k in &ks {
        let bits_est = (k as f64 * (base as f64).log2()) as u64 + 64;
        // SAFETY: указатель валиден, арена принадлежит этому потоку.
        unsafe { rh_prp_arena_reserve(arena, bits_est) };

        // ── GMP ──
        let mut st_gmp = PrpStat::default();
        let t0 = Instant::now();
        // SAFETY: st — валидная запись; C не сохраняет указатель.
        let r_gmp = unsafe { rh_prp_test(arena, base, k, 0, BACKEND_GMP, 0, &mut st_gmp) };
        let t_gmp = t0.elapsed().as_secs_f64();

        if !has_gw {
            println!("{k:>7} {:>9} {t_gmp:>10.3} {:>12}", st_gmp.bits, verdict(r_gmp));
            continue;
        }

        // ── GWNUM ──
        let mut st_gw = PrpStat::default();
        let t1 = Instant::now();
        // SAFETY: те же инварианты, что и выше.
        let r_gw = unsafe { rh_prp_test(arena, base, k, 0, BACKEND_GWNUM, 0, &mut st_gw) };
        let t_gw = t1.elapsed().as_secs_f64();

        let agree = r_gmp == r_gw;
        if !agree {
            mismatches += 1;
        }
        println!(
            "{k:>7} {:>9} {t_gmp:>10.3} {t_gw:>10.3} {:>6.2}x {:>12} {:>10.4} {:>8}{}",
            st_gmp.bits,
            if t_gw > 0.0 { t_gmp / t_gw } else { 0.0 },
            verdict(r_gmp),
            st_gw.max_roundoff,
            st_gw.gerbicz_checks,
            if agree { "" } else { "  <-- РАСХОЖДЕНИЕ" }
        );
    }

    // Проверка корректности verify_factor: 111 = 3·37 для base=10, k=3.
    // SAFETY: та же арена, валидные аргументы.
    let ok = unsafe { rh_prp_verify_factor(arena, 10, 3, 37, 0) };
    assert_eq!(ok, 1, "verify_factor: 37 должно делить R_3(10) = 111");
    // Сам кандидат делителем не считается: R_2(10) = 11.
    // SAFETY: см. выше.
    let self_div = unsafe { rh_prp_verify_factor(arena, 10, 2, 11, 0) };
    assert_eq!(self_div, 0, "verify_factor: q = N не является собственным делителем");

    // SAFETY: чистый геттер thread-local переменной.
    println!("\nПик GMP-пула: {} КиБ", unsafe { rh_gmp_pool_hiwater() } / 1024);

    // SAFETY: арена больше не используется.
    unsafe { rh_prp_arena_free(arena) };

    if mismatches > 0 {
        eprintln!("РАСХОЖДЕНИЙ GMP vs GWNUM: {mismatches}");
        std::process::exit(1);
    }
    if has_gw {
        println!("вердикты GMP и GWNUM совпали на всех числах");
    }
}
