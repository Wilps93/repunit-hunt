//! Безопасная обёртка над этапом P-1 (native/prp/pm1_ecm.c, libecm).
//!
//! ЗАЧЕМ ЭТАП НУЖЕН. Любой простой делитель q числа R_k(b) имеет вид
//! q = 2*m*k + 1, то есть q-1 УЖЕ содержит известный крупный множитель k.
//! P-1 стартует с сида 2^(2k) и потому «бесплатно» покрывает этот множитель:
//! нужная гладкость требуется только от m. За ~1% стоимости PRP этап снимает
//! несколько процентов кандидатов — см. `Tuner::pm1_worth_it`.
//!
//! Если libecm не подключён (нет ECM_DIR / RH_ECM_SYSTEM), нативная сторона
//! возвращает `RH_ERR_NO_BACKEND`, а `available()` — `false`.

use super::prp::{PrpArena, RhArena};
use super::{check, NativeError};
use std::os::raw::{c_char, c_int};

/// Зеркало `rh_pm1_params_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Pm1Params {
    pub b1: u64,
    pub b2: u64,
    /// Известный множитель показателя (сам k): сид берётся как 2^(2k).
    pub k_known: u64,
}

extern "C" {
    fn rh_pm1_available() -> c_int;
    fn rh_pm1_factor(
        a: *mut RhArena,
        base: u64,
        k: u64,
        p: *const Pm1Params,
        out: *mut c_char,
        len: usize,
    ) -> c_int;
}

/// Собран ли P-1-бэкенд (libecm).
pub fn available() -> bool {
    // SAFETY: чистая функция без аргументов.
    unsafe { rh_pm1_available() != 0 }
}

/// Запустить P-1 для R_k(b).
///
/// `Ok(Some(f))` — найден делитель (десятичная строка), `Ok(None)` — не найден.
pub fn factor(
    arena: &mut PrpArena,
    base: u64,
    k: u64,
    params: &Pm1Params,
) -> Result<Option<String>, NativeError> {
    // Делители могут быть крупными (P-1 иногда вытаскивает и составной
    // кофактор), поэтому буфер с большим запасом.
    let mut buf = vec![0u8; 4096];
    let ptr = arena.as_ptr();
    // SAFETY: arena взята по &mut (эксклюзивный доступ к C-состоянию);
    // buf живёт до конца функции, длина передаётся честно.
    let r = unsafe {
        rh_pm1_factor(
            ptr,
            base,
            k,
            params as *const Pm1Params,
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
        )
    };
    if r < 0 {
        check(r)?;
    }
    if r != 1 {
        return Ok(None);
    }
    let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    buf.truncate(end);
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| NativeError::Internal)
}
