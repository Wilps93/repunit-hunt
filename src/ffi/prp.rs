//! RAII-обёртка над GMP/GWNUM PRP-ареной + thread-local пул.

use super::{check, NativeError};
use std::cell::RefCell;
use std::os::raw::{c_char, c_int, c_uint};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend { Auto = 0, Gmp = 1, Gwnum = 2 }

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PrpStat {
    pub bits: u64,
    pub squarings: u64,
    pub elapsed_sec: f64,
    pub backend_used: u32,
    pub gerbicz_checks: u32,
    pub max_roundoff: f64,
}

#[repr(C)]
pub(crate) struct RhArena { _p: [u8; 0] }

extern "C" {
    fn rh_prp_arena_new() -> *mut RhArena;
    fn rh_prp_arena_free(a: *mut RhArena);
    fn rh_prp_arena_reserve(a: *mut RhArena, bits: u64);
    fn rh_prp_test(a: *mut RhArena, base: u64, k: u64, mr: c_uint,
                   backend: u32, gerbicz_l: u32, st: *mut PrpStat) -> c_int;
    fn rh_prp_verify_factor(a: *mut RhArena, base: u64, k: u64, q_lo: u64, q_hi: u64) -> c_int;
    fn rh_prp_decimal(a: *mut RhArena, base: u64, k: u64, buf: *mut c_char, len: usize) -> usize;
    fn rh_gwnum_available() -> c_int;
    fn rh_gmp_pool_hiwater() -> usize;
    fn rh_gmp_pool_capacity() -> usize;
}

// SAFETY: чистая функция без аргументов, состояния не трогает.
pub fn gwnum_available() -> bool { unsafe { rh_gwnum_available() != 0 } }

pub struct PrpArena { ptr: *mut RhArena }
// SAFETY: перемещение между потоками безопасно; разделение — нет.
unsafe impl Send for PrpArena {}

impl PrpArena {
    pub fn new() -> Result<Self, NativeError> {
        // SAFETY: конструктор возвращает валидный указатель или NULL.
        let p = unsafe { rh_prp_arena_new() };
        if p.is_null() { return Err(NativeError::NoMem); }
        Ok(Self { ptr: p })
    }

    pub fn reserve_bits(&mut self, bits: u64) {
        // SAFETY: ptr валиден.
        unsafe { rh_prp_arena_reserve(self.ptr, bits) }
    }

    pub fn is_prp(&mut self, base: u64, k: u64, mr: u32, backend: Backend, gerbicz_l: u32)
        -> Result<(bool, PrpStat), NativeError>
    {
        let mut st = PrpStat::default();
        // SAFETY: st — валидная запись; C не сохраняет указатель.
        let r = unsafe { rh_prp_test(self.ptr, base, k, mr, backend as u32, gerbicz_l, &mut st) };
        if r < 0 { check(r)?; unreachable!() }
        Ok((r == 1, st))
    }

    pub fn verify_factor(&mut self, base: u64, k: u64, q: u128) -> Result<bool, NativeError> {
        let lo = q as u64;
        let hi = (q >> 64) as u64;
        // SAFETY: ptr валиден.
        let r = unsafe { rh_prp_verify_factor(self.ptr, base, k, lo, hi) };
        if r < 0 { check(r)?; }
        Ok(r == 1)
    }

    pub fn decimal(&mut self, base: u64, k: u64) -> Result<String, NativeError> {
        // SAFETY: первый вызов с NULL только вычисляет длину.
        let need = unsafe { rh_prp_decimal(self.ptr, base, k, std::ptr::null_mut(), 0) };
        let mut buf = vec![0u8; need + 2];
        // SAFETY: буфер достаточного размера.
        let w = unsafe { rh_prp_decimal(self.ptr, base, k, buf.as_mut_ptr() as *mut c_char, buf.len()) };
        buf.truncate(w);
        String::from_utf8(buf).map_err(|_| NativeError::Internal)
    }

    /// Сырой указатель на арену — для соседних FFI-модулей (см. `ffi::pm1`).
    /// Требует `&mut`, поэтому одновременный доступ к C-состоянию исключён.
    pub(crate) fn as_ptr(&mut self) -> *mut RhArena { self.ptr }

    /// (пик использования, ёмкость) thread-local пула GMP, в байтах.
    pub fn pool_usage() -> (usize, usize) {
        // SAFETY: чистые геттеры thread-local переменных.
        unsafe { (rh_gmp_pool_hiwater(), rh_gmp_pool_capacity()) }
    }
}

impl Drop for PrpArena {
    fn drop(&mut self) {
        // SAFETY: единственный владелец, вызывается один раз.
        unsafe { rh_prp_arena_free(self.ptr) };
        self.ptr = std::ptr::null_mut();
    }
}

thread_local! {
    static TLS: RefCell<Option<PrpArena>> = const { RefCell::new(None) };
}

/// Выполнить операцию с thread-local ареной (создаётся лениво, один раз на поток).
pub fn with_arena<R>(f: impl FnOnce(&mut PrpArena) -> R) -> Result<R, NativeError> {
    TLS.with(|c| {
        let mut o = c.borrow_mut();
        if o.is_none() { *o = Some(PrpArena::new()?); }
        Ok(f(o.as_mut().unwrap()))
    })
}