//! Безопасные обёртки над нативными слоями.
//! Инварианты:
//!  * все сырые указатели инкапсулированы, освобождение — в Drop (RAII);
//!  * ни один тип не является Sync (одновременный доступ к C-состоянию запрещён);
//!  * все входные данные — примитивы или срезы с явным временем жизни.

pub mod gpu;
pub mod pm1;
pub mod prp;

use std::os::raw::c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NativeError {
    #[error("out of memory")]            NoMem,
    #[error("invalid argument")]         InvalidArg,
    #[error("CUDA error")]               Cuda,
    #[error("no CUDA device")]           NoDevice,
    #[error("arithmetic overflow")]      Overflow,
    #[error("internal error")]           Internal,
    #[error("FFT roundoff too large")]   FftError,
    #[error("Gerbicz check FAILED — вероятна аппаратная ошибка!")] Gerbicz,
    #[error("backend not compiled in")]  NoBackend,
}

pub fn check(v: c_int) -> Result<(), NativeError> {
    use NativeError::*;
    match v {
        0 => Ok(()),
        -1 => Err(NoMem),
        -2 => Err(InvalidArg),
        -3 => Err(Cuda),
        -4 => Err(NoDevice),
        -5 => Err(Overflow),
        -7 => Err(FftError),
        -8 => Err(Gerbicz),
        -9 => Err(NoBackend),
        _ => Err(Internal),
    }
}

/// Тонкая обёртка над кодом возврата нативного слоя.
/// Существует, чтобы вызывающий не мог случайно проигнорировать статус:
/// единственный способ им воспользоваться — `ok()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Status(pub c_int);

impl Status {
    #[inline]
    pub fn from_raw(v: c_int) -> Self { Self(v) }
    #[inline]
    pub fn ok(self) -> Result<(), NativeError> { check(self.0) }
}
