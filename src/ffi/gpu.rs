//! RAII-обёртка над CUDA trial-factoring контекстом (v2, batch API).
//!
//! МОДЕЛЬ РАБОТЫ:
//!
//! 1. `upload_ks(&ks)` — загружаем батч показателей (до 8192) один раз;
//! 2. `tf_batch(...)` — гоняем диапазоны m; для каждого k проверяются
//!    кандидаты q = 2*m*k + 1;
//! 3. результат — до `TF_MAX_FACTORS` попаданий + счётчик реально
//!    протестированных кандидатов (вход для тюнера).
//!
//! БЕЗОПАСНОСТЬ:
//!   * контекст привязан к устройству, `!Sync` — одновременный доступ запрещён;
//!   * вся device-память живёт внутри C-контекста и освобождается в `Drop`;
//!   * Rust получает только копии значений в `#[repr(C)]`-структурах на стеке.

use super::Status;
use std::os::raw::{c_char, c_int};

/// Совпадает с RH_TF_MAX_FACTORS в native/include/rh_common.h.
pub const TF_MAX_FACTORS: usize = 256;

/// Ширина модульной арифметики на устройстве. Совпадает с `rh_width_t`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Width {
    W64 = 0,
    W96 = 1,
    W128 = 2,
}

impl Width {
    /// Минимальная ширина, покрывающая все q < `q_max`.
    /// Границы — требования REDC: 2^63 / 2^95 / 2^127.
    pub fn for_qmax(q_max: u128) -> Width {
        if q_max < (1u128 << 63) {
            Width::W64
        } else if q_max < (1u128 << 95) {
            Width::W96
        } else {
            Width::W128
        }
    }

    /// Максимальное q, которое ширина может обработать (исключительно).
    pub fn q_limit(self) -> u128 {
        match self {
            Width::W64 => 1u128 << 63,
            Width::W96 => 1u128 << 95,
            Width::W128 => 1u128 << 127,
        }
    }
}

/// Найденный делитель: q = q_hi*2^64 + q_lo для k = ks[k_index].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TfHit {
    pub q_lo: u64,
    pub q_hi: u64,
    pub k_index: u32,
    pub _pad: u32,
}

impl TfHit {
    pub fn q(&self) -> u128 {
        ((self.q_hi as u128) << 64) | self.q_lo as u128
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfResult {
    pub hits: [TfHit; TF_MAX_FACTORS],
    pub candidates_tested: u64,
    pub count: u32,
    /// Сколько попаданий не влезло в буфер (0 в норме).
    pub lost: u32,
}

impl Default for TfResult {
    fn default() -> Self {
        Self {
            hits: [TfHit::default(); TF_MAX_FACTORS],
            candidates_tested: 0,
            count: 0,
            lost: 0,
        }
    }
}

impl TfResult {
    pub fn hits(&self) -> &[TfHit] {
        &self.hits[..(self.count as usize).min(TF_MAX_FACTORS)]
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuInfo {
    pub device_id: c_int,
    pub sm_count: c_int,
    pub max_threads_per_sm: c_int,
    pub cc_major: c_int,
    pub cc_minor: c_int,
    pub clock_khz: c_int,
    pub global_mem_bytes: u64,
    pub name: [c_char; 128],
}

impl GpuInfo {
    pub fn name_str(&self) -> String {
        let bytes: Vec<u8> = self
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[repr(C)]
struct RhGpuCtx {
    _private: [u8; 0],
}

#[cfg(rh_cuda)]
extern "C" {
    fn rh_gpu_init(device_id: c_int, out: *mut *mut RhGpuCtx) -> c_int;
    fn rh_gpu_destroy(ctx: *mut RhGpuCtx);
    fn rh_gpu_query(device_id: c_int, info: *mut GpuInfo) -> c_int;
    fn rh_gpu_device_count() -> c_int;
    fn rh_gpu_upload_ks(ctx: *mut RhGpuCtx, ks: *const u64, n: u32) -> c_int;
    fn rh_gpu_tf_batch(
        ctx: *mut RhGpuCtx,
        base: u64,
        m_start: u64,
        m_span: u64,
        width: u32,
        res: *mut TfResult,
    ) -> c_int;
    fn rh_gpu_suggest_span(ctx: *const RhGpuCtx, n_k: u32, width: u32) -> u64;
}

/// Максимум показателей в одном батче — совпадает с `MAX_K` в tf_host.cu.
pub const MAX_K_PER_BATCH: usize = 8192;

pub struct GpuContext {
    #[cfg(rh_cuda)]
    ptr: *mut RhGpuCtx,
    /// Сколько показателей загружено последним upload_ks — страховка от
    /// вызова tf_batch до загрузки батча.
    n_k: usize,
}

// SAFETY: контекст можно ПЕРЕДАТЬ в другой поток (владение), но не разделять:
// `!Sync` обеспечивается сырым указателем внутри.
unsafe impl Send for GpuContext {}

impl GpuContext {
    /// `device_id < 0` — выбрать устройство с максимальным числом SM.
    #[cfg(rh_cuda)]
    pub fn new(device_id: i32) -> anyhow::Result<Self> {
        let mut ptr: *mut RhGpuCtx = std::ptr::null_mut();
        // SAFETY: out — валидный указатель на стеке.
        let st = unsafe { rh_gpu_init(device_id, &mut ptr) };
        Status::from_raw(st).ok()?;
        anyhow::ensure!(!ptr.is_null(), "rh_gpu_init вернул NULL-контекст");
        Ok(Self { ptr, n_k: 0 })
    }

    #[cfg(not(rh_cuda))]
    pub fn new(_device_id: i32) -> anyhow::Result<Self> {
        anyhow::bail!("сборка без CUDA")
    }

    #[cfg(rh_cuda)]
    pub fn device_count() -> usize {
        // SAFETY: чистая функция без аргументов.
        (unsafe { rh_gpu_device_count() }).max(0) as usize
    }
    #[cfg(not(rh_cuda))]
    pub fn device_count() -> usize {
        0
    }

    #[cfg(rh_cuda)]
    pub fn query(device_id: i32) -> anyhow::Result<GpuInfo> {
        let mut info = std::mem::MaybeUninit::<GpuInfo>::uninit();
        // SAFETY: C полностью заполняет структуру при RH_OK.
        let st = unsafe { rh_gpu_query(device_id, info.as_mut_ptr()) };
        Status::from_raw(st).ok()?;
        Ok(unsafe { info.assume_init() })
    }
    #[cfg(not(rh_cuda))]
    pub fn query(_device_id: i32) -> anyhow::Result<GpuInfo> {
        anyhow::bail!("сборка без CUDA")
    }

    /// Загрузить батч показателей. Инвалидирует кэш CUDA-графов на устройстве.
    #[cfg(rh_cuda)]
    pub fn upload_ks(&mut self, ks: &[u64]) -> anyhow::Result<()> {
        anyhow::ensure!(!ks.is_empty(), "пустой батч k");
        anyhow::ensure!(ks.len() <= MAX_K_PER_BATCH, "батч k > {MAX_K_PER_BATCH}");
        // SAFETY: срез валиден на время вызова; C копирует его немедленно.
        let st = unsafe { rh_gpu_upload_ks(self.ptr, ks.as_ptr(), ks.len() as u32) };
        Status::from_raw(st).ok()?;
        self.n_k = ks.len();
        Ok(())
    }
    #[cfg(not(rh_cuda))]
    pub fn upload_ks(&mut self, _ks: &[u64]) -> anyhow::Result<()> {
        anyhow::bail!("сборка без CUDA")
    }

    /// Прогнать m ∈ [m_start, m_start+m_span) для всех загруженных k.
    #[cfg(rh_cuda)]
    pub fn tf_batch(
        &mut self,
        base: u64,
        m_start: u64,
        m_span: u64,
        width: Width,
    ) -> anyhow::Result<TfResult> {
        anyhow::ensure!(self.n_k > 0, "tf_batch до upload_ks");
        let mut res = TfResult::default();
        // SAFETY: res — валидная запись; C не сохраняет указатель после возврата.
        let st = unsafe {
            rh_gpu_tf_batch(self.ptr, base, m_start, m_span, width as u32, &mut res)
        };
        Status::from_raw(st).ok()?;
        if res.lost != 0 {
            // Потерянные делители не портят результат — их кандидаты просто
            // уйдут в PRP, — но это дорого, и о таком надо знать.
            log::warn!(
                "TF: потеряно {} попаданий сверх буфера ({TF_MAX_FACTORS}) на m={m_start}",
                res.lost
            );
        }
        Ok(res)
    }
    #[cfg(not(rh_cuda))]
    pub fn tf_batch(
        &mut self,
        _base: u64,
        _m_start: u64,
        _m_span: u64,
        _width: Width,
    ) -> anyhow::Result<TfResult> {
        anyhow::bail!("сборка без CUDA")
    }

    /// Рекомендуемый размер шага по m (из occupancy), ~30-80 мс на launch.
    #[cfg(rh_cuda)]
    pub fn suggest_span(&self, n_k: usize, width: Width) -> u64 {
        // SAFETY: ptr валиден, функция читает только поля контекста.
        unsafe { rh_gpu_suggest_span(self.ptr, n_k as u32, width as u32) }
    }
    #[cfg(not(rh_cuda))]
    pub fn suggest_span(&self, _n_k: usize, _width: Width) -> u64 {
        1 << 16
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        #[cfg(rh_cuda)]
        {
            if !self.ptr.is_null() {
                // SAFETY: единственный владелец; C-функция идемпотентна.
                unsafe { rh_gpu_destroy(self.ptr) };
                self.ptr = std::ptr::null_mut();
            }
        }
    }
}
