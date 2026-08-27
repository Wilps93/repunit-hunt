//! CPU affinity, NUMA-локальность и huge pages.
//!
//! ПРОБЛЕМА: rayon по умолчанию позволяет ОС мигрировать потоки между
//! NUMA-узлами. Для GMP/GWNUM-нагрузки (работа с многомегабайтными буферами)
//! это даёт до 2x падение из-за remote memory access.
//!
//! РЕШЕНИЕ: прибиваем worker i к логическому ядру i (round-robin по узлам),
//! а память арены аллоцируем ПОСЛЕ pinning => first-touch policy кладёт
//! страницы на локальный узел.

#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpu: usize) -> bool {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread(_cpu: usize) -> bool { false }

/// Включить THP для процесса (влияет на mmap в rh_alloc.c).
#[cfg(target_os = "linux")]
pub fn enable_thp() {
    unsafe {
        // PR_SET_THP_DISABLE = 41; передаём 0 => НЕ отключать
        libc::prctl(41, 0isize, 0isize, 0isize, 0isize);
    }
}
#[cfg(not(target_os = "linux"))]
pub fn enable_thp() {}

/// Топология: сколько логических CPU, сколько NUMA-узлов.
pub struct Topology { pub cpus: usize, pub nodes: usize }

pub fn topology() -> Topology {
    let cpus = num_cpus::get();
    let nodes = detect_numa_nodes();
    Topology { cpus, nodes }
}

#[cfg(target_os = "linux")]
fn detect_numa_nodes() -> usize {
    std::fs::read_dir("/sys/devices/system/node")
        .map(|d| d.filter_map(|e| e.ok())
                  .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                  .count())
        .unwrap_or(1).max(1)
}
#[cfg(not(target_os = "linux"))]
fn detect_numa_nodes() -> usize { 1 }

/// Порядок ядер для pinning: сначала заполняем узел 0, потом узел 1 и т.д.
/// Это лучше, чем round-robin, т.к. при частичной загрузке все потоки
/// оказываются на одном узле (общий L3).
pub fn cpu_order() -> Vec<usize> {
    let t = topology();
    if t.nodes <= 1 { return (0..t.cpus).collect(); }
    let per_node = t.cpus / t.nodes;
    let mut v = Vec::with_capacity(t.cpus);
    for n in 0..t.nodes {
        for i in 0..per_node { v.push(n * per_node + i); }
    }
    while v.len() < t.cpus { let l = v.len(); v.push(l); }
    v
}