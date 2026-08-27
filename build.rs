//! Сборка нативных слоёв v2:
//!   CUDA: tf_kernel.cu (3 специализации ширины) + tf_host.cu (graphs/streams)
//!   C:    GMP-бэкенд, GWNUM-бэкенд, pool-allocator, P-1 (libecm)
//!
//! Опциональные бэкенды выбираются фичами; отсутствие библиотеки не ломает сборку
//! ядра — просто отключается соответствующий путь.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUDA_SRC: &[&str] = &["native/cuda/tf_kernel.cu", "native/cuda/tf_host.cu"];
const INC: &str = "native/include";

/// Volta..Hopper + PTX для forward-compat (Blackwell JIT).
const DEFAULT_ARCHS: &[&str] = &["70", "75", "80", "86", "89", "90"];

fn main() -> anyhow::Result<()> {
    for f in CUDA_SRC { println!("cargo:rerun-if-changed={f}"); }
    for f in ["native/prp", INC, "native/cuda", "build.rs"] {
        println!("cargo:rerun-if-changed={f}");
    }
    for v in ["CUDA_PATH","CUDA_HOME","RH_CUDA_ARCHS","GMP_DIR","GWNUM_DIR","ECM_DIR",
              "RH_MAXREG","RH_PORTABLE","RH_GMP_STATIC"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    // rustc >= 1.80 требует декларации нестандартных cfg
    for c in ["rh_cuda", "rh_gwnum", "rh_pm1"] {
        println!("cargo:rustc-check-cfg=cfg({c})");
    }

    let out = PathBuf::from(env::var("OUT_DIR")?);

    build_c_layer(&out)?;
    link_math_libs()?;

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        match find_cuda_root() {
            Some(root) => {
                build_cuda(&root, &out)?;
                link_cuda(&root)?;
                println!("cargo:rustc-cfg=rh_cuda");
            }
            None => println!("cargo:warning=CUDA toolkit не найден — сборка без GPU TF."),
        }
    }
    Ok(())
}

// ───────────────────────── C-слой ─────────────────────────
fn build_c_layer(_out: &Path) -> anyhow::Result<()> {
    let gwnum = env::var("CARGO_FEATURE_GWNUM").is_ok() && env::var("GWNUM_DIR").is_ok();
    // libecm не поставляет pkg-config, поэтому включаем P-1 только при явном
    // указании: ECM_DIR=<префикс> или RH_ECM_SYSTEM=1 (libecm в системных путях).
    let pm1 = env::var("CARGO_FEATURE_PM1").is_ok()
        && (env::var("ECM_DIR").is_ok() || env::var("RH_ECM_SYSTEM").is_ok());
    if env::var("CARGO_FEATURE_PM1").is_ok() && !pm1 {
        println!("cargo:warning=libecm не указан (ECM_DIR / RH_ECM_SYSTEM) — этап P-1 отключён.");
    }
    if env::var("CARGO_FEATURE_GWNUM").is_ok() && !gwnum {
        println!("cargo:warning=GWNUM_DIR не задан — PRP пойдёт через GMP.");
    }

    let mut b = cc::Build::new();
    // prp_gwnum.c и pm1_ecm.c компилируются ВСЕГДА: без RH_HAVE_GWNUM /
    // RH_HAVE_ECM они сводятся к заглушкам, возвращающим RH_ERR_NO_BACKEND.
    // Иначе rh_gwnum_available() / rh_pm1_available() остались бы
    // неразрешёнными символами — их зовут диспетчер и Rust-слой.
    //
    // prp_gwnum.c при этом собирается ОТДЕЛЬНО (см. ниже): в корне SDK
    // Prime95 лежит собственный ecm.h, который при общем -I перекрывает
    // системный заголовок GMP-ECM и ломает сборку pm1_ecm.c.
    b.include(INC)
     .file("native/prp/rh_alloc.c")
     .file("native/prp/rh_arena.c")
     .file("native/prp/prp_gmp.c")
     .file("native/prp/prp_dispatch.c")
     .file("native/prp/pm1_ecm.c")
     .flag_if_supported("-O3")
     .flag_if_supported("-std=gnu11")
     .flag_if_supported("-fno-plt")
     .flag_if_supported("-fvisibility=hidden")
     .flag_if_supported("-fno-semantic-interposition")
     .warnings(false);

    if env::var("RH_PORTABLE").is_err() {
        b.flag_if_supported("-march=native").flag_if_supported("-mtune=native");
    }
    if let Ok(d) = env::var("GMP_DIR") { b.include(format!("{d}/include")); }

    if pm1 {
        if let Ok(d) = env::var("ECM_DIR") { b.include(format!("{d}/include")); }
        b.define("RH_HAVE_ECM", "1");
        println!("cargo:rustc-cfg=rh_pm1");
    }

    b.compile("rh_native");

    // ── GWNUM-слой отдельной библиотекой, со своими include-путями ──
    let mut g = cc::Build::new();
    g.include(INC)
     .file("native/prp/prp_gwnum.c")
     .flag_if_supported("-O3")
     .flag_if_supported("-std=gnu11")
     .flag_if_supported("-fno-plt")
     .flag_if_supported("-fvisibility=hidden")
     .warnings(false);
    if env::var("RH_PORTABLE").is_err() {
        g.flag_if_supported("-march=native").flag_if_supported("-mtune=native");
    }
    if let Ok(d) = env::var("GMP_DIR") { g.include(format!("{d}/include")); }
    if gwnum {
        let gw = env::var("GWNUM_DIR").unwrap();
        // Заголовки лежат либо в корне SDK, либо в подкаталоге gwnum/.
        // Берём именно тот каталог, где реально есть gwnum.h.
        let hdr = if Path::new(&format!("{gw}/gwnum.h")).exists() {
            gw.clone()
        } else {
            format!("{gw}/gwnum")
        };
        g.include(hdr).define("RH_HAVE_GWNUM", "1");
        println!("cargo:rustc-cfg=rh_gwnum");
    }
    g.compile("rh_gwnum_layer");

    Ok(())
}

fn link_math_libs() -> anyhow::Result<()> {
    if let Ok(d) = env::var("GMP_DIR")  { println!("cargo:rustc-link-search=native={d}/lib"); }
    if let Ok(d) = env::var("ECM_DIR")  { println!("cargo:rustc-link-search=native={d}/lib"); }
    if let Ok(d) = env::var("GWNUM_DIR"){
        println!("cargo:rustc-link-search=native={d}/gwnum");
        println!("cargo:rustc-link-search=native={d}");
    }

    if let Ok(o) = Command::new("pkg-config").args(["--libs-only-L","gmp"]).output() {
        if o.status.success() {
            for t in String::from_utf8_lossy(&o.stdout).split_whitespace() {
                if let Some(p) = t.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={p}");
                }
            }
        }
    }

    // GWNUM — статическая библиотека. В SDK она называется gwnum.a и лежит
    // в подкаталоге платформы (linux64/, macosx64/, ...), а rustc ищет
    // libgwnum.a, поэтому копируем её в OUT_DIR под нужным именем.
    if env::var("CARGO_FEATURE_GWNUM").is_ok() {
        if let Ok(d) = env::var("GWNUM_DIR") {
            match locate_gwnum_lib(&d) {
                Some(src) => {
                    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
                    let dst = out.join("libgwnum.a");
                    match std::fs::copy(&src, &dst) {
                        Ok(_) => {
                            println!("cargo:rustc-link-search=native={}", out.display());
                            println!("cargo:rustc-link-lib=static=gwnum");
                            // Ассемблерные FFT-модули GWNUM собраны без -fPIC
                            // (R_X86_64_64 против локальных символов), поэтому
                            // итоговый бинарь линкуем как non-PIE. Касается
                            // только целей, которые реально тянут GWNUM.
                            println!("cargo:rustc-link-arg=-no-pie");
                        }
                        Err(e) => println!(
                            "cargo:warning=не удалось скопировать {}: {e}", src.display()),
                    }
                }
                None => println!(
                    "cargo:warning=gwnum.a не найдена в {d} (искали в linux64/, linux/,                      gwnum/linux64/, gwnum/linux/) — GWNUM-путь будет отключён"),
            }
        }
    }
    // Линкуем libecm только когда pm1_ecm.c собран с RH_HAVE_ECM,
    // иначе он остаётся заглушкой и внешних символов не тянет.
    if env::var("CARGO_FEATURE_PM1").is_ok()
        && (env::var("ECM_DIR").is_ok() || env::var("RH_ECM_SYSTEM").is_ok())
    {
        println!("cargo:rustc-link-lib=dylib=ecm");
    }
    if env::var("RH_GMP_STATIC").is_ok() {
        println!("cargo:rustc-link-lib=static=gmp");
    } else {
        println!("cargo:rustc-link-lib=dylib=gmp");
    }
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    if cfg!(target_os="linux")      { println!("cargo:rustc-link-lib=dylib=stdc++"); }
    else if cfg!(target_os="macos") { println!("cargo:rustc-link-lib=dylib=c++"); }
    Ok(())
}

/// Ищем статическую библиотеку GWNUM в типичных местах SDK Prime95.
fn locate_gwnum_lib(root: &str) -> Option<PathBuf> {
    let plat = if cfg!(target_os = "macos") {
        ["macosx64", "macosx"]
    } else {
        ["linux64", "linux"]
    };
    // ВАЖЕН ПОРЯДОК. В SDK каталог linux64/ содержит только предсобранные
    // ассемблерные FFT-модули — без C-части (gwinit2, allocgiant, gwtogiant...).
    // Полная библиотека появляется в корне gwnum/ после `make -f make64`,
    // поэтому её ищем ПЕРВОЙ.
    let mut cands: Vec<PathBuf> = vec![
        PathBuf::from(format!("{root}/gwnum/gwnum.a")),
        PathBuf::from(format!("{root}/gwnum.a")),
        PathBuf::from(format!("{root}/libgwnum.a")),
    ];
    for sub in plat {
        cands.push(PathBuf::from(format!("{root}/{sub}/gwnum.a")));
        cands.push(PathBuf::from(format!("{root}/gwnum/{sub}/gwnum.a")));
    }
    cands.into_iter().find(|p| p.exists())
}

// ───────────────────────── CUDA ─────────────────────────
fn nvcc_name() -> &'static str { if cfg!(windows) {"nvcc.exe"} else {"nvcc"} }

fn find_cuda_root() -> Option<PathBuf> {
    for var in ["CUDA_PATH","CUDA_HOME"] {
        if let Ok(p) = env::var(var) {
            let p = PathBuf::from(p);
            if p.join("bin").join(nvcc_name()).exists() { return Some(p); }
        }
    }
    if let Ok(o) = Command::new("which").arg("nvcc").output() {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !s.is_empty() { return Path::new(&s).parent()?.parent().map(|p| p.to_path_buf()); }
        }
    }
    ["/usr/local/cuda","/opt/cuda","/usr/lib/cuda"].iter()
        .map(PathBuf::from)
        .find(|p| p.join("bin").join(nvcc_name()).exists())
}

fn archs() -> Vec<String> {
    env::var("RH_CUDA_ARCHS").ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_else(|| DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect())
}

fn build_cuda(root: &Path, out: &Path) -> anyhow::Result<()> {
    let nvcc = root.join("bin").join(nvcc_name());
    let a = archs();
    let mut objs = Vec::new();

    for src in CUDA_SRC {
        let stem = Path::new(src).file_stem().unwrap().to_string_lossy().to_string();
        let obj = out.join(format!("{stem}.o"));
        let mut c = Command::new(&nvcc);
        c.args(["-c", src, "-o"]).arg(&obj)
         .arg("-I").arg(INC)
         .args(["-O3","-std=c++17","-lineinfo","--extra-device-vectorization"])
         .args(["-Xptxas","-O3,-v"])
         .args(["-Xcompiler","-fPIC,-O3"])
         // Relocatable device code не нужен — всё в одном TU-шаблоне
         .arg("--expt-relaxed-constexpr");
        for x in &a { c.arg(format!("-gencode=arch=compute_{x},code=sm_{x}")); }
        if let Some(l) = a.last() { c.arg(format!("-gencode=arch=compute_{l},code=compute_{l}")); }
        if let Ok(m) = env::var("RH_MAXREG") { c.arg(format!("-maxrregcount={m}")); }
        anyhow::ensure!(c.status()?.success(), "nvcc failed: {src}");
        objs.push(obj);
    }

    let lib = out.join("librh_gpu.a");
    let _ = std::fs::remove_file(&lib);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    anyhow::ensure!(Command::new(ar).arg("crs").arg(&lib).args(&objs).status()?.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=rh_gpu");
    Ok(())
}

fn link_cuda(root: &Path) -> anyhow::Result<()> {
    for s in ["lib64","lib/x64","lib"] {
        let p = root.join(s);
        if p.exists() { println!("cargo:rustc-link-search=native={}", p.display()); }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}