*[Русская версия](README.md) · English*

# repunit-hunt

A hybrid CPU/GPU searcher for generalized repunit primes

$$R_k(b) = \frac{b^k - 1}{b - 1} = 1 + b + b^2 + \dots + b^{k-1}$$

Pipeline: generate k → sieve by small divisors (CPU) → trial factoring (GPU,
Montgomery 64/96/128 bit) → P−1 (libecm) → PRP (GMP, or GWNUM with a
Gerbicz–Li check).

---

The repository has two parts: the **searcher** for repunit primes
(Rust + CUDA + C) and the **paper** (`paper/`), a statistical analysis of the
sequences found.

## The paper

The directory `paper/` contains the work "The observation scheme in the
statistics of generalized repunit primes: calibrating the pooled test of the
Lenstra–Pomerance–Wagstaff constant" — in Russian
([`paper_ru.pdf`](paper/paper_ru.pdf), laid out per GOST R 7.0.7-2021) and in
English ([`paper_en.pdf`](paper/paper_en.pdf), international format).

In brief: the LPW conjecture predicts the density of repunit primes with a
universal constant e^γ ≈ 1.781. Estimates published since 1993 sat about 10%
above it — and that excess turns out to be an artifact of the observation
scheme. The conditional likelihood is derived for the scheme "observe until the
N-th event", whereas the search runs to a fixed frontier with a random number
of finds. Under the actual scheme the correct pooled estimator is (M−B)/S, not
(M−1)/S:

    κ̂ = 1.838   95% CI [1.61, 2.13]   p = 0.64 against e^γ

Every number and figure in the paper is reproduced by a single run of
`paper/analysis.py` with a fixed generator seed; for details see
[`paper/README.en.md`](paper/README.en.md).

## Repository layout

| Path | What it is |
|---|---|
| `src/`, `native/`, `benches/`, `build.rs` | the searcher: sieve, trial factoring, P−1, PRP |
| `config/default.toml` | pipeline parameters, with a rationale for every value |
| `paper/paper_ru.tex`, `paper/paper_ru.pdf` | the paper, Russian edition |
| `paper/paper_en.tex`, `paper/paper_en.pdf` | the paper, English edition |
| `paper/analysis.py` | the whole analysis: numbers, simulations, figures → `results.txt`, `figs/` |
| `paper/data/` | OEIS b-files (OEIS license, see below) |
| `paper/verify/` | independent exhaustive recomputation up to n < 10⁴ — **the appendix of the paper rests on it** |
| `paper/verify_32k/` | a later run carried to k_max = 32000 |
| `paper/METHOD.en.md`, `paper/FINDINGS.en.md` | working notes on the method and the findings |

> **About the two verification directories.** The canonical one is `verify/`:
> it is homogeneous (k_max = 10000 for all twenty bases) and yields exactly the
> numbers quoted in the appendix — 153 confirmed terms, 128 of them in the main
> set b ≤ 20. It is the one `compare_verify.py` and `check_numbers.py` read.
>
> `verify_32k/` is a later run carried to k_max = 32000, but not uniformly:
> different bases stop at 10000, 20000 or 32000. It is broader (169 terms) and
> therefore useful as additional evidence, but it does not support a claim of
> the form "all indices below X have been checked", and it does not match the
> numbers in the paper.

---

## 1. Mathematical background

### 1.1 Why only prime k

If $d \mid k$ then $R_d(b) \mid R_k(b)$. So for composite k the repunit is
composite for certain, and only prime k need be enumerated.

### 1.2 The form of the divisors

Let k be prime and q a prime divisor of $R_k(b)$. Then $b^k \equiv 1 \pmod q$,
that is $\mathrm{ord}_q(b) \mid k$; and k is prime, so the order is either 1
or k.

* **$\mathrm{ord}_q(b) = k$.** By Fermat's little theorem $k \mid q-1$; q is
  odd, hence $2k \mid q-1$ and
  $$q = 2mk + 1.$$
  These are exactly the q the GPU kernel enumerates — instead of every number
  up to the bound, only every $2k$-th one is tested.

* **$\mathrm{ord}_q(b) = 1$ — the edge case.** Then $q \mid b-1$, and the test
  `powmod(b, k, q) == 1` succeeds **always**, that is, it reports a spurious
  divisor. In fact
  $$R_k(b) = 1 + b + \dots + b^{k-1} \equiv k \pmod q,$$
  so $q \mid R_k(b) \iff q \mid k$, which for prime k means precisely
  $k = q$.

* **$q \mid b$.** Then $R_k(b) \equiv 1 \pmod q$ — q is never a divisor.

* **The divisor must be proper: $q < R_k(b)$.** Otherwise a repunit prime
  "sieves itself out": $R_2(10) = 11$, and the candidate
  $q = 2\cdot1\cdot5+1 = 11$ honestly divides it exactly. The check is needed
  only for tiny k — already at $(k-1)\log_2 b > 128$ the repunit exceeds any
  128-bit q.

All three branches are implemented explicitly both in the GPU kernel
([native/cuda/tf_kernel.cu](native/cuda/tf_kernel.cu)) and in the CPU sieve
([src/sieve/small.rs](src/sieve/small.rs)), and both are covered by tests.

### 1.3 The primality test

$R_k(b)$ has no $N \pm 1$ form with known factorization, so a deterministic
proof (N−1/N+1, ECPP) is outside the scope of this tool. We run a strong Fermat
test to base 2 plus Miller–Rabin rounds — that is, we produce **PRP
candidates**, as PFGW/LLR do. Bases are drawn only from the range $[2, N-2]$:
for $a \equiv 0, \pm 1 \pmod N$ a round carries no information, and $a = N$
(say $N = 3$, $a = 3$) would declare a prime composite — which is how the small
repunits $R_2(2) = 3$ and $R_2(10) = 11$ used to be lost. The final proof is a
separate tool's job (Primo/ECPP).

### 1.4 Arithmetic modulo b^k − 1

$R_k(b)$ divides $b^k - 1$, hence

$$x \equiv 3^{E} \pmod{b^k-1} \quad\Longrightarrow\quad x \bmod R_k = 3^{E} \bmod R_k.$$

This lets us work modulo a number that is log₂(b−1) bits longer but has a
**special form**, 1·b^k + (−1). For that form GWNUM does without Barrett
reduction, whereas a general modulus requires `gwsetup_general_mod`. The
difference has been measured (b = 10):

| k | bits | general mod | b^k − 1 | speedup |
|---|---|---|---|---|
| 5 003 | 16 617 | 0.008 ms/iter | 0.003 | 2.81× |
| 10 007 | 33 240 | 0.015 | 0.005 | 2.73× |
| 20 011 | 66 472 | 0.042 | 0.015 | 2.79× |

On the live number R₄₉₀₈₁ (163 041 bits) the test sped up from 16.6 to 4.3
seconds. If the form is not supported (too large a base, or an exotic k) the
code falls back to `gwsetup_general_mod`: correctness before speed.

**Multiplication by the base goes only through `GWMUL_MULBYCONST`.** A separate
`gwsmallmul` call silently corrupts the result on large numbers: a 12-iteration
cross-check against GMP agreed at 66 476 bits but disagreed at 163 044 and
332 203 bits (roundoff jumped to exactly 0.5), even though plain squarings at
the same sizes were computed correctly. The sanctioned mechanism — the constant
is set via `gwsetmulbyconst` and applied inside the squaring itself — agrees
with GMP up to 664 396 bits and saves one operation.

### 1.5 Error control in the GWNUM path

GWNUM computes a Fermat PRP test to base 3 on IBDWT, which is several times
faster than GMP, but the FFT works in floating point, so control is needed. The
scheme is:

* **rounding error** (`gw_get_maxerr`) is checked every 128 iterations; if it
  exceeds 0.40 the computation restarts with a larger FFT (up to 4 attempts);
* **every positive verdict** is recomputed by an independent GMP path.
  Statistically this is free — PRP candidates number a handful per million
  tests — and it rules out a false "PRP" caused by an FFT failure completely;
* the converse case, a false "composite", that is a **lost find**, can be
  caught only by recomputation, and there are two mechanisms for it:
  `double_check_ratio` recomputes a fraction of the "composite" verdicts with a
  second backend during the search itself, and `--verify` re-checks a finished
  worklog (section 4.1).

**There is no Gerbicz–Li check here, and that is deliberate.** It applies when
the exponent yields a long chain of squarings; the author of GWNUM says so
directly (Prime95, `commonb.c`): "We can do Gerbicz error checking if b=2 and
there are a long string of squarings". Here E = N−1 is an arbitrary bit string,
and the chain has the form x ← x²·3^bit. For it the Gerbicz invariant becomes
d_{j+1} = d_j^(2^L)·3^(S_j), where S_j is the sum of L-bit chunks of E, and
checking it costs ~2L operations per L iterations, that is, more than 100%
overhead. No cheap variant exists for an arbitrary exponent.

---

## 2. Architecture

```
src/
├── main.rs         CLI, thread pool, NUMA pinning
├── config.rs       TOML config + validation
├── pipeline.rs     the pipeline and its channels
├── tuner.rs        adaptive TF depth and the P−1 decision
├── affinity.rs     CPU affinity, NUMA, huge pages
├── worklog.rs      JSONL worklog, resume after a restart
├── report.rs       final JSON
├── sieve/
│   ├── kbase.rs    segmented sieve of Eratosthenes (generator of k)
│   └── small.rs    sieve by small divisors q (enumeration "from the q side")
└── ffi/            safe RAII wrappers over the native layer
native/
├── tests/          standalone checks of the GPU arithmetic and the GWNUM path
├── include/        the shared ABI (rh_common.h) + per-layer headers
├── cuda/
│   ├── tf_kernel.cu  kernels of three widths (Mont64/96/128)
│   ├── tf_host.cu    streams, CUDA Graphs, persistent buffers
│   └── rh_mont.cuh   Montgomery arithmetic (templates)
└── prp/
    ├── prp_dispatch.c  backend selection by size of the number
    ├── prp_gmp.c       GMP backend (Miller–Rabin)
    ├── prp_gwnum.c     GWNUM/IBDWT + Gerbicz–Li
    ├── pm1_ecm.c       P−1 with seed 2^(2k)
    ├── rh_arena.c      arena of reusable mpz_t
    └── rh_alloc.c      thread-local pool for GMP allocations
```

Three decisions determine the performance:

1. **Zero allocations in the hot loop.** `mpz_init` is called once per thread
   (the arena); temporary GMP buffers go through a thread-local bump allocator
   on huge pages.
2. **Batching over k on the GPU.** One launch processes `tf_k_batch` exponents:
   for large k the range of m is too narrow to occupy the device with a single
   k. The index layout is such that a whole warp works on one k — there is no
   divergence in the powmod loop.
3. **Every divisor found is verified on the CPU.** A divisor from the GPU is
   not taken on trust: it is re-checked through GMP. A failed check is logged
   as an error — a signal of a kernel bug or an unstable card.

---

## 3. Building

The native layer targets **Linux** (`mmap`, `clock_gettime`,
`sched_setaffinity`); other platforms will need `rh_alloc.c` and `affinity.rs`
edited.

Required: Rust ≥ 1.75, a C compiler, **GMP** (dev package).
Optional: CUDA Toolkit ≥ 11.0, GWNUM (Prime95 SDK), libecm (GMP-ECM).

```bash
# Full build (CUDA + GMP)
cargo build --release

# CPU only (no GPU stage)
cargo build --release --no-default-features

# With GWNUM and P−1
GWNUM_DIR=/opt/gwnum ECM_DIR=/usr/local cargo build --release
```

Build-time environment variables:

| Variable | Meaning |
|---|---|
| `CUDA_PATH` / `CUDA_HOME` | root of the CUDA Toolkit (otherwise `nvcc` is looked up in PATH) |
| `RH_CUDA_ARCHS` | comma-separated SM list, default `70,75,80,86,89,90` + PTX |
| `GMP_DIR` | prefix of your own GMP build (e.g. one with `--enable-fat`) |
| `RH_GMP_STATIC=1` | link GMP statically |
| `GWNUM_DIR` | GWNUM root; without it PRP goes through GMP |
| `ECM_DIR` / `RH_ECM_SYSTEM=1` | libecm; without them the P−1 stage is disabled |
| `RH_MAXREG` | `-maxrregcount` for nvcc (tuned by occupancy) |
| `RH_PORTABLE=1` | no `-march=native` (for distribution packages) |

Cargo features: `cuda`, `gwnum`, `pm1` (all on by default), `numa`. A missing
library does not break the build — the corresponding stage is simply switched
off, and `--devices` will show which backends are available.

### 3.1 The deployed environment (WSL2 on this machine)

A Windows host does not build the project: the native layer is POSIX-only. The
working environment is set up in WSL2 Ubuntu 26.04:

| What | Where / version |
|---|---|
| Rust | 1.98.0, system-wide in `/opt/rust` (PATH is set by `/etc/profile.d/rust.sh`) |
| gcc / ar / pkg-config | from `build-essential` |
| lld | required — `[target.x86_64-unknown-linux-gnu]` in `.cargo/config.toml` asks for it |
| GMP | `libgmp-dev` 6.3.0 |
| libecm | `libecm-dev` 7.0.6 — the P−1 stage is available |
| CUDA | `nvidia-cuda-toolkit` 12.4.131, `nvcc` in `/usr/bin` |
| GPU | GeForce GTX 1650, CC 7.5 (sm_75), 14 SM |

`.cargo/config.toml` sets `RH_CUDA_ARCHS=75` and `RH_ECM_SYSTEM=1` for this
machine. The `[env]` section does not override variables already set, so to
build for other hardware it is enough to export your own list:
`RH_CUDA_ARCHS=80,90 cargo build --release`.

```bash
wsl -d Ubuntu
cd /mnt/c/Users/<you>/Downloads/repunit-hunt

# keep target on ext4: on /mnt/c (drvfs) the build is several times slower
export CARGO_TARGET_DIR=~/rh-target

cargo build --release --no-default-features --features "cuda pm1"
cargo test  --release --no-default-features
$CARGO_TARGET_DIR/release/repunit-hunt --devices
```

**GWNUM is built and linked in** (Prime95 SDK 30.19,
`/opt/gwnum-src/extracted`). Distributions do not ship it, so the procedure is:

```bash
curl -O https://www.mersenne.org/download/software/v30/30.19/p95v3019b20.source.zip
unzip p95v3019b20.source.zip -d /opt/gwnum-src/extracted
cd /opt/gwnum-src/extracted/gwnum && make -f make64     # do not skip this!
```

The last step is critical: the `linux64/` directory in the archive contains
only the prebuilt assembly FFT modules, without the C part (`gwinit2`,
`allocgiant`, `gwtogiant`). The complete library appears in `gwnum/gwnum.a`
only after `make`, and that is what build.rs looks for. The GWNUM assembly is
built without `-fPIC`, so with GWNUM linked in the binary is non-PIE — build.rs
adds `-no-pie` automatically.

---

## 4. Running

```bash
# Information about devices and backends
./target/release/repunit-hunt --devices

# A search driven by a config
./target/release/repunit-hunt --config config/default.toml

# A quick run without the GPU
./target/release/repunit-hunt --base 10 --kmin 3 --kmax 20000 --no-gpu
```

| Flag | Meaning |
|---|---|
| `--config <path>` | TOML config (the flags below take priority) |
| `--base <b>` | the repunit base |
| `--kmin` / `--kmax` | exponent range, half-open interval `[kmin, kmax)` |
| `--threads <n>` | worker threads, `0` = one per core |
| `--no-gpu`, `--no-pm1` | disable the corresponding stage |
| `--devices` | show GPUs and available backends, then exit |
| `--double-check <ratio>` | recompute a fraction of the "composite" verdicts with a second backend on the fly |
| `--verify` | re-check the worklog and exit (see 4.1) |
| `--verify-ratio <ratio>` | what fraction of the "composite" verdicts to recompute under `--verify` (default 0.02) |

Logging goes through `env_logger`:
`RUST_LOG=debug ./target/release/repunit-hunt ...`

### 4.1 Re-checking the worklog (double-check)

```bash
# sampled (2% of the composites) — fast
./target/release/repunit-hunt --config config/default.toml --verify

# exhaustive, recomputing every composite
./target/release/repunit-hunt --config config/default.toml --verify --verify-ratio 1.0
```

What is checked for each record in `worklog.jsonl`:

| Record | Check |
|---|---|
| `factored` | does the recorded q divide R_k(b), and is it proper (q < N) — cheap, all of them are checked |
| `PRP` | the find is recomputed in exact GMP arithmetic — always |
| `composite` | the verdict is recomputed with another backend: this is the only way a lost find is caught |

Exit code 1 if even one disagreement is found — the mode suits cron and CI. The
sample is deterministic (a hash of base and k), so the same `--verify-ratio`
always checks the same numbers.

### Resuming work

Every closed exponent is written to `worklog.jsonl` (append-only JSONL; PRP
finds are flushed immediately). On the next run such k are skipped, so an
interrupted search continues where it stopped. The result is mirrored in
`results.json` (atomic replacement via a temporary file).

---

## 5. Configuration (config/default.toml)

The key parameters:

* `bitsieve_q_limit` — the bound of the CPU sieve. It removes most candidates
  cheaply; raising it makes sense as long as the sieve is built in seconds.
* `tf_q_min` / `tf_q_hard_max_bits` — the trial factoring window. The lower
  bound is raised automatically to `bitsieve_q_limit`: below that the sieve has
  already done the work.
* `tf_adaptive` — the TF depth is chosen by the tuner: widening the range of q
  pays off as long as the expected saving on PRP (≈ `t_prp / ln q`) exceeds the
  cost of enumerating the next decade.
* `tf_k_batch` — how many k go to the GPU in one batch (≤ 8192, that is `MAX_K`
  in `tf_host.cu`).
* `mr_rounds` — extra Miller–Rabin bases on top of base 2.
* `prp_backend` — `auto` switches to GWNUM at `bits ≥ gwnum_threshold_bits`.
* `pin_threads`, `gmp_pool_mb` — NUMA locality and pool size per thread.

---

## 6. Checking a result

`results.json` contains **PRPs**, not proven primes. What to do after a find:

1. Re-check with an independent tool: `pfgw64 -tc -q"(10^k-1)/9"` or LLR.
2. Prove primality: Primo (ECPP) for numbers up to ~50 000 digits.
3. Submit to the [Prime Pages](https://t5k.org/) or the relevant projects.

GPU false positives are logged at level `error` and summarized at the end of a
run — a non-zero counter means the trial factoring results cannot be trusted (a
kernel bug, overclocking, or degrading card memory).

---

## 6.1 Performance of the GPU stage

Everything measured on a GTX 1650 (14 SM, Turing) under WSL2,
`native/tests/bench_gpu.cu`, a batch of 256 exponents.

**Launch overhead.** Under WSL the GPU is paravirtualized and every driver call
crosses the VM boundary, so launching through a CUDA Graph (one submit instead
of five calls) is markedly cheaper:

| variant | ms/launch |
|---|---|
| ordinary async calls, no graph | 51.5 |
| graph rebuilt every time | 37.5 |
| **graph built once** | **34.4** |

The original code captured the graph via stream capture, and its signature
included `m_start` — while the pipeline increments m on every launch, so the
graph was rebuilt every time. The graph is now built by hand, and only the
kernel node's parameters are updated between launches
(`cudaGraphExecKernelNodeSetParams`). This is verified not by timing (clock
scatter under WSL reaches 20%) but by a direct counter:
`rh_gpu_graph_builds()` reports **1 rebuild per 208 launches**.

**The small-prime filter turned out to be the main brake.** Turing has no
hardware integer division, and 54 `q % p` operations per candidate cost more
than the powmod they save. Division was replaced by multiplication by the
inverse modulo 2⁶⁴:

$$p \mid q \iff (q \cdot p^{-1} \bmod 2^{64}) \le \lfloor (2^{64}-1)/p \rfloor$$

The constants are generated by `native/tests/gen_small_primes.py` (which also
self-checks them).

| primes in the filter | 54 | 32 | 16 | **12** | 8 |
|---|---|---|---|---|---|
| division, ms/launch (k~10⁵) | 36.1 | 21.3 | — | — | 13.1 |
| multiplication, ms/launch (k~10⁵) | 12.5 | 10.9 | 9.9 | **9.4** | 9.0 |
| multiplication, ms/launch (k~10⁶) | 13.8 | 12.1 | 11.0 | **10.5** | 10.2 |

The curve is flat over 8..16; 12 is the default. In total the GPU stage was
sped up **from 36.1 to 10.5 ms per launch — by a factor of 3.4.**

**The hit buffer.** The original 16 records per launch proved too few: at the
start of the m range (where q is only slightly above the CPU sieve bound) the
density of divisors is high, and a batch of hundreds of exponents produces
dozens of hits. Over a long run this caused 8 overflows — and every lost
divisor means that instead of being removed instantly a candidate goes on to a
full PRP test at a hundred thousand bits. The capacity was raised to 256
records (6 KiB), and the `lost` field of `rh_tf_result_t` now reports the exact
number of hits that did not fit, rather than merely the fact of an overflow.
The effect is visible on the reference run (b=10, k<5000): the GPU removes 43
candidates instead of 31, and 275 instead of 287 go on to PRP — that is, the
divisors that were being found and then lost have come back.

---

## 7. What has been verified

| Check | Result |
|---|---|
| `cargo test` (sieve, edge cases, segment boundaries) | 8/8 |
| b=10, k < 3000, CPU | `[2, 19, 23, 317, 1031]` — matches the known list of repunit primes |
| b=10, k < 3000, with GPU | the same list; the GPU removed 11 candidates, 0 false positives |
| b=2, k < 700 | `[2,3,5,7,13,17,19,31,61,89,107,127,521,607]` — exactly the Mersenne exponents |
| Every record in `worklog.jsonl` | the divisor really does divide R_k; no prime was sieved out |
| Kernel branching against honest division | 50 388 combinations (b,k,q), 0 disagreements |
| P−1 (forced mode) | removed 66 candidates, all 66 divisors verified — genuine |
| Montgomery arithmetic on the GPU | 134 vectors against a reference: Mont64 43/43, Mont96 50/50, Mont128 41/41 |
| Kernels `rh_tf_k64/k96/k128` | each finds a genuine repunit divisor in a narrow window in m |
| Two threads on one card (`gpu_devices=[0,0]`) | the result does not change, no false positives |
| Compilation of the GWNUM backend | against the real Prime95 SDK headers — clean |
| **GWNUM on live numbers** | verdicts agreed with GMP at 6 sizes (1 050…69 761 bits) and on all 353 candidates in the range k=3000…6000 |
| **R_49081(10), 163 041 bits** | GWNUM found a PRP in 11.9 s, GMP verification confirmed it |
| GWNUM speedup over GMP | 2.9× (6.6k bits) → 8.1× (70k bits); over the range 3000–6000, 4.4× |
| Fallback when GWNUM is absent | `prp_backend="gwnum"` without the library computes on GMP with a warning, no candidate is lost |
| `cargo clippy` | no warnings |
| On-the-fly double-check | 179 "composites" recomputed with a second backend, 0 disagreements |
| `--verify` on a clean worklog | 119 divisors + 5 PRPs + 179 composites, 0 disagreements, exit code 0 |
| **`--verify` on a tampered worklog** | all three forgeries caught: a fake divisor, a false PRP and a lost find; exit code 1 |
| **Long run, k = 5000…60000** | the net effect of the optimizations: 1:18:48 → **14:28** (5.4×), the find R_49081(10) still there, 0 errors |
| Worklog re-check | 3376 divisors, the find confirmed, 50 recomputed composites — no disagreements |
| **Twelve bases, b = 2…13, k < 600** | the PRP lists agreed with an independent reference for every b, including the degenerate cases b=4, 8, 9 |

To reproduce the native checks: `bash native/tests/run_tests.sh` (the vectors
are regenerated by
`python native/tests/gen_vectors.py native/tests/mont_vectors.h`).

**What the checks found, and what was fixed**

* **Mont96 gave a wrong result on every input**: the 96-bit REDC shifted by 128
  bits instead of 96 (the low 32 bits were lost) and "teleported" the carry
  across a limb. In Mont128 the same chain happens to be correct, because there
  the low limbs are zeroed by construction. Both widths now carry the chain in
  hardware (`add.cc → addc.cc → addc.cc → addc`).
* **A divisor could equal the number itself**: R_2(10) = 11 was sieved out by
  the "divisor" 11. A proper divisor q < R_k(b) is now required — in the sieve,
  in the kernel and in `rh_prp_verify_factor`.
* **Miller–Rabin bases outside [2, N−2]**: for N = 3 a round with base 3
  declared a prime composite, losing R_2(2) = 3 and R_2(10) = 11.
* **An exponent could vanish from the results on ANY PRP error.** On `Err`,
  `pipeline.rs` only wrote a line to the log and **created no worklog record**:
  the candidate is neither "composite" nor "PRP" but simply absent — and the
  disappearance can be noticed only by comparison with an external list. Found
  by an extended recomputation: **b = 2, k = 1279 (M_1279, a known Mersenne
  prime) went missing** from a `--kmax 20000` run, leaving
  `ERROR ... PRP k=1279: internal error`. The same k computed separately passes
  cleanly on every backend — the failure showed up only under multi-threaded
  load. Now, on an error, the candidate is finished on GMP (which depends
  neither on the FFT nor on GWNUM state), and if even that fails a
  `"status":"failed"` record is written so that the exponent stays visible.
* **The size estimate was doubled for b = 2.** `prp_dispatch.c` had
  `est_bits = (k-1)·(floor(log2 b) + 1)`; for b = 2 that is `(k-1)·2` instead
  of `(k-1)·1`. Consequence: the measured GWNUM switchover threshold (2500
  bits) was effectively applied to b = 2 as 1250 bits — precisely the zone
  where, by the project's own measurements, GWNUM is slower than GMP (at 1017
  bits, ten times slower), and where setting it up on too small a number is
  what caused the failure. Replaced with an honest `log2`. This is the root
  cause of the disappearance of M_1279.
* **A lost find in the GWNUM path: roundoff was not checked after the loop.**
  The rounding error was monitored once every 128 iterations, while the final
  value of `gw_get_maxerr()` was only recorded in the statistics and **not
  compared against the threshold**. A spike on one of the last 127 iterations
  went unnoticed, and a "composite" verdict was returned from a knowingly
  corrupted residue — that is, a find was lost silently. Found by an exhaustive
  recomputation of the OEIS sequences (`paper/verify_sequences.sh`):
  **R_3181(23) was declared composite although it is a PRP** — a term of
  A204940. Diagnosis: `maxerr = 0.5000` (the maximum possible), the first
  breach of the threshold on iteration 14383 of 14384, that is, after the last
  periodic check at 14336. With `margin = 1` the FFT grows from 768 to 1024,
  `maxerr` drops to 0.0015, and the verdict is right. Frequency: 1 case per 96
  pairs (b, k) in a scan over `b ∈ {3..26}`, `k ∈ {1009..10007}`. A final
  threshold check was added, returning `RH_ERR_FFT_ERROR`, which triggers the
  normal restart with a larger FFT.
* **A candidate vanished if GWNUM could not meet the roundoff budget in 4
  attempts.** `prp_dispatch.c` returned an error, and on an error
  `pipeline.rs` only writes a log line and **creates no worklog record** — the
  exponent disappeared from the results without a trace. Such a case is now
  finished on GMP: slowly, but correctly.
* **A sham Gerbicz check** in the GWNUM path compared the wrong identity and,
  on top of that, never ran (its condition fired after L² ≈ 4·10⁶ iterations —
  two orders of magnitude beyond practical sizes). Replaced by the scheme of
  section 1.4.
* `flatten()` on the iterator of worklog lines could loop forever on a
  repeating read error — replaced with `map_while`.
* `rh_gmp_pool_reset()` was removed from the Rust wrapper: the arena holds
  buffers from the pool, and resetting the top would corrupt live numbers.

**How the re-checking mechanism was itself checked.** It is not enough to show
that a clean worklog yields no disagreements — one has to be sure the mechanism
is capable of seeing them at all. So three forgeries were planted in the
worklog on purpose: the divisor q=13 for R_7(10) (does not divide it), R_19(10)
marked composite (it is in fact prime), and R_5(10) marked as a PRP (it is in
fact 41·271). All three were caught, each with its own diagnosis, exit code 1.

### The net effect of the optimizations: three runs over one range

k = 5000…60000, numbers from 16 000 to 200 000 bits, the same machine:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| **time** | 1:18:48 | 1:03:30 | **14:28** |
| removed by the sieve | 1915 | 2298 | 2298 |
| removed by the GPU | 165 | 278 | **1078** |
| PRP tests | 3308 | 2812 | **2012** |
| memory | 715 MiB | 744 MiB | 1241 MiB |
| R_49081 | ✓ | ✓ | ✓ (8.3 s) |

Between runs 1 and 2: a faster GPU kernel, a larger hit buffer, tuner
calibration. Between 2 and 3: arithmetic modulo b^k−1, a GWNUM threshold of
2500, a corrected PRP cost model, all 16 logical cores.

Curiously, the prediction that "a cheaper PRP will reduce the benefit of trial
factoring" was not borne out: the GPU began removing four times as many. Both
sides of the economics changed at once, and the correct model let the tuner use
the filter where the old one — which underestimated the cost of PRP twofold —
did not.

### What the long run showed

An hour and a quarter of continuous work over the range k = 5000…60000
(numbers from 16 000 to 200 000 bits), all stages enabled:

| stage | candidates removed |
|---|---|
| CPU sieve (q ≤ 2²⁴) | 1915 of 5388 (36%) |
| GPU trial factoring | 165 |
| P−1 | 0 |
| PRP | 3308 tested, 1 found |

* **Memory is stable**: 683 → 693 MiB over 79 minutes, peak 715 MiB. No leaks.
* **The tuner calibrates correctly**: the GPU rate it measured (0.30 G/s)
  matched an independent benchmark (340 million candidates/s), and its PRP cost
  estimate (20 s at 163k bits) matched the actual 16.7 s.
* **P−1 never ran, and that is the right decision.** At B1 = 10⁵ the stage
  costs ~3.7·10⁵ modular multiplications, whereas a whole PRP test of a
  163 041-bit number costs 1.6·10⁵. That is, P−1 here is dearer than the test
  it is supposed to save, and the adaptive mode honestly disables it. The stage
  pays off only when B1 is far smaller than the size of the number — at
  millions of bits, not hundreds of thousands.

### Tuning the stages from measurements

Three parameters were set not by eye but from the measurements of this run.

**GPU calibration at startup.** The tuner had a hard-wired estimate of 8
billion candidates/s — "as on a top-end card". The reality of a GTX 1650 is
0.30 billion, a miss by a factor of 27, and until statistics accumulated the TF
depth was chosen at random. Now the GPU thread performs one probe launch
(~10 ms) and reports the real rate to the tuner; the starting constants were
also lowered — an error towards shallow TF is safer than one towards uselessly
deep TF.

**The CPU sieve bound scales with the range.** The sieve is the cheapest stage,
but its cost does not depend on k while its return does: the dearer the PRP
test, the more it pays to remove a candidate in advance. Measured
(k = 5000…15000):

| q bound | 2²⁴ | 2²⁶ | 2²⁷ | 2²⁸ |
|---|---|---|---|---|
| time | 0.2 s | 1.0 s | 2.25 s | 5.4 s |
| removed | 30.1% | 32.3% | 33.1% | 33.8% |
| memory | 179 MiB | 256 MiB | 275 MiB | 519 MiB |

The ceiling was raised to 2²⁷ (beyond that the return falls off while memory
doubles), but the effective bound is computed as `k_max · 2000` and only then
capped by the config: on a short run a sieve to 2²⁷ would take longer to build
than the whole search takes to run.

**The PRP cost model was wrong — and that cost more than anything else.** The
tuner had `t ∝ bits^1.15`, that is, an almost linear dependence. A fit over
2756 real measurements from the worklog (numbers of 20 000…199 310 bits) gives
an exponent of **2.24**:

| | model in the code | from measurements |
|---|---|---|
| formula | `bits^1.15` | `bits^2.24` |
| mean error | 6.11 s | 1.03 s |
| prediction for 163 041 bits | 9.7 s | 17.8 s (actual 16.6) |
| prediction for 200 000 bits | 12.3 s | 28.2 s |

The exponent 2.24 is physically justified: $R_k(b)$ has no special form, so
GWNUM works through `gwsetup_general_mod` — IBDWT multiplication plus Barrett
reduction. An iteration costs O(n log n), there are n iterations, hence
~n²·log n.

The consequence of the error was direct: the tuner halved the estimated benefit
of trial factoring and hardly used the GPU. Measured on one subrange
(k = 9000…18000):

| model | removed by the sieve | removed by the GPU | went to PRP | GPU share |
|---|---|---|---|---|
| `bits^1.15` | 454 | **3** | 490 | 0.6% |
| `bits^2.24` | 413 | **172** | 362 | **32.2%** |

**The GWNUM threshold was five times too high.** It was set to 10 000 bits,
whereas the measured crossover with GMP is around 2000:

| bits | 1017 | 1994 | 3010 | 5017 | 9966 |
|---|---|---|---|---|---|
| GWNUM speedup | 0.10× | 1.06× | 1.95× | 3.21× | 4.31× |

The whole range 2000…10 000 bits was computed on GMP, losing a factor of two to
four there. The threshold was lowered to 2500.

**P−1 is off by default.** The old heuristic pulled a success probability of
"≈4%" out of thin air. An honest count in modular multiplications: stage 1
costs ≈ 2.9·B1, a whole PRP test ≈ `bits`. At B1 = 10⁵ and a 163 041-bit number
that is 2.9·10⁵ against 1.6·10⁵ — the stage is dearer than what it saves. The
probability is now estimated by the Dickman function ρ(ln m / ln B1) from a
table, rather than invented. P−1 begins to pay off at millions of bits; that is
where to enable it.

### Thread count and virtualization

Measured on an i7-10700 (8 physical cores, 16 logical, AVX2, 16 MiB L3), range
k = 9000…18000:

| threads | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|
| time | 41.1 s | 22.7 s | 17.0 s | **15.8 s** | 15.8 s | 15.8 s |
| µs/bit | 6.5 | 7.2 | 7.8 | 9.6 | — | — |

The optimum is exactly the number of LOGICAL cores; beyond that it is a
plateau. Each individual test is slower there (9.6 against 7.2 µs/bit), but
total throughput is higher: hyper-threading hides memory latency well on an FFT
workload. The default `threads = 0` means precisely "one per logical core" —
there is no reason to change it by hand.

**On WSL2.** Processor instructions execute natively there, so GWNUM runs at
full speed, and PRP is nearly all of the running time. Virtualization hits only
the CUDA driver calls, and that has been measured: without CUDA Graphs a launch
cost 51.5 ms against 34.4 ms with them, so the overhead was a third. But after
PRP was sped up threefold the GPU stage accounts for about one percent of the
total time (15.68 s against 15.80 s), so leaving virtualization altogether
would gain single-digit percent — at the price of porting `rh_alloc.c`
(mmap/madvise), `affinity.rs` (sched_setaffinity) and `clock_gettime` to MSVC.

It makes far more sense to look at the processor: the i7-10700 has no AVX-512,
and GWNUM is noticeably faster with it.

### The reachable depth of trial factoring

The multiplier m is passed to the kernel as a `uint64_t`, so enumeration can
reach only

$$q = 2mk + 1 < 2^{65}\,k.$$

For k ≈ 10³ that is ~2⁷⁵, for k ≈ 10⁶, ~2⁸⁵. The practical conclusion: the
128-bit branch (q ≥ 2⁹⁵) engages only for k > 2³⁰, so an ordinary search uses
Mont64 and Mont96. If the tuner asks for a depth beyond what is reachable, the
range in m is silently clamped — see `clamp` in `gpu_worker`.

---

## License

The repository is licensed in parts.

**The searcher's source code** (`src/`, `native/`, `benches/`, `build.rs`,
`config/`) — at your option [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), as declared in `Cargo.toml`.

**The paper and the analysis** (`paper/`: text, figures, scripts, derived data)
— [CC BY 4.0](paper/LICENSE).

**Exception:** `paper/data/*.txt` are b-files downloaded from
[OEIS](https://oeis.org). They are included for reproducibility without network
access and remain under the OEIS terms (CC BY-NC-SA 4.0), not under this
repository's licenses. To download them again from the original source:
`bash paper/fetch_data.sh`.

The Mersenne prime exponents come from the open PrimeNet report of the
[GIMPS](https://www.mersenne.org/) project.

## Citation

    Dokuchaev T. The observation scheme in the statistics of generalized
    repunit primes: calibrating the pooled test of the Lenstra–Pomerance–
    Wagstaff constant. 2026. ORCID: 0009-0006-0510-5225
