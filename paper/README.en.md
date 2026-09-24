*[Русский](README.md) · English*

# Materials for the paper on the Lenstra–Pomerance–Wagstaff constant

Reproducible analysis for the paper "The observation scheme in the statistics of
generalized repunit primes: an exact conditional test of the
Lenstra–Pomerance–Wagstaff constant". The current version of the paper is 3; the
changes since version 1 are listed in [`CHANGES_v3.md`](CHANGES_v3.md) (in Russian).

The paper exists in two language editions with the same content and the same
numbers:

* `paper_ru.tex` — Russian, laid out per GOST R 7.0.7-2021;
* `paper_en.tex` — English, international format.

Every decimal number in both editions is checked against the script output
(`check_all_numbers_v3.py`); GOST is checked for the Russian edition.

## Quick start

```bash
# 1. Data (already in data/; this step is only needed to refresh it)
bash fetch_data.sh          # OEIS b-files, main set b <= 20
bash fetch_extra.sh         # b = 21..26, robustness of the selection bound

# 2. Every number in the paper + the figures
python3 analysis_v2.py      # [S0]-[S12] -> results_v2.txt, figs_v2/  (about a minute)
bash make_results_v3.sh     # [S13]-[S25] -> results_v3.txt  (about 5 minutes;
                            #   --full recomputes the c_b calibration, 10 more minutes)

# 3. The paper
bash build_paper.sh            # -> paper_ru.pdf and paper_en.pdf
bash build_paper.sh paper_en   # English only
```

## Full validation

```bash
bash validate.sh          # runs 1-4
bash validate.sh --full   # plus recomputing the c_b calibration and the sequences n < 10^4
```

| Run | What it checks |
|---|---|
| 1. Clean room | everything derived is rebuilt from `data/` and `paper_*.tex` |
| 2. Cross-checks | every number in the paper, key numbers, GOST, data, self-overlap, typography, PDF |
| 3. Seed robustness | conclusions under three generator seeds |
| 4. Live sources | OEIS b-files and the GIMPS frontier today (known drift is printed but does not fail the run) |
| 5. Independent recomputation | `repunit-hunt` recomputes every n < 10⁴ (`--full`) |

Exit code 0 only if everything passes. Individual checks:

```bash
python3 check_all_numbers_v3.py   # EVERY decimal number of both editions against the script output
python3 check_numbers_v3.py       # key numbers of both editions
python3 check_gost.py             # 32 checks of paper_ru.tex against GOST R 7.0.7-2021
python3 check_stability_v3.py     # seed robustness
python3 check_overlap.py          # self-overlap with earlier drafts
python3 compare_verify.py         # recomputation n < 10^4 against OEIS
python3 compare_verify_ext.py     # extended recomputation against OEIS
```

Dependencies of the analysis: `numpy`, `scipy`, `matplotlib`. Building the paper:
`texlive-latex-recommended`, `texlive-lang-cyrillic`, `texlive-latex-extra` (on
Windows, TinyTeX with `babel-russian`, `cyrillic`, `lh`, `cm-super`,
`hyphen-russian` is enough).

## What is where

| File | Purpose |
|---|---|
| `paper_ru.tex`, `paper_ru.pdf` | the paper, version 3, Russian (GOST R 7.0.7-2021) |
| `paper_en.tex`, `paper_en.pdf` | the paper, version 3, English |
| `CHANGES_v3.md` | what changed in version 3 and which checks were run |
| `analysis_v2.py` → `results_v2.txt`, `figs_v2/` | exact conditional inference: tags [S0]–[S12], figures |
| `stopping_rules.py` | [S13] stopping rules that violate (A1) |
| `lpw_second_order.py` | [S14]–[S17], [S20]–[S22] second-order term, model exp(c₁/t+c₂/t²), B = 20, §7.3 |
| `frontiers.py` | [S18] documented frontiers: check of (A1), hybrid estimate |
| `cb_empirical.py` → `results_S19.txt` | [S19] calibration of c_b from the divisibility of R_b(p) |
| `cb_theory.py` | [S23] first-principles prediction of c_b and paired comparison |
| `sim_second_order.py` | [S24]–[S25] uncertainty of c_b, simulation check of the method |
| `make_results_v3.sh` → `results_v3.txt` | assembles the output [S13]–[S25] |
| `check_all_numbers_v3.py`, `check_numbers_v3.py` | numbers of the paper against the output, exit code 1 on a disagreement |
| `check_stability_v3.py` | seed robustness |
| `check_gost.py` | GOST R 7.0.7-2021 (argument: file name, default `paper_ru.tex`) |
| `check_overlap.py` | self-overlap with earlier drafts |
| `validate.sh` | the orchestrator of all version-3 checks |
| `data/` | OEIS b-files, 20 sequences (snapshot of 27 Aug. 2026) |
| `verify_sequences.sh`, `compare_verify.py`, `verify/` | exhaustive recomputation n < 10⁴ (153 terms), the basis of the appendix |
| `verify_range.sh`, `verify_plan.py`, `compare_verify_ext.py`, `verify_ext/` | extended recomputation n ≥ 10⁴, see [`VERIFY_EXTENDED.md`](VERIFY_EXTENDED.md) (in Russian) |
| `verify_32k/` | a later run up to k_max = 32000, not used in the paper |

**Version 1 archive** (kept so that the history stays checkable):
`paper_ru_v1.tex`, `paper_en_v1.tex` and their PDFs, `analysis.py` →
`results.txt`, `figs/`, `check_numbers.py`, `check_method.py`,
`check_stability.py`, `METHOD.md`, `FINDINGS.md`, `validate_v1.sh`.
`analysis.py` also serves as the shared data loader for `analysis_v2.py`.

All random number generators are initialized with fixed seeds, so the Monte Carlo
quantities reproduce bit for bit.

## Independent recomputation of the sequences

Cross-checking against OEIS confirms that the listed indices give primes, but it
does not guarantee that the list is **complete**. A missing term would shift
`N_b`, and with it both pooled estimates. So every index `n < 10^4` was
recomputed consecutively with the searcher from the repository root:

```bash
bash ../paper/build_and_verify.sh                 # build repunit-hunt
DC=1.0 bash verify_sequences.sh 10000             # exhaustive recomputation
python3 compare_verify.py                         # comparison against OEIS
```

`DC=1.0` enables a blanket double-check: every negative verdict is recomputed
by a second, independent backend. Without it a lost find (a false "composite"
caused by an FFT failure) stays invisible — that is exactly how the bug in the
GWNUM path was found, described in `FINDINGS.md`, §4.

Current result: **153 terms confirmed, 0 spurious, 0 missing** across all
twenty bases.

The continuation beyond n >= 10^4 (in resumable ranges) is described in
[`VERIFY_EXTENDED.md`](VERIFY_EXTENDED.md) (in Russian).
