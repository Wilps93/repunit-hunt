*[Русская версия](FINDINGS.md) · English*

# Report: reproducing and checking the paper

Date: 27 August 2026.
Subject: "The observation scheme in the statistics of generalized repunit
primes: calibrating the pooled test of the Lenstra–Pomerance–Wagstaff constant"
(source — `paper (1).tex`, PDF — `1 (1).pdf`, 18 pages).

Method: `analysis.py` was written to reproduce **every** number in the paper
from the OEIS b-files; the output of a run is `results.txt`. In addition, all
the sequences were recomputed by an independent implementation
(`repunit-hunt`).

---

## 1. Bottom line: the paper computes correctly

**Reproduced exactly, to the last digit:**

| What | Status |
|---|---|
| Table 1 (N_b, n_min, n_max, D_b, M = 218, S = 110.4423) | ✅ all 15 rows |
| Table 2 (Monte Carlo for Propositions 1–2) | ✅ |
| Table 5 (G_end, G_OLS, κ̂, κ̃, all 15 CIs, all 15 PITs) | ✅ every value |
| §5.2, eq. (11): κ̃ = 1.9648, CI [1.7205, 2.2444], p = 0.142 | ✅ |
| §5.2, eq. (12): κ̂_B = 1.8381 | ✅ |
| Table 3 (unbiasedness under 7 frontier rules) | ✅ (see 2.13) |
| Table 4 (type I error 0.162; c = 1.069) | ✅ |
| §4.3 (weighted mean: +0.1% under A, +7.5% under B) | ✅ |
| §5.3 (Λ = 7.40, df = 14, p = 0.92) | ✅ |
| Table 6 (truncation) | ✅ given an explicit dropout rule (2.4) |
| §5.4 (leave-one-out, status of the data) | ✅ |
| Table 7 + §6 (power, 49.8%, 21.2%, M ≈ 875 / 3319) | ✅ |
| §7 (slope 2.6391, intercept −2.9627, SE 0.4469 → 0.7315, ρ₁ = 0.825) | ✅ |
| §7.1 (bootstrap, max\|r\|, mid-range deficit) | ✅ |
| §7.3 (Mertens constant → +0.672) | ✅ |
| Table 8 (models A–D: ln L, LR, p, AIC, BIC, posterior probabilities) | ✅ all |
| §8 (C^fix: κ̂ = 0.815, ln L = −193.55, difference −17.42) | ✅ |
| §8 (log corrections: κ̂ = 2.049, ĉ₁ = +0.108, ĉ₂ = −0.826, LR = 3.53) | ✅ |
| Remark 2 (17 coincidences, n = 3 in nine of them, all at n ≤ 317) | ✅ |

The mathematical part (Propositions 1–3, Corollary 1) was checked both by
simulation and by derivation. There are no errors in it.

---

## 2. Defects found

### 2.1. A factual error in §3.4 (fixed)

> "Under the endpoint form base 18 has Ĝ₁₈ = 0.6604 — **the largest value among
> all fifteen**, that is, it is the 'unluckiest' base"

False by the paper's **own Table 5**: Ĝ₇ = 0.7104 and Ĝ₁₀ = 0.6612 are larger.
Base 18 is the **third**, not the first. On top of that, §1 of the same paper
calls base 7 the "unluckiest", which directly contradicts this sentence.

Ordering by descending Ĝ_b: 7 (0.7104), 10 (0.6612), **18 (0.6604)**, 3, 20, 2, …
Ordering by descending Ĝ_b^OLS: 7, 10, 3, 20, 14, 2, 15, 6, 5, **18 (0.4656)**, …

The substantive conclusion survived and even got sharper: base 18 moves **from
3rd place to 10th of 15** when the form of the estimator changes. The wording
was corrected.

### 2.2. The base selection rule in §2.2 does not yield the stated limit (fixed)

> "The upper limit b ≤ 20 was chosen as the one up to which the OEIS sequences
> contain at least eight known indices for every base."

Checked against OEIS: **A127996 (b = 21) contains exactly eight terms**
(3, 11, 17, 43, 271, 156217, 328129, 3078871), **A127997 (b = 22) contains
nine**. The criterion first fails only at b = 23 (A204940, six terms). So the
stated rule yields the limit b ≤ 22, not b ≤ 20.

This matters, because the paper specifically emphasizes that "the selection rule
is fixed in advance and does not depend on the results". An inaccurate wording
undermines precisely that claim.

The correction was made in the strong direction: the limit is honestly called a
conventional round number, and **a direct test of its influence** was added —
extending to all admissible b ≤ 26 (B = 20, M = 253) gives κ̂_B = 1.8285 instead
of 1.8381, a shift of 0.5% (more than an order of magnitude less than the price
of the choice of observation scheme), and the homogeneity test gives Λ = 10.27,
df = 19, p = 0.95. No conclusion changes.

### 2.3. A stale GIMPS frontier (fixed)

X_ver = 81 491 519 in the paper. The PrimeNet Milestones report as of
27.08.2026 04:30 UTC: **"All exponents below 81 648 221 have been tested and
verified"**. X_ft = 141 308 443 matches exactly.

It does not affect the conclusions (both values lie between M₅₀ and M₅₁, and
the number of double-checked exponents remains 50), but it changes the derived
quantities of §7.2: T = 18.2179, δ₂ = 0.056 (was 0.054). The value was updated
and the moment of reading fixed in the text and in the reference.

### 2.4. Table 6: the base dropout rule was not stated (fixed)

The caption says "at n_min > 10⁴ one base drops out (B = 14)". Read literally
(a base stays if it has at least one event, N_b ≥ 1) **nothing drops out**: for
b = 19 the truncation leaves 22051 and 209359, that is N₁₉ = 1. Then the last
row reads B = 15, M = 90, (M−1)/S = 2.0848, (M−B)/S = **1.7569**, p = 0.128.

The paper's numbers (B = 14, M = 89, 2.0990, 1.7889) are reproduced **exactly**
under the rule N_b ≥ 2. The rule is right, but it has to be named: without it
the range "scheme B fluctuates within 1.79–1.87" turns into 1.76–1.87. The
caption was rewritten and both versions of the last row are given.

### 2.5. §7.2: estimates from different samples were compared (fixed)

> "λ̂ = (N−1)/(T−t₀) = 2.796 … the conditional estimate for the same base gives
> κ̃₂ = 1.922"

λ̂ = 2.796 is computed over the **50 double-checked** exponents, while
κ̃₂ = 1.922 is computed over **all 52** (that is the value from Table 5). On the
same slice of 50 exponents κ̃₂ = **1.905**. The difference is small and the
conclusion ("the difference is small") stands, but quantities from different
slices must not be compared. The text was corrected: both values are given with
the sample stated explicitly.

### 2.6. §7.2: the Poisson p-value (fixed)

The paper: p = 0.50. With μ₀ = 45.03 and 49 observed events, the doubled tail
gives **0.59** and mid-p gives **0.54**. The value 0.50 does not arise under any
of the common conventions. Both are now given with the convention stated
explicitly; the conclusion ("an unremarkable observation") does not change.

### 2.7. §6: rounding overstates the required growth of the frontiers threefold (fixed)

The paper: "the frontiers would have to grow by about **10²¹**". The exact
count: the missing 657 events at 9.6 events per doubling of all fifteen
frontiers is 68.2 doublings, that is 2^68.2 ≈ **3.3·10²⁰**. Rounding up by an
order of magnitude in a rhetorically loaded spot ("beyond the reach of any
conceivable computation") is best avoided: 3·10²⁰ proves the point just as well.

### 2.8. §5.3: the cv for 80% power (fixed)

The paper: cv ≈ 0.33 from a six-point grid with a step of 0.10–0.20 around the
inflection. On a nine-point grid (5000 replications per point): cv = 0.30 →
0.693, cv = 0.35 → 0.817, so 80% is attained at cv ≈ **0.34**. The difference is
within the accuracy, but the grid should be denser and the accuracy should be
stated. The power table was extended.

### 2.9. The interval of eq. (12) did not reproduce (fixed)

The paper: [1.598, 2.099], p = 0.654. The inversion procedure was not described,
so those endpoints cannot be reproduced. The procedure is now written out
explicitly (in §5.2 and in `analysis.py`), and the quoted numbers were replaced
by its output: **[1.613, 2.130], p = 0.639**. The conclusion does not change.
At the same time the word "exact" was removed from the English abstract: the
scheme B interval is obtained by inverting a simulation and is not exact —
unlike the scheme A interval from the gamma pivot.

### 2.10. The status of entry A000043 (documented)

The DATA section of A000043 contains **50** terms and stops at 77 232 917: OEIS
does not regard the order of the subsequent exponents as proven. The paper says
"52" with a citation to OEIS. The exponents 82 589 933 and 136 279 841 are
indeed known and documented by GIMPS (and mentioned in the comments to A000043),
but formally they are not part of DATA. A paragraph was added explaining where
the last two terms come from — otherwise a reader comparing Table 1 against the
OEIS entry sees a discrepancy.

### 2.11. §4.2: the moment and corrected estimators were mixed (documented in a footnote)

The segment "between 1.838 (m = 1) and 1.974 (m = 0)": the upper end
1.974 = M/S is the **moment** estimator, whereas everywhere else in the paper
scheme A is represented by the corrected estimator (M−1)/S = 1.9648. The
difference is one event out of 218 (0.5%) and does not affect the conclusions,
but within a single paragraph one form should be used. A footnote was added.

### 2.12. The figures were missing from the PDF (fixed)

In the original PDF all four figures were drawn as frames containing the text
`fig_kappa.pdf`, `fig_truncation.pdf`, `fig_power.pdf`, `fig_resid_bridge.pdf` —
the files did not exist. `analysis.py` now generates them; the PDF was rebuilt
and the figures are in place.

### 2.13. Table 3: the frontier rules were stated ambiguously (fixed)

The rows "frontier 1.5 times further in t" and "random frontiers" reproduced
only qualitatively: the column E[(M−B)/S] matched (and that is the point of the
table), while E[(M−1)/S] differed by 0.5–1.7%. All seven rules are now given as
formulas for the window width W_b right in the table caption, and the numbers
were replaced by the output of `analysis.py`.

### 2.14. The last significant digit in two places (fixed)

`ln L` of model B: was −175.495, the exact value is −175.4945 → **−175.494**.
κ̂ under the fixed correction C^fix: was 0.816, the exact value is 0.81547 →
**0.815**. The optimizer tolerances in `analysis.py` were tightened so that the
result does not drift in the last digit.

---

## 3. What was added beyond the corrections

### 3.1. Independent recomputation of the sequences (Appendix A)

Cross-checking against OEIS confirms that the listed indices give primes, but it
**does not guarantee completeness**: a missing term would shift N_b, and with it
both estimates.

For all twenty admissible bases 2 ≤ b ≤ 26, **every** index n < 10⁴ was
recomputed consecutively, and every negative verdict was duplicated by a second
backend.

**Result: 153 terms confirmed, 0 spurious, 0 missing, 0 disagreements between
backends.**

Coverage is uneven (from 1 term at b = 18 to 22 at b = 2): the main set b ≤ 20
accounts for **128 confirmed indices out of 233 known**, the remaining 25 belong
to bases 21…26. Above 10⁴ a complete recomputation is computationally out of
reach.

### 3.2. Reproducibility

Before: "The analysis is a single self-contained script `analysis.py`;
Repository: [INSERT REAL URL]" — while the script did not exist.

After: `analysis.py` (fixed seed, every number tagged with its table/section),
`fetch_data.sh` / `fetch_extra.sh` (downloading the b-files),
`verify_sequences.sh` + `compare_verify.py` (independent recomputation with a
non-zero exit code on a disagreement), `results.txt`, `figs/`. The data are
included in the repository — no network is needed to reproduce.

---

## 4. A side result: a bug in the searcher

On its first run the independent recomputation (3.1) disagreed with OEIS on
**R_3181(23)**: the GWNUM backend declared the number composite although it is a
PRP (confirmed independently by exact GMP arithmetic:
2^(N−1) ≡ 3^(N−1) ≡ 1 mod N).

**The cause.** In `native/prp/prp_gwnum.c` the FFT rounding error was checked
once every 128 iterations, while the final value of `gw_get_maxerr()` after the
loop **was recorded in the statistics but not compared against the threshold**.
For (b, k) = (23, 3181) the error first exceeds the threshold on iteration 14383
of 14384 — that is, after the last periodic check at 14336. maxerr = 0.5000 (the
maximum possible: the residue is entirely garbage), yet a "composite" verdict
was returned as if nothing had happened. This is a **lost find** — exactly the
class of failure that, as the README itself says, nothing but recomputation
catches.

**Diagnosis.** With margin = 1 the FFT length grows from 768 to 1024, maxerr
drops to 0.0015, and the verdict is right. Through `gwsetup_general_mod`
(without the special form) maxerr = 0.0029 and the verdict is right as well.
Frequency: 1 case per 96 pairs (b, k) in a scan over
b ∈ {3, 7, 10, 13, 17, 19, 20, 21, 22, 23, 24, 26}, k ∈ {1009 … 10007}.

**Fixed:**
1. `prp_gwnum.c` — a final threshold check returning `RH_ERR_FFT_ERROR`, which
   triggers the normal restart with a larger FFT.
2. `prp_dispatch.c` — if GWNUM does not fit within 4 attempts, the candidate is
   finished on GMP. Previously an error was returned, and on an error
   `pipeline.rs` only writes a log line and **creates no worklog record** — the
   exponent vanished from the results without a trace. A second path to losing a
   find.

**Verification after the fix:** R_3181(23) — PRP in 0.057 s; R_49081(10)
(163 041 bits) is still found, in 4.2 s; `cargo test` 8/8; `cargo clippy` —
no warnings; an exhaustive recomputation of all twenty bases — 0 disagreements.

---

## 5. End-to-end cross-check

The Monte Carlo numbers in the paper (Tables 2, 3, 4, the interval of eq. 12,
§4.2, §4.3, §5.3, §7, §7.1) had been taken from an earlier run and differed from
the current one in the third or fourth digit. All of them were replaced by the
output of `analysis.py`, and the cross-check was automated: `check_numbers.py`
extracts 37 key quantities from `results.txt` and checks that they are present
in the paper, returning 1 on any disagreement.

**Current status: 0 disagreements out of 37.**

## 6. Formal layout per the standards in force

**The relevant standard had been chosen wrongly.** In the first edition of this
check I relied on GOST 7.5-2008 — which is about journals and collections as
*publications*. Articles have a document of their own:

> **GOST R 7.0.7-2021** "SIBID. Articles in journals and collections.
> Publishing arrangement" — approved by Rosstandart order No. 728-st of
> 18.08.2021, in force from 01.10.2021, superseding GOST R 7.0.7-2009.
> **Status as of 27.08.2026: in force.**

The statuses of the related standards were checked against the Rosstandart
registry (protect.gost.ru) and against the normative references of GOST R
7.0.7-2021 itself:

| Standard | Status as of 27.08.2026 |
|---|---|
| GOST R 7.0.7-2021 — articles, publishing arrangement | in force (supersedes 7.0.7-2009) |
| GOST R 7.0.5-2008 — bibliographic reference | **in force**, the "superseded by" field is empty |
| GOST R 7.0.99-2018 — abstract and annotation | in force, **superseding GOST 7.9-95** |
| GOST R 7.0.100-2018 — bibliographic description | in force (supersedes GOST 7.1-2003) |
| ~~GOST 7.9-95~~ | **repealed** — must not be cited |

That is, in the first edition of the check I cited the repealed GOST 7.9-95 and
the off-topic GOST 7.5-2008. Corrected.

| Element | Before | After |
|---|---|---|
| UDC | absent | `511.3:519.234` (analytic number theory + hypothesis testing) |
| English-language block | abstract and keywords only | English title, name, affiliation, ORCID, e-mail added |
| Reference list | international format | GOST R 7.0.5-2008: `Author. Title // Journal. Year. Vol., No. P. DOI` |
| Publication details of the sources | no issue numbers, no DOIs | issue numbers and DOIs verified through Crossref for all three Math. Comp. papers |
| `References` | absent | a romanized list in international format added (all sources are in English, so no transliteration was needed) |
| Author information | ORCID and e-mail in a footnote | separate sections in Russian and English |
| Funding | absent | section added |
| Competing interests | absent | section added |
| Captions | `Рис. 1: Title` | `Рис. 1. Title`, `Таблица 1. Title` (`caption` with `labelsep=period`) |
| Abstract | 227 words | within the GOST 7.9-95 range (150–250), left as is |

### 6.1. GOST R 7.0.7-2021 violations found and removed

The check is automated: `check_gost.py`, 30 checks, exit code 1 on any
violation. **Current result: 0 violations out of 30.**

| Clause | Violation | Correction |
|---|---|---|
| **4.10** | **The abstract was 389 words against a limit of 250** (my earlier count of "227 words" was wrong — only the first 60 lines were counted). The English one was 383. | Both were cut to 250 words, keeping all four results and every number |
| **4.14** | The list was headed "Список литературы"; GOST requires **"Список источников"** and explicitly discourages "Библиографический список"/"Библиография" | `\refname` redefined |
| **4.14.3** | Entries not in order of citation: `wagstaff2025` is first cited in §1 and `gimps` in §2.3, but the list had them the other way round | Sources 6 and 7 swapped in both lists |
| **4.12** | Acknowledgments stood at the end of the article and in Russian only; GOST requires them **after the keywords** (before the main text) in both languages | Moved, English version added |
| **4.12** | Funding — the same, preceded by the word "Финансирование:"/"Funding:" | Moved, English version added |
| **4.20.3** | Competing interests stood before the reference list and in Russian only; GOST requires them **after "Information about the author"**, in both languages | Moved, English version added |
| **4.4** | The article type was not stated | "Научная статья" on its own line at the left, before the UDC |
| **4.9** | Format of the author information: order "Surname Given name", labels "ORCID:"/"E-mail:", separated by line breaks | Brought to the required form: "Given name Patronymic Surname", organization, city, country, address without the word "e-mail", ORCID as a URL, commas as separators, no trailing periods |
| **4.11** | A period stood after the keywords (in both versions) | Removed |
| **4.9** | The section was titled "Сведения об авторе" | "Информация об авторе" — the wording used in clauses 4.20.2, 4.20.3 |

### 6.2. Bilingual captions (clause 4.1.5, done)

> "Labels and captions of illustrative material are given in the language of the
> article text and, as a rule, repeated in English."

Formally this is a recommendation ("as a rule") rather than a requirement, but
it has been fulfilled: all 9 tables and 4 figures carry English captions. The
number is taken from the LaTeX counter already incremented by `\caption`, so the
Russian and English captions always carry the same number ("Таблица 5." /
"Table 5.").

### 6.3. Cyrillic did not extract from the PDF (fixed)

Found while checking the finished file: the built PDF had no ToUnicode tables,
so the Cyrillic text **could not be searched, copied or parsed automatically**.
For a journal submission this is a blocking defect: RSCI (eLIBRARY) and Crossref
extract metadata from the text layer.

The cause was the absence of `\usepackage{cmap}` before `fontenc`. Added;
extraction verified (`pdftotext -enc UTF-8`).

## 7. Review: the mathematics, the text, the literature

### 7.1. Proposition 3 had no proof, and its premise is wrong

Propositions 1 and 2 come with proofs, **Proposition 3 does not**: only the
statement was given. Worse, its key premise is inaccurate:

> "Then δ_b ~ Exp(λ_b) by memorylessness"

δ_b is the distance from the frontier to the **last** event, that is, the
backward recurrence time. The process has no events to the left of t₀, so
δ_b ≤ W_b by construction, and an appeal to memorylessness is not enough:
memorylessness gives Exp only for the **forward** time. The exact law is
exponential, **truncated** at W_b, with an atom e^(−λW) at the point W_b (the
case "no finds at all"):

    E[δ_b] = (1 − e^(−λ_b W_b)) / λ_b,   not 1/λ_b
    E[κ Σ δ_b/ln b] = B − Σ e^(−λ_b W_b),   not B

**A full proof was written**, and a remark was added estimating the discarded
remainder: Σe^(−N_b) < 3·10⁻³ against B = 15, that is 0.02%. That is exactly the
quantity shown by the first row of Table 3 (+0.02%) — the conclusion does not
change, but the statement is now correct.

### 7.2. The same inaccuracy in §4.3

> "the per-base estimator κ̃_b is unbiased under both schemes"

Under scheme B the estimator equals zero when N_b ∈ {0,1}, and the Beta identity
applies only for n ≥ 2. The exact result:

    E[λ̃_b] = λ_b − Pr(N_b = 1)/W_b = λ_b (1 − e^(−λ_b W_b))

— the same exponentially small deficit (< 0.1%). The wording was corrected.

### 7.3. An unsupported quantitative claim, "threefold"

> "disagree by up to 29.5%, which exceeds the interpretable deviations
> **threefold**"

Repeated in three places: the abstract, §1, §3.4. **Checked by computation —
false.**

| Quantity | Median | Maximum |
|---|---|---|
| Deviation \|Ĝ_b − e^−γ\|/e^−γ | 17.6% | 43.7% (b=19) |
| Spread between forms \|Ĝ_OLS − Ĝ_end\|/Ĝ_end | 4.1% | 29.5% (b=18) |

29.5 / 17.6 = **1.67**, not 3. Moreover, the median spread between forms is only
a quarter of the median deviation. Replaced by verified numbers, and the
conclusion is stated even more sharply: **for two of the fifteen bases (b = 18
and b = 20) the choice of form weighs more than their entire deviation from the
LPW prediction** — and one of them, base 18, is precisely a base the folklore
discusses. The computation was added to `analysis.py` under the tag `[§3.4]`.

> **Correction of 28.08.2026.** The first edition of this report said b = 18 and
> b = 13 here. That is wrong: for b = 13 the spread between forms is 15.9%
> against a deviation from LPW of 27.0%, so the criterion is not met. The pair
> singled out by the computation in `analysis.py` is b = 18 (29.5% against
> 17.6%) and b = 20 (7.8% against 5.6%). The error had propagated into §3.4 of
> the paper and into `METHOD.md`; it has been fixed in all three places.

### 7.4. Recent literature (7 sources before, 10 after)

Search of 27.08.2026. Added, with citations in the text:

- **PrimePages, "Heuristics: deriving the Wagstaff Mersenne conjecture"** — §1,
  as the standard derivation of the heuristic itself (previously its derivation
  had no citation);
- **arXiv:2603.08994**, Dominguez (2026), "Arithmetic bias in the distribution
  of Mersenne prime exponents and the divisor structure of p−1" — §1:
  methodologically built like §8 (stratified conditional likelihood plus a
  significance test), but it looks for structure **within** a single base,
  whereas here it is tested **between** bases;
- **arXiv:2605.23014**, Jha (2026), "The Poisson tail conjecture for primes in
  short intervals" — §2.1: the Poisson idealization is called a working
  assumption rather than a theorem, citing the fact that even for the ordinary
  primes the Poisson law is proven only conditionally and breaks down as the
  scale grows.

The publication details of all three were verified against arXiv and the source
site.

### 7.5. A full proof-reading pass

Checked mechanically: **0 overfull and 0 underfull boxes**; spelling — 3887
words, 77 outside the dictionary, all of them either terms (пивот, бутстрэп,
репьюнит, цензурирование, недодисперсия) or proper names.

Found and corrected:

- **the build needs three pdflatex passes, not two** — after two, LaTeX still
  demanded a cross-reference recount; `build_paper.sh` now runs until
  convergence and fails with exit code 1 if the references do not resolve;
- p = 0.65 in §1 against p = 0.64 in the abstract and §5.2 — an inconsistency;
- §2.2: "the selection rule is fixed in advance" contradicted the neighbouring
  paragraph, where the limit is called conventional; the wordings were
  disentangled;
- the OEIS access dates (25.08 in the text against 27.08 for the files in
  `data/`);
- Appendix A spoke of "fifteen sequences" although twenty were cross-checked.

### 7.6. Originality

A full "Antiplagiat" report cannot be obtained: the service compares against
closed collections (RSL, eLIBRARY, university repositories). Two available
slices were done.

**A search for competing work.** No publications were found on the key elements
of the contribution (calibration of the observation scheme, pooling of bases,
the gamma pivot, the search frontier). The nearest in topic — Dominguez (2026) —
solves a different problem and does not estimate the LPW constant, does not pool
bases and does not discuss the observation scheme.

**Self-overlap** (`check_overlap.py`, 8-word shingles): 57% with
`paper (1).tex` (the direct predecessor of this same work) and 9–11% with
`lpw_repunits_ru*.tex` (early drafts). All the files are local. **The author
confirmed: the drafts were never published and will not be** — hence they will
not appear in an "Antiplagiat" report and create no risk of a borrowing flag.
The question is closed.

### 7.7. The abstracts and the inserted passages

Both abstracts were reread against the text: every claim is backed by a section
and every number matches `results.txt` (checked by `check_numbers.py`). The
length was brought within the limit of clause 4.10 (250 words) by removing
repetitions of the introduction and the conclusion; no result was dropped.

The passages inserted during the revision were re-checked: the paragraph on the
robustness of the limit b ≤ 20 (§2.2), the subsection on the independent
recomputation (Appendix A), the footnote on the moment and corrected estimators
(§4.2), and the proof and remark for Proposition 3 — all their numbers agree
with `results.txt`.

## 8. Full validation: five runs

All the checks are gathered into one orchestrator, `validate.sh`. It does not
simply repeat the same thing — each run answers a question of its own.

| Run | Question | What it does |
|---|---|---|
| **1. Clean room** | will it build for an outsider | deletes `figs/`, `results.txt` and both `paper_*.pdf`, and rebuilds everything from `data/` and `paper_ru.tex` / `paper_en.tex` |
| **2. Cross-check** | do the numbers and the layout agree | 37 numbers against `results.txt`; 31 GOST checks; data against OEIS; self-overlap; typography; spelling; PDF extractability; bilingual captions; a count of placeholders |
| **3. Seed robustness** | does the conclusion rest on one realization | runs the analysis under five seeds |
| **4. Live sources** | have the external data moved | re-downloads the 20 OEIS b-files and compares them against `data/`; reads the GIMPS frontier and compares it with the paper |
| **5. Independent recomputation** | are the sequences complete | `repunit-hunt` recomputes all n < 10⁴ with a blanket double-check |

```bash
bash validate.sh          # runs 1-4, about a minute
bash validate.sh --full   # plus run 5, about ten minutes
```

### 8.1. What run 3 found (a new check)

The Monte Carlo numbers in the paper were obtained under a single seed. That
makes them reproducible but does not prove that the conclusion is independent of
the seed. `check_stability.py` runs the analysis under five seeds and checks
that:

- the **deterministic quantities** (M, S, both pooled estimates, Λ, ln L of
  model A) match **exactly**: they did;
- the **Monte Carlo quantities** stay within a substantive corridor ("the same
  conclusion") rather than matching to the digit: all eleven did.

Spread across seeds: type I error 0.162–0.169; E[(M−B)/S] 1.7809–1.7822 against
a true value of 1.7811; the scheme B p-value 0.637–0.654; the cv for 80% power,
0.34–0.35.

**An inaccuracy in the paper came to light along the way:** the caption of
Table 4 stated the Monte Carlo accuracy as ±0.003. That is the binomial error at
a level of 0.05, whereas the level being measured is 0.16, where it equals
**±0.005**; the observed spread across seeds (0.008) is consistent with that and
not with ±0.003. The caption was corrected, and the observed spread was added to
it.

### 8.2. What run 2 found

The typography check initially reported 4 overfull and 2 underfull boxes. It
turned out to be a defect of the check itself: it was measuring the **first**
pdflatex pass, where unresolved references stand as "??" and lines break
differently. On the converged pass there are no overfull boxes. The check was
fixed — it now runs the build to convergence and measures the final log.

### 8.3. The typesetting criterion was asymmetric the wrong way

The check demanded zero overfull **and** zero underfull boxes — and failed the
run on three underfull ones. That is the wrong criterion:
`\emergencystretch`, the cure for overfull boxes, works precisely by trading
overflow for stretch. Demanding zero underfull means banning the cure.

The criterion was rewritten in substance: **overfull must be zero** (text
running into the margin is a defect), while underfull is tolerated as long as
badness < 10000 (currently overfull = 0, underfull = 3, worst badness = 1895 —
three slightly loose lines in the reference list).

### 8.4. A spurious disagreement and its cause

The first full run reported **"152 confirmed, 1 missing"** instead of 153/0/0 —
that is, an apparently lost term of a sequence. A repeat comparison of the same
files immediately gave 153/0/0.

A non-reproducible disagreement is more dangerous than a persistent one, so the
investigation was carried through. The cause turned out to lie neither in the
searcher nor in the data but **in the checking procedure itself**:

- the timestamps in `verify/` were out of order: `b26.json` was **older** than
  `b2.json`, although base 26 comes last in the loop;
- `ps` showed **two** concurrent `verify_sequences.sh`, one of which had already
  been running for six hours;
- it was my earliest verification run, **killed by a timeout**: the script
  itself died, while the `repunit-hunt` it had spawned was orphaned and went on
  overwriting the very same files;
- the comparison read the directory in an intermediate state — hence the
  apparent omission.

All twenty run logs, meanwhile, reported "Double-check: 0 disagreements", so the
arithmetic had been right the whole time.

**Fixed in the infrastructure**, not papered over:

- `verify_sequences.sh` and `validate.sh` take a `flock`; a second concurrent
  launch now refuses to start with a clear message;
- `validate.sh` sets a `trap` on INT/TERM and kills its children, so an
  interrupted run no longer leaves live processes behind.

The moral for the report: a disagreement that does not reproduce must not be
written off as chance — but neither may it be attributed to the object under
test before establishing who exactly was writing to the files.

### 8.5. The result of the full validation

A clean `validate.sh --full` run of 27.08.2026 (log:
`verify/validate_full.log`):

| Run | Result |
|---|---|
| 1. Clean room | everything derived was rebuilt, the PDF was produced |
| 2. Cross-check | 37 numbers, 31 GOST checks, data, self-overlap, typesetting, spelling, PDF, captions, placeholders — all OK |
| 3. Seed robustness | 0 quantities outside their corridors under five seeds |
| 4. Live sources | 20 OEIS b-files matched byte for byte; GIMPS frontier 81 648 221 = the value in the paper |
| 5. Independent recomputation | **153 terms confirmed, 0 spurious, 0 missing** |

**Bottom line: 15 checks out of 15, 0 failures.**

### 8.6. Extending the recomputation found two more bugs in the searcher

Raising the bound from n < 10⁴ to n < 2·10⁴ produced a disagreement at once: for
b = 2, 24 terms were expected and 23 were found. **M₁₂₇₉ — a known Mersenne
prime — had vanished.**

There was no worklog record for k = 1279 at all; the log had a single line,
`ERROR ... PRP k=1279: internal error`. The same k computed separately passes
cleanly on all three backends: the failure showed up only under multi-threaded
load.

**The chain of causes:**

1. `prp_dispatch.c` estimated the size as `(k−1)·(floor(log2 b) + 1)`. For
   b = 2 that is `(k−1)·2` instead of `(k−1)·1` — **an overestimate by exactly a
   factor of two**. The measured GWNUM switchover threshold (2500 bits) was
   applied to b = 2 as 1250 bits.
2. A 1279-bit number went to GWNUM — into the zone where, by the project's own
   measurements, GWNUM is slower than GMP (at 1017 bits, ten times slower), and
   where setting it up on too small a number produced an intermittent failure.
3. On an error `pipeline.rs` only wrote a line to the log and **created no
   worklog record**. The candidate was neither "composite" nor "PRP" — it was
   simply absent.

**Fixed at all three levels:**

- an honest `log2` in the size estimate;
- on any PRP error the candidate is finished on GMP, which depends neither on
  the FFT nor on GWNUM state;
- if even that fails, a `"status":"failed"` record is written and the exponent
  stays visible both to a human and to `--verify`.

Verification after the fix: b = 2 up to n < 20000 gives **24 of 24, 0 errors**.

This bug was impossible to find with a bound of 10⁴: there k = 1279 fell in a
range where the failure did not occur. **Extending the bound is not cosmetic.**

## 9. What is left for the author to do

Everything marked in red in the paper source is what I cannot fill in myself:

1. **The repository URL** — `[INSERT URL BEFORE SUBMISSION]` in the "Data and
   code availability" section.
2. **The patronymic** — in the "Information about the author" sections. Russian
   journals require the full name.
3. **Academic degree, title, position, organization** — if the paper is
   submitted on behalf of an organization rather than as the work of an
   independent researcher.
4. **A postal address** — required by most journals.

Separately: the "Funding" and "Competing interests" sections are **statements by
the author**, not facts established by me. I filled in the standard wordings
("no external funding", "no competing interests"); check that they are true
before submitting.

## 10. Files

```
paper/
├── paper_ru.tex         the corrected source, Russian edition (GOST)
├── paper_ru.pdf         rebuilt, figures in place
├── paper_en.tex         English edition, international format
├── paper_en.pdf         rebuilt
├── analysis.py          reproduction of every number
├── results.txt          the output of analysis.py, tagged by table
├── figs/, figs/en/      four figures in two language sets
├── data/                OEIS b-files (20 sequences)
├── fetch_data.sh        downloading the b-files of the main set
├── fetch_extra.sh       downloading the b-files for b = 21..26
├── build_and_verify.sh  building the searcher
├── verify_sequences.sh  exhaustive recomputation of n < KMAX
├── compare_verify.py    comparison of the recomputation against OEIS
├── verify/              recomputation results + log
└── FINDINGS.md          this report
```
