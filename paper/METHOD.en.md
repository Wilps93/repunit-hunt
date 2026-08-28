*[Русская версия](METHOD.md) · English*

# Methodology: how to estimate the LPW constant from generalized repunit primes

Extracted from `analysis.py` — not a retelling of it, but a description of what
actually runs. Every item carries the tag under which the corresponding number
is printed in `results.txt` and checked by `check_numbers.py`.

It works without knowledge of the undocumented search bounds. It gives exact
intervals where they are exact, and honestly simulated ones where they are not.

---

## 0. Notation and limits of applicability

| Symbol | Meaning |
|---|---|
| `b` | the base, `2 ≤ b`, **not a perfect power** (otherwise `R_b(n)` factors algebraically) |
| `n` | the index; `R_b(n) = (bⁿ−1)/(b−1)` can be prime **only for prime n** |
| `t` | `t = ln n` — the working axis. Everything below lives in logarithms |
| `t₀` | `t₀ = ln n_min`, the smallest known index. **The origin, not an event** |
| `N_b` | the number of events strictly to the right of `t₀` = (number of known indices) − 1 |
| `D_b` | `D_b = t_{N_b} − t₀ = ln n_max − ln n_min` |
| `κ` | the constant being estimated. LPW: `κ = e^γ = 1.781072…` |
| `λ_b` | the intensity of the process for base b: `λ_b = κ / ln b` |
| `B`, `M`, `S` | the number of bases; `M = Σ N_b`; `S = Σ D_b / ln b` |

> **The main trap.** `D_b` is a difference of **logarithms**, not of indices.
> For `b = 2`: `D₂ = ln 136 279 841 − ln 2 = 18.0371`. Take
> `n_max − n_min = 136 279 839` instead and the quantity is inflated by a factor
> of 7.6 million, driving the estimate from 1.838 down to ~10⁻⁷.

**The model.** The indices form a homogeneous Poisson point process on the `t`
axis with intensity `λ_b = κ/ln b`. This is a working assumption, not a
theorem: even for the ordinary primes the Poisson law has been established only
conditionally.

---

## 1. Sufficient statistics  `[Табл. 1]`

For each base exactly three numbers are needed: `N_b`, `n_min`, `n_max`. The
full lists of indices are needed only for §5 (truncation) and §7 (regression).

```python
t   = log(idx)          # idx -- the sorted list of known indices
N   = len(idx) - 1
t0  = t[0]
D   = t[-1] - t[0]
expo = D / log(b)       # the base's contribution to S
```

Selection of bases: all non-perfect-powers in the declared range. A base stays
in the sample if `N_b ≥ min_events` (default 1; the truncation table uses 2 —
**the rule must be declared**, it changes the result).

---

## 2. Per-base estimates  `[Табл. 5]`

```python
G_end   = D / (N * log(b))          # the endpoint form
G_ols   = polyfit(arange(N+1), t, 1)[0] / log(b)   # OLS slope of t_k on k, k = 0..N
kappa_hat   = N     * log(b) / D    # MLE, biased upwards by N/(N-1)
kappa_tilde = (N-1) * log(b) / D    # exactly unbiased
```

**What is unbiased here and what is not** — this is the main negative result:

- `G_end` and `G_ols` are **exactly** unbiased. The widespread suspicion of a
  bias due to right-censoring is **false**.
- Only the reciprocal `κ̂ = 1/Ĝ` is biased, by the factor `N/(N−1)`, and that
  is Jensen's inequality, not censoring.
- Unbiased `κ` and `1/κ` cannot be reciprocals of each other: the unbiased pair
  is `(Ĝ_b, κ̃_b)`, and `κ̃_b ≠ 1/Ĝ_b`.

The exact confidence interval (scheme A) and the PIT quantile:

```python
lo, hi = gamma.ppf(0.025, N) / expo,  gamma.ppf(0.975, N) / expo
pit    = gamma.cdf(KAPPA_LPW * expo, N)
```

**Do not interpret the ordering of the bases among themselves.** The forms of
the estimator disagree by up to 29.5% (b = 18), against a median deviation from
`e^−γ` of 17.6%. For b = 18 and b = 20 the choice of form weighs more than
their entire deviation from LPW: b = 18 moves from 3rd place to 10th of 15 from
a change of form alone. The "luckiness" label describes the choice of estimator,
not a property of the primes.

---

## 3. Two observation schemes — this is where the whole point lies  `[§5.2]`

### Scheme A — "until the N-th event" (inverse sampling)

We observe until `N_b` events have accumulated. Then `D_b ~ Gamma(N_b, λ_b)`,
and `κS ~ Gamma(M, 1)` — an exact pivot.

```python
kappa_A = (M - 1) / S
lo, hi  = gamma.ppf(0.025, M) / S,  gamma.ppf(0.975, M) / S
p       = 2 * min(gamma.cdf(KAPPA_LPW * S, M), gamma.sf(KAPPA_LPW * S, M))
```

### Scheme B — a fixed frontier (this is how the data arose)

The search covered the whole of `(t₀, ln L_b]`; `N_b` is a random variable and
the frontier `L_b` is fixed. With `N_b` fixed, `D_b` is informative about the
frontier, not about the intensity — so the scheme A likelihood stops being a
likelihood.

Let `W_b = ln L_b − t₀` and `δ_b = ln L_b − t_{N_b}` (with `δ_b = W_b` when
`N_b = 0`). Then `δ_b` is an exponential variable **truncated** at `W_b`:

```
Pr(δ_b > x) = e^(−λ_b x)   for 0 ≤ x < W_b
Pr(δ_b = W_b) = e^(−λ_b W_b)          <- an atom, not a tail value
E[δ_b] = (1 − e^(−λ_b W_b)) / λ_b     <- NOT 1/λ_b
```

An appeal to "memorylessness" is **not enough** here: the process has no events
to the left of `t₀`, so `δ_b ≤ W_b` by construction.

The full exposure `S_full = S + Σ δ_b/ln b = Σ W_b/ln b` is **deterministic**,
and `κ·S_full = Σ λ_b W_b = E[M]` is a moment identity. Hence

```python
kappa_B = (M - B) / S
```

because `E[κ Σ δ_b/ln b] = B − Σ e^(−λ_b W_b)`.

**The estimator is unbiased up to the discarded remainder,** not absolutely.
The remainder is the expected number of bases with no find at all; for
`λ_b W_b ≈ N_b ≥ 7` it is below `3·10⁻³` against `B = 15`, that is, 0.02%.
Simulation gives exactly that.

> **What you must not do.** Do not substitute `W_b = ln n_max − t₀`. That makes
> `δ_b = 0` and you get scheme A at m = 0, that is, `M/S`. The whole point of
> `(M−B)/S` is that **`L_b` need not be known** — it does not enter the formula.

### The price of the choice — and why it cannot be removed

`(M−1)/S` and `(M−B)/S` are functions of **the same** sufficient statistics
`(N_b, D_b)`. No processing of the data will say which is right: the choice is
determined by how the data were collected. The price is 7%, twice the residual
gap with LPW.

Getting round it via "pieces that are unbiased by construction" does not work:
the per-base `κ̃_b` is unbiased under both schemes (with the same exponentially
small correction), but the weighted mean with inverse-variance weights has a
bias of +7.5% under scheme B — the weights are themselves random and correlate
with the estimates.

Hence the only radical remedy: **publish the search frontiers `L_b`**.

---

## 4. Simulating scheme B  `[Табл. 3, Табл. 4]`

The **whole process** is simulated, not just `δ_b`:

```python
for bd in bases:
    lam = kappa / bd.lnb
    W   = frontier_rule(bd, lam)        # window width
    n   = poisson(lam * W)              # the number of finds is random
    n   = resample_while(n < 1)         # every base has at least one find
    u   = uniform(0, 1)
    D   = W * u**(1/n)                  # the maximum of n uniforms on (0, W]
    M  += n
    S  += D / bd.lnb
```

The frontier placement rules are stated through `W_b` and **must be written
out**:

| rule | `W_b` |
|---|---|
| `E[#] = observed` | `N_b / λ_b` |
| `+s mean intervals` | `(N_b + s) / λ_b` |
| `c times further in t` | `c · D_b` |
| random | `(N_b/λ_b) · U`, `U ~ Unif(0.7, 1.8)`, drawn separately for each base and each replication |

The scheme B confidence interval comes from inversion: for a grid of `κ`,
scheme B is simulated and one seeks the `κ` at which the observed statistic
falls on the 0.025 and 0.975 quantiles of the simulated distribution.

**What the calibration shows.** Under scheme B a nominal 5% test rejects a true
hypothesis in 16.2% of cases (spread across five seeds: 0.162–0.169) — the
claimed precision of the intervals is overstated roughly threefold.

---

## 5. Homogeneity of the bases  `[§5.3]`

```python
kappa_hat = M / S
Lambda    = 2 * sum(N_b * log(kappa_hat_b / kappa_hat))   # ~ chi2(B-1)
```

**A non-rejection is informative exactly to the extent that the test is
powerful.** The test attains 80% power only at a coefficient of variation of
`κ_b` around 0.34. So the correct conclusion is "base-dependent frequencies are
**not required** by the data", not "they do not exist".

---

## 6. Power  `[Табл. 7, §6]`

Against `κ = r·e^γ` at level α, in closed form:

```python
lo, hi = gamma.ppf(a/2, M), gamma.ppf(1-a/2, M)
power  = gamma.cdf(lo*r, M) + gamma.sf(hi*r, M)
```

| sample | deviation resolvable at 80% power |
|---|---|
| Mersenne only, M = 51 | ≥ 49.8% |
| 15 bases pooled, M = 218 | ≥ 21.2% |
| for 10% one needs | M ≈ 875 |
| for 5% one needs | M ≈ 3319 |

Doubling all fifteen frontiers yields ~9.6 additional events. The missing 657
events amount to ~68 doublings, that is, a growth of the frontiers by a factor
of `3.3·10²⁰`.

**A rule for reading this literature.** A report that "the data agree with LPW
to within a few percent" is not evidence for LPW at the level of a few percent:
it is compatible with a constant that is wrong by a factor of one and a half.

---

## 7. Robustness — a mandatory part, not an appendix

| check | tag | what it does |
|---|---|---|
| lower truncation | `[Табл. 6]` | varies `t₀`; the trend of the scheme A estimate disappears under scheme B |
| leave-one-base-out | `[§5.4]` | no single base determines the conclusion |
| status of the data | `[§5.4]` | restricting b = 2 to the double-checked exponents; excluding the PRP indices |
| selection limit | `[§2.2]` | extending the set of bases |
| generator seed | `check_stability.py` | five seeds; the deterministic quantities must match exactly, the Monte Carlo ones must stay within their corridors |

---

## 8. An error of the same kind in a different guise  `[§7]`

Regressing the counting function on `t` at the event points produces an
apparently significant intercept. It is an artifact of conditioning, and there
are three confirmations of that:

- the residuals of the cumulative count are autocorrelated (`ρ₁ = 0.825`);
  Newey–West with eight lags raises the standard error by a factor of 1.64;
- the correct null model holds `[t₀, T]` fixed and conditions on `N−1` events —
  which means `N−1` uniform points on `(t₀, T]`;
- **a bootstrap over the event points does not help**: it reproduces the
  variability within the realized configuration and cannot recover the
  variability of the process that produced it. The correct null distribution is
  6.5 times wider than the bootstrap one.

Estimating `λ` from the last event forces the residual to vanish at both ends —
the trajectory becomes an analogue of a Brownian bridge, for which a negative
excursion in the middle arises of its own accord.

---

## 9. The result on the data of 27.08.2026

```
M = 218,  S = 110.4423,  B = 15

scheme A:  (M-1)/S = 1.9648   95% [1.7205, 2.2444]   p = 0.142
scheme B:  (M-B)/S = 1.8381   95% [1.613, 2.130]     p = 0.639
```

The excess over `e^γ` falls from 10.3% to 3.2%. The range generated by the
uncertainty in the frontier assumption: from 1.838 (frontier independent of the
finds) to 1.974 (search stopped at a find). **Both ends are compatible with
LPW.**

Homogeneity: `Λ = 7.40`, df = 14, `p = 0.92`.
Structural corrections: none gives a significant improvement.

---

## 10. How to reproduce

```bash
python3 analysis.py        # all the numbers -> results.txt, figures -> figs/
python3 check_numbers.py   # cross-check the paper against results.txt
python3 check_stability.py # seed robustness
bash validate.sh --full    # everything at once, including the independent recomputation of the data
```

The generator seed is fixed, so the Monte Carlo quantities reproduce bit for
bit. The data are included in the repository — no network is needed to
reproduce them.
