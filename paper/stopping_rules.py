"""[S13] Scheme B under informative and non-informative stopping rules.

For each rule, simulates the 15 bases of the main data set (lambda_b = e^gamma / ln b,
t measured from t_0) and reports
  * E[(M-1)/S]/kappa, E[(M-B)/S]/kappa  -- relative mean of the two pooled estimators;
  * E[M] / (kappa E[S])                 -- Wald's identity implies >= 1 for every rule;
  * type I error of the exact two-sided Poisson test K | D ~ Poisson(kappa S), 5% level.

Rules (W0_b = N_b / lambda_b is the budget, N_b the observed counts):
  A1-fixed     frontier at W0_b, independent of the finds             (A1 holds)
  A1-random    frontier at U * W0_b, U ~ Uniform(0.5, 2)                (A1 holds)
  stop-at-find search to W0_b, then stop at the next find (delta = 0)   (scheme A)
  drought-g    search to W0_b, then stop once g mean gaps pass without a find
               (delta = max(W0_b - last find, g / lambda_b))           (A1 violated)

Usage: python stopping_rules.py [replications]   (default 20000; seed 20260924)
"""
import sys
import numpy as np
from scipy import stats

SEED = 20260924
R = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
KAPPA = np.exp(np.euler_gamma)
BASES = np.array([2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20])
N_OBS = np.array([51, 22, 18, 16, 9, 10, 12, 13, 12, 10, 9, 11, 7, 10, 8])
C = 1 / np.log(BASES)
LAM = KAPPA * C
B = len(BASES)


def events_fixed(rng, lam, w):
    n = rng.poisson(lam * w)
    if n == 0:
        return 0, None
    return n, rng.uniform(0, w, n).max()


def events_until(rng, lam, w0, gap):
    """Search to w0; afterwards stop at the next find (gap=0) or after a drought of
    length `gap` (gap>0). Returns (count, position of last find)."""
    t, n, last = 0.0, 0, None
    while True:
        e = rng.exponential(1 / lam)
        if gap == 0:
            t += e
            n += 1
            last = t
            if t >= w0:
                return n, last
            continue
        stop_at = max(w0, (last if last is not None else -np.inf) + gap)
        if last is not None and t + e > stop_at:
            return n, last
        t += e
        n += 1
        last = t


def one_rep(rng, rule):
    m, s = 0, 0.0
    for j in range(B):
        lam, w0 = LAM[j], N_OBS[j] / LAM[j]
        if rule == "A1-fixed":
            n, d = events_fixed(rng, lam, w0)
        elif rule == "A1-random":
            n, d = events_fixed(rng, lam, rng.uniform(0.5, 2.0) * w0)
        elif rule == "stop-at-find":
            n, d = events_until(rng, lam, w0, 0.0)
        else:
            n, d = events_until(rng, lam, w0, float(rule.split("-")[1]) / lam)
        if n == 0:
            return None  # a base without finds: the conditional analysis drops the replication
        m += n
        s += d * C[j]
    return m, s


def poisson_two_sided_p(k, mu):
    return min(1.0, 2 * min(stats.poisson.cdf(k, mu), stats.poisson.sf(k - 1, mu)))


def main():
    rng = np.random.default_rng(SEED)
    rules = ["A1-fixed", "A1-random", "stop-at-find",
             "drought-0.5", "drought-1", "drought-2"]
    print(f"[S13] R={R}, seed={SEED}")
    print(f"{'rule':14s} {'(M-1)/S':>9s} {'(M-B)/S':>9s} {'EM/kES':>8s} {'size':>7s}")
    for rule in rules:
        ms, ss = [], []
        for _ in range(R):
            out = one_rep(rng, rule)
            if out is not None:
                ms.append(out[0])
                ss.append(out[1])
        ms, ss = np.array(ms), np.array(ss)
        k = ms - B
        size = np.mean([poisson_two_sided_p(ki, KAPPA * si) <= 0.05 for ki, si in zip(k, ss)])
        print(f"{rule:14s} {np.mean((ms - 1) / ss) / KAPPA:9.4f} {np.mean(k / ss) / KAPPA:9.4f} "
              f"{ms.mean() / (KAPPA * ss.mean()):8.4f} {size:7.4f}")


if __name__ == "__main__":
    main()
