"""[S18] Documented search frontiers: an empirical check of (A1) and a hybrid estimator.

Only four bases have a stated, attributable search limit above the last known prime
(checked 2026-09-24; quotes in CHANGES_v3.md):
  b=2   GIMPS milestones: all exponents below 141308443 tested at least once,
        below 81648221 verified (snapshot of 2026-08-27 used in the paper;
        the 2026-09-24 report gives 141566917 and 83200483);
  b=10  OEIS A004023, S. Batalov, 2021-07-01: "Search limit is 10800000, currently.";
  b=22  OEIS A127997, R. Price, 2012-02-25: "No other terms < 100000.";
  b=26  OEIS A127999, R. Price, 2012-03-18: "No other terms less than 100000."

Under (A1) the gap delta_b = ln L_b - ln n_max,b satisfies lambda_b * delta_b ~ Exp(1)
(truncated at the window width, negligible here), independently across bases, so the
sum is Gamma(k, 1). Values near 0 point to "stop at a find" (m < 1), large values to
"stop after a drought" (m > 1), cf. Remark 9 and Table 5.

Hybrid estimator: for a base with a documented frontier the full likelihood applies,
N_b ~ Poisson(lambda_b W_b) with ALL events after t_0 and the full window; for the rest
the conditional law K_b | D_b ~ Poisson(lambda_b D_b). The sum is again exactly Poisson.
"""
import math

import numpy as np
from scipy import stats

KAPPA = math.exp(0.5772156649015329)

# b: (n_min, n_max, number of known primes, documented frontier L_b)
MAIN = {2: (2, 136279841, 52, 141308443), 3: (3, 8530117, 23, None), 5: (3, 3300593, 19, None),
        6: (2, 3360347, 17, None), 7: (5, 1264699, 10, None), 10: (2, 8177207, 11, 10800000),
        11: (17, 1868983, 13, None), 12: (2, 769543, 14, None), 13: (5, 1503503, 13, None),
        14: (3, 1724417, 11, None), 15: (3, 639833, 10, None), 17: (3, 1990523, 12, None),
        18: (2, 1270141, 8, None), 19: (19, 209359, 11, None), 20: (3, 984349, 9, None)}
FRONTIERS = {  # b: (n_max below the frontier, frontier, kind)
    2: [(77232917, 81648221, "verified"), (136279841, 141308443, "tested once")],
    10: [(8177207, 10800000, "searched")],
    22: [(27823, 100000, "no other terms")],
    26: [(26717, 100000, "no other terms")],
}


def garwood(k, s, a=0.05):
    return stats.chi2.ppf(a / 2, 2 * k) / (2 * s), stats.chi2.ppf(1 - a / 2, 2 * (k + 1)) / (2 * s)


def p_two(k, mu):
    return min(1.0, 2 * min(stats.poisson.cdf(k, mu), stats.poisson.sf(k - 1, mu)))


def main():
    print("[S18] documented frontiers: lambda_b * delta_b (Exp(1) under A1)")
    for kind_b2 in ("verified", "tested once"):
        xs = []
        for b, rows in FRONTIERS.items():
            for nmax, L, kind in rows:
                if b == 2 and kind != kind_b2:
                    continue
                x = KAPPA / math.log(b) * math.log(L / nmax)
                xs.append(x)
                print("  b=%2d  n_max=%10d  L=%10d  (%s)  lambda*delta=%.3f  Pr(Exp(1)<=x)=%.3f"
                      % (b, nmax, L, kind, x, 1 - math.exp(-x)))
        s = sum(xs)
        print("  b=2 frontier = %-11s: sum=%.3f over k=%d; mean=%.3f; Pr(Gamma(k,1) <= sum)=%.3f"
              % (kind_b2, s, len(xs), s / len(xs), stats.gamma.cdf(s, len(xs))))

    print("\n[S18] hybrid estimator, main set (full likelihood for b=2 [tested once], b=10)")
    K = 0
    S = 0.0
    for b, (nmin, nmax, P, L) in MAIN.items():
        if L is not None:
            K += P - 1
            S += math.log(L / nmin) / math.log(b)
        else:
            K += P - 2
            S += math.log(nmax / nmin) / math.log(b)
    lo, hi = garwood(K, S)
    print("  K=%d S=%.4f kappa=%.4f 95%% [%.4f; %.4f] p=%.3f" % (K, S, K / S, lo, hi, p_two(K, KAPPA * S)))

    print("\n[S18] base 5 with the 2024 PRPs 4939471, 5154509 (range above 3300593 NOT certified)")
    K5, S5 = 203 + 2, 110.4423 + math.log(5154509 / 3300593) / math.log(5)
    lo, hi = garwood(K5, S5)
    print("  K=%d S=%.4f kappa=%.4f 95%% [%.4f; %.4f] p=%.3f  (sensitivity only)"
          % (K5, S5, K5 / S5, lo, hi, p_two(K5, KAPPA * S5)))


if __name__ == "__main__":
    main()
