"""[S29] Additional robustness checks requested in the review of version 3.

  (a) Is the rise of (M-1)/S under lower truncation (Table 6: +6.8% from "none" to
      n > 10^4) evidence for anything? Its probability under scheme A and scheme B,
      simulated on the real origins, N_b and bases, with the truncation rule of Table 6.
  (b) A frontier known by construction: every n_max exceeds T = 2e5, so under contiguous
      search the window (n_min, T] was fully searched for all 15 bases, the exposure is
      known and the full Poisson likelihood applies (no conditioning, no (A1) needed for
      the frontier position).
  (c) Truncation with the origin placed at the cut itself (the first surviving event is
      then an event, not the origin).
  (d) Deterministic origins: R_b(2) = b + 1 is prime exactly for b = 2, 6, 10, 12, 18, so
      n_min = 2 there is arithmetic, not chance. Origin moved to the second index.
  (e) Base 18: the endpoint-vs-OLS discrepancy with the origin moved past the first gap,
      and the size of that gap (largest spacing among 7, conditionally uniform).
  (f) Power thresholds of the two-sided exact test for a DEFICIT of the constant.
  (g) Figure 4: pointwise position of the largest excursion and a variance-standardized
      supremum test (the plain sup test of section 7.1(ii) is insensitive near the ends).
  (h) Table 4, first row, with 600 000 replications.
  (i) The documented-frontier check of (A1) evaluated at kappa_hat_B instead of e^gamma.
  (j) Monte Carlo standard errors of Table 2.

Usage: python robust_extra.py   (seed 20260931)
"""
import math

import numpy as np

import analysis as v1
from analysis_v2 import cond, power, first_crossing
from lpw_second_order import SEQ, build, load, garwood, p_two

KAPPA = math.exp(0.5772156649015329)
SEED = 20260931


def trend(seqs, rng, reps=20000):
    bs = build(seqs)
    t0 = [bd.t0 for bd in bs]
    lnb = [bd.lnb for bd in bs]
    N = [bd.K + 1 for bd in bs]
    lam = [KAPPA / x for x in lnb]
    obsA, obsB = 2.0990 / 1.9648, 1.7889 / 1.8381

    def pooled(evs, cut):
        M = S = B = 0
        for ev, o, lb in zip(evs, t0, lnb):
            if cut is not None:
                e = ev[ev > math.log(cut)]
                if len(e) - 1 < 2:
                    continue
                o, e = e[0], e[1:]
            else:
                e = ev
            M, S, B = M + len(e), S + (e[-1] - o) / lb, B + 1
        return M, S, B

    for scheme in ("A", "B"):
        hA = hB = n = 0
        for _ in range(reps):
            evs = []
            for o, lm, n_ in zip(t0, lam, N):
                if scheme == "A":
                    evs.append(o + np.cumsum(rng.exponential(1 / lm, n_)))
                else:
                    W = n_ / lm
                    evs.append(np.sort(o + W * rng.random(rng.poisson(lm * W))))
            if min(len(e) for e in evs) < 2:
                continue
            M0, S0, B0 = pooled(evs, None)
            M4, S4, B4 = pooled(evs, 10000)
            if B4 == 0:
                continue
            n += 1
            hA += ((M4 - 1) / S4) / ((M0 - 1) / S0) >= obsA
            hB += ((M4 - B4) / S4) / ((M0 - B0) / S0) <= obsB
        print("  (a) scheme %s: P((M-1)/S rises by >= %.1f%%) = %.3f; P((M-B)/S falls by >= %.1f%%) = %.3f"
              % (scheme, 100 * (obsA - 1), hA / n, 100 * (1 - obsB), hB / n))


def main():
    rng = np.random.default_rng(SEED)
    seqs = load(SEQ)
    print("[S29] additional robustness checks (review of version 3)")
    trend(seqs, rng)

    T = 2e5
    M = sum(int(np.sum((idx > idx[0]) & (idx <= T))) for idx in seqs.values())
    S = sum((math.log(T) - math.log(idx[0])) / math.log(b) for b, idx in seqs.items())
    lo, hi = garwood(M, S)
    print("  (b) common frontier T = 2e5 (below every n_max): M = %d, S = %.3f, M/S = %.4f, "
          "95%% [%.3f; %.3f], p = %.3f" % (M, S, M / S, lo, hi, p_two(M, KAPPA * S)))

    for cut in (10, 100, 1000, 10000):
        K = 0
        S = 0.0
        for b, idx in seqs.items():
            ev = idx[idx > cut]
            if len(ev):
                K += len(ev) - 1
                S += (math.log(ev[-1]) - math.log(cut)) / math.log(b)
        lo, hi = garwood(K, S)
        print("  (c) origin at the cut n = %5d: K = %d, (M-B)/S = %.4f, 95%% [%.3f; %.3f], p = %.3f"
              % (cut, K, K / S, lo, hi, p_two(K, KAPPA * S)))

    s2 = {b: (idx[1:] if idx[0] == 2 else idx) for b, idx in seqs.items()}
    bs = build(s2)
    K = sum(bd.K for bd in bs)
    S = sum(bd.D / bd.lnb for bd in bs)
    print("  (d) origins n_min = 2 (b = %s) moved to the second index: (M-B)/S = %.4f, (M-1)/S = %.4f, p = %.3f"
          % (",".join(str(b) for b, idx in sorted(seqs.items()) if idx[0] == 2),
             K / S, (K + len(bs) - 1) / S, p_two(K, KAPPA * S)))

    l18 = np.log(seqs[18])
    L18 = math.log(18)
    ge = (l18[-1] - l18[0]) / ((len(l18) - 1) * L18)
    go = np.polyfit(np.arange(len(l18)), l18, 1)[0] / L18
    ge2 = (l18[-1] - l18[1]) / ((len(l18) - 2) * L18)
    go2 = np.polyfit(np.arange(len(l18) - 1), l18[1:], 1)[0] / L18
    g = (l18[1] - l18[0]) / (l18[-1] - l18[0])
    k = len(l18) - 1          # spacings of the k-1 interior points, uniform given D
    pgap = sum((-1) ** (j + 1) * math.comb(k, j) * max(0.0, 1 - j * g) ** (k - 1) for j in range(1, k + 1))
    print("  (e) b = 18: G_end %.4f, G_OLS %.4f (%+.1f%%); origin at %d: G_end %.4f, G_OLS %.4f (%+.1f%%); "
          "first gap = %.1f%% of the window, P(largest of %d spacings >= it) = %.4f"
          % (ge, go, 100 * (go / ge - 1), seqs[18][1], ge2, go2, 100 * (go2 / ge2 - 1), 100 * g, k, pgap))

    b1 = v1.build(v1.load_sequences())
    K, S = cond(b1)
    S2 = [bd for bd in b1 if bd.b == 2][0].expo
    g = np.arange(1.0, 0.3, -0.0005)
    print("  (f) 80%% power, two-sided exact test, deficit: Mersenne only -%.1f%%, pooled -%.1f%%"
          % (100 * (1 - first_crossing(lambda r: power(S2, r), g)),
             100 * (1 - first_crossing(lambda r: power(S, r), g))))

    # (g) Figure 4 and section 7.1(ii), under the SAME null law as analysis_v2.s9: N-1 = 49
    #     uniform points on (t0, ln X_ver], lambda from the last of them. Residuals are taken
    #     at the events, r_i = i - lambda (t_i - t0), i = 0..N-1, and standardized by the
    #     binomial bridge variance, z_i = |r_i| / sqrt((N-1) u_i (1 - u_i)),
    #     u_i = (t_i - t0) / (t_{N-1} - t0), over the interior events 0 < u_i < 1.
    #     The pointwise quantile is that of the step function r(t) at the observed t_i.
    t = np.log(np.array(sorted(seqs[2]))[:50])
    t0, N, T = t[0], 50, math.log(81_648_221)
    lam = (N - 1) / (t[-1] - t0)
    r = np.arange(N) - lam * (t - t0)
    i = int(np.argmax(np.abs(r)))
    u = (t - t0) / (t[-1] - t0)
    z = lambda rv, uv: np.abs(rv) / np.sqrt((N - 1) * uv * (1 - uv))
    zo = float(z(r[1:-1], u[1:-1]).max())
    reps = 40000
    E = np.sort(rng.uniform(t0, T, (reps, N - 1)), axis=1)      # simulated events 1..N-1
    lam_s = (N - 1) / (E[:, -1] - t0)
    rs = np.arange(1, N) - lam_s[:, None] * (E - t0)           # residuals at the events
    us = (E - t0) / (E[:, -1:] - t0)
    zs = z(rs[:, :-1], us[:, :-1]).max(axis=1)
    ri = (E <= t[i]).sum(axis=1) - lam_s * (t[i] - t0)        # step r(t) at the observed t_i
    print("  (g) figure 4: largest excursion r = %.3f at t = %.2f; pointwise 5%% quantile of the step "
          "r(t) there %.2f, median %.2f, P(r <= observed) = %.4f; standardized sup %.2f, p = %.3f"
          % (r[i], t[i], np.quantile(ri, 0.05), np.median(ri), np.mean(ri <= r[i]), zo,
             np.mean(zs >= zo)))

    # (h) Table 4, first row, with many more replications than [S10]
    from analysis_v2 import simulate_B
    from scipy import stats
    reps = 600_000
    Ms, Ss = simulate_B(b1, reps, np.random.default_rng(SEED + 8), lambda bd, l: bd.N / l)
    x = KAPPA * Ss
    rej = (x < stats.gamma.ppf(0.025, Ms)) | (x > stats.gamma.ppf(0.975, Ms))
    print("  (h) Table 4, rule 'E[#] = observed', %d replications: type I error of the Gamma pivot "
          "%.4f +- %.4f" % (reps, rej.mean(), math.sqrt(rej.mean() * (1 - rej.mean()) / reps)))

    # (i) documented frontiers evaluated at kappa_hat_B instead of e^gamma
    from frontiers import FRONTIERS
    kB = (sum(bd.K for bd in build(seqs))) / sum(bd.D / bd.lnb for bd in build(seqs))
    for kind_b2 in ("verified", "tested once"):
        xs = [math.log(L / nmax) / math.log(b) for b, rows in FRONTIERS.items()
              for nmax, L, kind in rows if b != 2 or kind == kind_b2]
        s0, s1 = KAPPA * sum(xs), kB * sum(xs)
        print("  (i) frontiers (b=2 %s): lower tail at e^gamma %.3f, at kappa_hat_B = %.4f (x %.4f) %.3f"
              % (kind_b2, stats.gamma.cdf(s0, len(xs)), kB, kB / KAPPA, stats.gamma.cdf(s1, len(xs))))

    # (j) Monte Carlo accuracy of Table 2 (60 000 replications), from the exact variances
    R = 60000
    for n in (7, 51):
        sd_l = n / ((n - 1) * math.sqrt(n - 2))       # SD of lambda_hat / lambda, D ~ Gamma(n)
        sd_g = 1 / math.sqrt(n)                       # SD of G_hat * lambda
        print("  (j) Table 2, N = %d: MC standard error %.4f for E[lambda^]/lambda, %.4f for E[G^]lambda"
              % (n, sd_l / math.sqrt(R), sd_g / math.sqrt(R)))


if __name__ == "__main__":
    main()
