"""[S24]-[S25] Uncertainty of c_b and a simulation check of the section-9 method.

[S24] Propagation of the calibration error of c_b (standard error ~0.045 per base,
      cb_empirical.py [S19]) into the estimate kappa = K / S(c). For each draw
      c_b ~ N(c_b_hat, se_b^2) the exact conditional inference is recomputed; the
      combined 95% interval is the envelope that mixes the Poisson (Garwood) and the
      c_b uncertainty: kappa = G / S(c), G ~ Gamma(K + 1/2, 1) (Jeffreys) with c drawn.
      Also reported: kappa with the first-principles prediction c_b^th of cb_theory.py.

[S25] Monte Carlo check of the method on synthetic data with KNOWN kappa = e^gamma and
      c_b = calibrated values. For each base the process lives on (t0, t0 + W] with the
      intensity (kappa / ln b)(1 + c_b / t), t0 = ln n_min of the real data and a frontier
      one mean gap beyond the real last find (an (A1) rule). The conditional analysis of
      the paper is applied with truncation n > 10^j (origin = first surviving event,
      N_b >= 2), with c = 0 (constant rate) and with the true c_b. Reported: mean
      kappa_hat / kappa and the coverage of the exact Garwood 95% interval.

Usage: python sim_second_order.py [replications]   (default 4000; seed 20260925)
"""
import math
import sys

import numpy as np
from scipy import stats

from lpw_second_order import CB_EMP, SEQ, build, load, garwood

KAPPA = math.exp(0.5772156649015329)
SEED = 20260925
SE_C = 0.045
# first-principles prediction, cb_theory.py [S23]
CB_TH = {2: 0.840, 3: 0.462, 5: 1.012, 6: 0.498, 7: 0.508, 10: 0.520, 11: 0.530, 12: 0.462,
         13: 0.747, 14: 0.514, 15: 0.523, 17: 0.709, 18: 0.840, 19: 0.541, 20: 1.012}


def S_of(bases, c):
    return sum((bd.D + c[bd.b] * bd.L) / bd.lnb for bd in bases)


def s24(seqs, rng, draws=20000):
    print("[S24] propagation of the calibration error of c_b (se = %.3f per base)" % SE_C)
    for tr in (None, 10):
        bs = build(seqs, truncate=tr)
        K = sum(bd.K for bd in bs)
        S0 = S_of(bs, CB_EMP)
        lo, hi = garwood(K, S0)
        ks = []
        for _ in range(draws):
            c = {b: v + SE_C * rng.standard_normal() for b, v in CB_EMP.items()}
            ks.append(rng.gamma(K + 0.5) / S_of(bs, c))
        ks = np.array(ks)
        sd_c = np.std([K / S_of(bs, {b: v + SE_C * rng.standard_normal()
                                     for b, v in CB_EMP.items()}) for _ in range(4000)])
        p_mix = 2 * min(np.mean(ks <= KAPPA), np.mean(ks >= KAPPA))
        kth = K / S_of(bs, CB_TH)
        print("  %-5s kappa = %.4f; Garwood (c fixed) [%.3f; %.3f]; SD from c alone %.4f; "
              "combined 95%% [%.3f; %.3f]; p = %.3f | with c_b^th: kappa = %.4f"
              % ("all" if tr is None else ">%d" % tr, K / S0, lo, hi, sd_c,
                 np.quantile(ks, 0.025), np.quantile(ks, 0.975), p_mix, kth))


def simulate_base(rng, lnb, c, t0, t_end):
    """Inhomogeneous Poisson on (t0, t_end] with rate (kappa/lnb)(1 + c/t), by thinning."""
    lam_max = KAPPA / lnb * (1 + max(c, 0) / t0)
    n = rng.poisson(lam_max * (t_end - t0))
    t = np.sort(rng.uniform(t0, t_end, n))
    keep = rng.random(n) < (1 + c / t) / (1 + max(c, 0) / t0)
    return t[keep]


def s25(seqs, rng, reps):
    print("\n[S25] simulation check of the section-9 method (%d replications, kappa = e^gamma, "
          "c_b = calibrated)" % reps)
    real = {bd.b: bd for bd in build(seqs, min_events=1)}
    cuts = [None, 10, 100, 1000, 10000]
    est0 = {tr: [] for tr in cuts}
    estc = {tr: [] for tr in cuts}
    cover0 = {tr: 0 for tr in cuts}
    coverc = {tr: 0 for tr in cuts}
    for _ in range(reps):
        sim = {}
        for b, bd in real.items():
            lam_end = KAPPA / bd.lnb * (1 + CB_EMP[b] / bd.tl)
            t_end = bd.tl + rng.exponential(1 / lam_end)
            ev = simulate_base(rng, bd.lnb, CB_EMP[b], bd.t0, t_end)
            sim[b] = np.exp(np.concatenate([[bd.t0], ev]))    # origin + events, as indices n
        for tr in cuts:
            bs = build(sim, truncate=tr)
            K = sum(bd.K for bd in bs)
            if K == 0:
                continue
            for c, est, cov in ((None, est0, cover0), (CB_EMP, estc, coverc)):
                S = sum(bd.D / bd.lnb for bd in bs) if c is None else S_of(bs, c)
                est[tr].append(K / S)
                lo, hi = garwood(K, S)
                cov[tr] += int(lo <= KAPPA <= hi)
    print("  %-8s %14s %10s %14s %10s" % ("trunc", "E[k0]/kappa", "cover(0)", "E[kc]/kappa", "cover(c)"))
    for tr in cuts:
        n = len(est0[tr])
        print("  %-8s %14.4f %10.3f %14.4f %10.3f"
              % ("none" if tr is None else ">%d" % tr, np.mean(est0[tr]) / KAPPA, cover0[tr] / n,
                 np.mean(estc[tr]) / KAPPA, coverc[tr] / n))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(SEED)
    seqs = load(SEQ)
    s24(seqs, rng)
    s25(seqs, rng, reps)


if __name__ == "__main__":
    main()
