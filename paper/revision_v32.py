"""[S30] Version 3.2: computations requested by the editor's decision and the three reviews
of version 3.1 (letter of 01.10.2026). Every number is printed with its meaning; the paper
quotes them under tag [S30].

  (a) m-family: (M - B m)/S for an unknown mean number m of "missing" events per base
      (Remark 9). For every stopping rule E[M] = kappa E[S] + B m (Wald), and the Poisson
      test of K_m = M - B m keeps its level under every rule of Table 3 (stopping_rules.py,
      column size_m). Range of m for which LPW is not rejected, with c = 0 and with the
      calibrated second-order exposure.
  (b) Sensitivity to gaps in the search: how many finds would have to be missing inside
      the windows for LPW to be rejected from above.
  (c) The only documented frontier that is synchronous with the data snapshot (b = 2):
      one-sided tails of lambda*delta ~ Exp(1).
  (d) Poisson versus Poisson-binomial: under the heuristic (probp) the events are Bernoulli
      trials at the primes, so Var(K | D) = sum w(1-w) < sum w; ratio for the main data.
  (e) Covariance between bases through shared divisors 2p+1 (Remark 4).
  (f) Selection of bases by the number of finds: bias of (M-B)/S if a base were admitted
      only with N_b >= 7 (Remark 1).
  (g) "Regression through the origin": bias of the slope through the point (0, 0).
  (h) Exponentially small deficit (6) at the lower 95% bound of mu for N = 7.
  (i) Section 6: number of conditional events and of finds at kappa = 1.1 e^gamma.
  (j) Bayes factors of Table 8 at n = 203 and n = 15.
  (k) Section 7.3: constant C_0 with the probabilities (probp) capped at 1.
  (l) Section 7: rho_1 by the sample ACF, Newey-West with the small-sample factor.
  (m) Model (23): parametric-bootstrap calibration of LR = 2.03 under the conditional law.
  (n) Truncation profile (section 5.4) for the strictly consistent selection N_b >= 1.
  (o) Homogeneity under scheme A: exact Dirichlet null.
  (p) Derived percentages quoted in the text.
  (q) Section 4.6: bias of kappa* (weights N_b - 2) under scheme B for two frontier rules.

Usage: python revision_v32.py   (seed 20261001; about 10 minutes)
"""
import math

import numpy as np
from scipy import stats

import analysis as v1
from lpw_second_order import CB_EMP, SEQ, build, load, garwood, p_two
from lpw_prime_sum import P, S_sum, expo_sum, idx_of, forced_zero

KAPPA = math.exp(0.5772156649015329)
SEED = 20261001
X_VER = 81_648_221
X_FT = 141_308_443


def p_two_int(k, mu):
    return p_two(int(round(k)), mu)


def accept_range(mu):
    """Integer K for which the two-sided exact Poisson test at 5% does not reject."""
    ks = np.arange(int(mu - 8 * math.sqrt(mu)), int(mu + 8 * math.sqrt(mu)))
    ok = [k for k in ks if p_two(int(k), mu) > 0.05]
    return min(ok), max(ok)


def part_a(seqs):
    bs = build(seqs)
    M = sum(bd.K + 1 for bd in bs)
    B = len(bs)
    S = sum(bd.D / bd.lnb for bd in bs)
    Ss = S_sum(bs, CB_EMP)
    print("  (a) m-family (M - B m)/S, M = %d, B = %d, S = %.4f; second order: S_sum = %.4f" % (M, B, S, Ss))
    print("      m   (M-Bm)/S  excess%%  95%% Garwood        p    |  K/S_sum   p")
    for m in (0, 0.4, 1, 1.4, 2, 3):
        k = M - B * m
        lo, hi = garwood(int(round(k)), S)
        print("    %4.1f   %.4f  %+6.1f  [%.3f; %.3f]  %.3f |  %.4f  %.3f"
              % (m, k / S, 100 * (k / S / KAPPA - 1), lo, hi, p_two_int(k, KAPPA * S),
                 k / Ss, p_two_int(k, KAPPA * Ss)))
    print("      m at which (M-Bm)/S = e^gamma: %.2f; with the second-order exposure: %.2f"
          % ((M - KAPPA * S) / B, (M - KAPPA * Ss) / B))
    for name, s in (("c = 0", S), ("calibrated c_b, prime sum", Ss)):
        klo, khi = accept_range(KAPPA * s)
        print("      %s: LPW not rejected for K in [%d; %d], i.e. for m in [%.2f; %.2f]"
              % (name, klo, khi, (M - khi) / B, (M - klo) / B))
    return bs, M, B, S, Ss


def part_b(bs, M, B, S, Ss):
    K = M - B
    for name, s in (("c = 0", S), ("calibrated c_b, prime sum", Ss)):
        k = K
        while garwood(k, s)[0] <= KAPPA:
            k += 1
        print("  (b) %s: LPW rejected from above only if at least %d finds (%.1f per base) are "
              "missing inside the windows (K = %d)" % (name, k - K, (k - K) / B, k))


def part_c():
    lam2 = KAPPA / math.log(2)
    for name, L, nmax in (("X_ver", X_VER, 77_232_917), ("X_ft", X_FT, 136_279_841)):
        x = lam2 * math.log(L / nmax)
        print("  (c) b = 2, %s: lambda*delta = %.3f; P(Exp(1) <= x) = %.3f, P(Exp(1) >= x) = %.3f"
              % (name, x, 1 - math.exp(-x), math.exp(-x)))


def weights(b, c, n0, n1):
    lo, hi = np.searchsorted(P, n0, side="right"), np.searchsorted(P, n1, side="left")
    p = P[lo:hi]
    w = np.minimum(1.0, KAPPA * (np.log(p) + c) / (p * math.log(b)))
    for j in range(min(30, p.size)):
        if forced_zero(b, int(p[j])):
            w[j] = 0.0
    return p, w


def part_d(seqs):
    for tr in (None, 10):
        bs = build(seqs, truncate=tr)
        ew = vw = 0.0
        for bd in bs:
            _, w = weights(bd.b, CB_EMP[bd.b], idx_of(bd.t0), idx_of(bd.tl))
            ew += w.sum()
            vw += (w * (1 - w)).sum()
        print("  (d) %-5s Bernoulli (Poisson-binomial) model: E[K|D] = %.1f, Var/E = %.3f, "
              "SD overstated by the Poisson law by %.1f%%"
              % ("all" if tr is None else ">%d" % tr, ew, vw / ew, 100 * (1 / math.sqrt(vw / ew) - 1)))


def sqfree(b):
    s, d = 1, 2
    while d * d <= b:
        while b % (d * d) == 0:
            b //= d * d
        d += 1
    return b


def part_e(seqs):
    """X_b(p) = 0 when q = 2p+1 is prime and (b/q) = 1 (forced divisor, k = 1); otherwise a
    Bernoulli trial with probability w_b/(1 - rho_b). rho_b = rho_SG/2 with the Hardy-
    Littlewood density rho_SG = 2 C_2 / ln(2p) of Sophie Germain primes. For a pair with
    the same squarefree part the forced events coincide (P(A and A') = rho_SG/2), otherwise
    they are independent (rho_SG/4). Cov = w w' [P(notA, notA')/((1-rho)(1-rho')) - 1]."""
    bs = build(seqs)
    win = {bd.b: (idx_of(bd.t0), idx_of(bd.tl)) for bd in bs}
    lo_all = 10
    wts = {}
    for bd in bs:
        p, w = weights(bd.b, CB_EMP[bd.b], max(win[bd.b][0], lo_all), win[bd.b][1])
        wts[bd.b] = (p, w)
    var = sum((w * (1 - w)).sum() for p, w in wts.values())
    ew = sum(w.sum() for p, w in wts.values())
    cov_same = cov_all = 0.0
    same_pairs = []
    blist = sorted(wts)
    for i, b in enumerate(blist):
        for b2 in blist[i + 1:]:
            p1, w1 = wts[b]
            p2, w2 = wts[b2]
            common, i1, i2 = np.intersect1d(p1, p2, return_indices=True)
            if common.size == 0:
                continue
            rho = 1.3203 / np.log(2 * common)
            if sqfree(b) == sqfree(b2):
                f = 1 / (1 - rho / 2) - 1
            else:
                f = (1 - rho + rho / 4) / (1 - rho / 2) ** 2 - 1
            c = float((w1[i1] * w2[i2] * f).sum())
            cov_all += c
            if sqfree(b) == sqfree(b2):
                cov_same += c
                same_pairs.append((b, b2, c))
    print("  (e) covariance through shared divisors 2p+1, primes p > %d: pairs with the same "
          "squarefree part %s; all 105 pairs: sum Cov = %.3f against sum Var = %.1f; "
          "Var(K) inflated by a factor %.3f; E[K] = %.1f, Var(K)/E[K] = %.3f without and %.3f with "
          "the covariance"
          % (lo_all, ", ".join("%d/%d: %.3f" % x for x in same_pairs), cov_all, var,
             1 + 2 * cov_all / var, ew, var / ew, (var + 2 * cov_all) / ew))


def part_f(seqs, rng, reps=20000):
    bs = build(seqs)
    lam = np.array([KAPPA / bd.lnb for bd in bs])
    lnb = np.array([bd.lnb for bd in bs])
    Nobs = np.array([bd.K + 1 for bd in bs])
    W = Nobs / lam
    est = {1: [], 7: []}
    for _ in range(reps):
        n = rng.poisson(lam * W)
        d = np.array([W[j] * rng.random() ** (1 / n[j]) if n[j] > 0 else 0.0 for j in range(len(n))])
        for cut in (1, 7):
            keep = n >= cut
            if keep.sum() == 0:
                continue
            est[cut].append((n[keep].sum() - keep.sum()) / (d[keep] / lnb[keep]).sum())
    print("  (f) selection by the number of finds (scheme B, frontier W0 = N_b/lambda_b): "
          "E[(M-B)/S]/kappa = %.4f with N_b >= 1, %.4f if a base were admitted only with N_b >= 7 "
          "(bias %+.1f%%)" % (np.mean(est[1]) / KAPPA, np.mean(est[7]) / KAPPA,
                              100 * (np.mean(est[7]) / KAPPA - 1)))


def part_g(seqs):
    bs = build(seqs)
    out = []
    for bd in bs:
        N = bd.K + 1
        lam = KAPPA / bd.lnb
        out.append((3 * bd.t0 * lam / (2 * N + 1), bd.b))
    out.sort()
    b2 = [x for x in out if x[1] == 2][0]
    print("  (g) slope through the point (0, 0) of the (k, ln n_k) plane: relative bias "
          "3 t0 lambda/(2N+1) from %.1f%% (b=%d) to %.1f%% (b=%d) and %.1f%% (b=%d); b=2: %.1f%%"
          % (100 * out[0][0], out[0][1], 100 * out[-2][0], out[-2][1], 100 * out[-1][0],
             out[-1][1], 100 * b2[0]))


def part_h():
    mu = stats.chi2.ppf(0.05, 14) / 2
    print("  (h) N = 7: one-sided lower 95%% bound of mu = %.2f; deficit (1+mu)e^-mu there = %.1f%%"
          % (mu, 100 * (1 + mu) * math.exp(-mu)))


def part_i():
    S10 = 470.15
    K = 1.1 * KAPPA * S10
    print("  (i) section 6: at kappa = 1.1 e^gamma and S = %.2f the expected conditional count is "
          "K = %.1f and the number of finds M = K + 15 = %.1f" % (S10, K, K + 15))


def part_j():
    lnL = {"I": (-35.753, 1), "II": (-35.162, 3), "III": (-35.458, 2), "IV": (-34.797, 4)}
    for n in (203, 15):
        bic = {k: -2 * l + p * math.log(n) for k, (l, p) in lnL.items()}
        print("  (j) Table 8, BIC at n = %d: Bayes factor of model I against II, III, IV: %s"
              % (n, ", ".join("%.1f" % math.exp((bic[k] - bic["I"]) / 2) for k in ("II", "III", "IV"))))


def part_k():
    c2 = 0.96
    x = 1e8
    p = P[P <= x]
    w = KAPPA * (np.log(p) + c2) / (p * math.log(2))
    main = KAPPA / math.log(2) * (math.log(x) + c2 * math.log(math.log(x)))
    print("  (k) section 7.3, c_2 = %.2f, x = 1e8: C_0 = %.3f with the raw probabilities, "
          "%.3f with the probabilities capped at 1" % (c2, w.sum() - main, np.minimum(w, 1).sum() - main))


def part_l(seqs):
    exps = np.array(sorted(seqs[2]), dtype=float)
    exps = exps[exps < X_VER]
    N = len(exps)
    t = np.log(exps)
    y = np.arange(N, dtype=float)
    X = np.column_stack([np.ones(N), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    rho_cc = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    rho_acf = float((r[:-1] @ r[1:]) / (r @ r))
    nw = v1.newey_west_se(X, r, lags=8)[0]
    print("  (l) section 7: rho_1 = %.3f as the correlation of adjacent residuals, %.3f as the "
          "sample ACF; Newey-West SE (Bartlett, 8 lags) %.4f without and %.4f with the factor "
          "n/(n-k) = %d/%d" % (rho_cc, rho_acf, nw, nw * math.sqrt(N / (N - 2)), N, N - 2))


def part_m(seqs, rng, reps=10000):
    """LR of model (23) against c1 = c2 = 0 under the conditional likelihood (14), kappa
    profiled out in closed form; integrals by 200-point Gauss-Legendre."""
    from scipy import optimize
    xg, wg = np.polynomial.legendre.leggauss(200)
    bs = build(seqs)
    lnb = np.array([bd.lnb for bd in bs])
    t0 = np.array([bd.t0 for bd in bs])
    tl = np.array([bd.tl for bd in bs])
    nodes = [0.5 * (b - a) * xg + 0.5 * (b + a) for a, b in zip(t0, tl)]
    wts = [0.5 * (b - a) * wg for a, b in zip(t0, tl)]

    def prof(th, inner):
        c1, c2 = th
        K = sum(len(x) for x in inner)
        A = sum(float(w @ np.exp(c1 / x + c2 / x ** 2)) / l for x, w, l in zip(nodes, wts, lnb))
        s = sum(float(np.sum(c1 / x + c2 / x ** 2)) for x in inner)
        return K * math.log(K / A) - K + s

    def lr(inner):
        f = lambda th: -prof(th, inner)
        best = min((optimize.minimize(f, x0, method="Nelder-Mead",
                                      options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
                    for x0 in ([0.0, 0.0], [0.75, -1.44], [2.0, -3.0])), key=lambda r: r.fun)
        return 2 * (-best.fun - prof([0.0, 0.0], inner)), best.x

    obs, th = lr([bd.inner for bd in bs])
    K = sum(bd.K for bd in bs)
    S = sum(bd.D / bd.lnb for bd in bs)
    k0 = K / S
    sims = []
    for _ in range(reps):
        inner = [np.sort(rng.uniform(a, b, rng.poisson(k0 * (b - a) / l))) for a, b, l in zip(t0, tl, lnb)]
        if sum(len(x) for x in inner) == 0:
            continue
        sims.append(lr(inner)[0])
    sims = np.array(sims)
    print("  (m) model (23): LR = %.3f (c1 = %.2f, c2 = %.2f), chi2_2 p = %.3f; parametric "
          "bootstrap under the conditional null (%d replications): p = %.3f, 95%% quantile of LR "
          "%.2f (chi2_2: %.2f)" % (obs, th[0], th[1], stats.chi2.sf(obs, 2), len(sims),
                                   np.mean(sims >= obs), np.quantile(sims, 0.95), stats.chi2.ppf(0.95, 2)))


def part_n(seqs, rng, reps=20000):
    bs = build(seqs)
    t0 = [bd.t0 for bd in bs]
    lnb = [bd.lnb for bd in bs]
    N = [bd.K + 1 for bd in bs]
    lam = [KAPPA / x for x in lnb]
    obsA, obsB = 2.0848 / 1.9648, 1.7569 / 1.8381
    obsC = 1.7569 / 1.8712          # from n > 10 to n > 10^4 (strictly consistent rows)

    def pooled(evs, cut):
        M = S = B = 0
        for ev, o, lb in zip(evs, t0, lnb):
            if cut is not None:
                e = ev[ev > math.log(cut)]
                if len(e) - 1 < 1:
                    continue
                o, e = e[0], e[1:]
            else:
                e = ev
            M, S, B = M + len(e), S + (e[-1] - o) / lb, B + 1
        return M, S, B

    for scheme in ("A", "B"):
        hA = hB = hC = n = 0
        for _ in range(reps):
            evs = []
            for o, lm, n_ in zip(t0, lam, N):
                if scheme == "A":
                    evs.append(o + np.cumsum(rng.exponential(1 / lm, n_)))
                else:
                    W = n_ / lm
                    evs.append(np.sort(o + W * rng.random(rng.poisson(lm * W))))
            if min(len(e) for e in evs) < 1:
                continue
            M0, S0, B0 = pooled(evs, None)
            M4, S4, B4 = pooled(evs, 10000)
            M1, S1, B1 = pooled(evs, 10)
            if B4 == 0 or B1 == 0:
                continue
            n += 1
            hA += ((M4 - 1) / S4) / ((M0 - 1) / S0) >= obsA
            hB += ((M4 - B4) / S4) / ((M0 - B0) / S0) <= obsB
            hC += ((M4 - B4) / S4) / ((M1 - B1) / S1) <= obsC
        print("  (n) selection N_b >= 1, frontier W0 = N_b/lambda_b, scheme %s: P((M-1)/S rises "
              "by >= %.1f%%) = %.3f; P((M-B)/S falls by >= %.1f%%) = %.3f; P((M-B)/S falls from "
              "n > 10 to n > 10^4 by >= %.1f%%) = %.3f"
              % (scheme, 100 * (obsA - 1), hA / n, 100 * (1 - obsB), hB / n, 100 * (1 - obsC), hC / n))


def part_o(seqs, rng, reps=400_000):
    bs = build(seqs)
    N = np.array([bd.K + 1 for bd in bs], dtype=float)
    s = np.array([bd.D / bd.lnb for bd in bs])
    M, S = N.sum(), s.sum()
    lam_obs = 2 * float(np.sum(N * np.log(N * S / (M * s))))
    g = rng.gamma(N, size=(reps, len(N)))
    sim = 2 * np.sum(N * np.log(N * g.sum(axis=1, keepdims=True) / (M * g)), axis=1)
    print("  (o) homogeneity under scheme A (likelihood (8)): Lambda_A = %.3f; exact Dirichlet "
          "null p = %.3f (%d replications), chi2_14 p = %.3f"
          % (lam_obs, np.mean(sim >= lam_obs), reps, stats.chi2.sf(lam_obs, 14)))


def part_p():
    print("  (p) derived: (M-1)/S rises from 1.9648 to 2.0848 (N_b >= 1) by %+.1f%%; (M-B)/S from "
          "n > 10 to n > 10^4: c = 0 %+.1f%% (N_b >= 1), %+.1f%% (N_b >= 2); calibrated c_b, prime "
          "sum: %+.1f%% (N_b >= 1), %+.1f%% (N_b >= 2)"
          % (100 * (2.0848 / 1.9648 - 1), 100 * (1.7569 / 1.8712 - 1), 100 * (1.7889 / 1.8712 - 1),
             100 * (1.6625 / 1.7767 - 1), 100 * (1.6925 / 1.7767 - 1)))


def part_q(seqs, rng, reps=400_000):
    bs = build(seqs)
    for name, Wf in (("W = N_b/lambda_b", lambda bd, lam: (bd.K + 1) / lam),
                     ("W = 2 D_b", lambda bd, lam: 2 * bd.D)):
        num = np.zeros(reps)
        den = np.zeros(reps)
        for bd in bs:
            lam = KAPPA / bd.lnb
            W = Wf(bd, lam)
            n = rng.poisson(lam * W, size=reps)
            bad = n < 1
            while np.any(bad):
                n[bad] = rng.poisson(lam * W, size=int(bad.sum()))
                bad = n < 1
            D = W * rng.random(reps) ** (1.0 / n)
            num += np.maximum(n - 2, 0) * (n - 1) * bd.lnb / D
            den += np.maximum(n - 2, 0)
        star = num / den
        se = 100 * np.std(star) / KAPPA / math.sqrt(reps)
        print("  (q) kappa* (weights N_b - 2), scheme B, frontier %s: bias %+.2f%% +- %.2f%% (%d replications)"
              % (name, 100 * (np.mean(star) / KAPPA - 1), se, reps))


def main():
    rng = np.random.default_rng(SEED)
    seqs = load(SEQ)
    print("[S30] version 3.2: computations for the editor's decision of 01.10.2026")
    bs, M, B, S, Ss = part_a(seqs)
    part_b(bs, M, B, S, Ss)
    part_c()
    part_d(seqs)
    part_e(seqs)
    part_f(seqs, rng)
    part_g(seqs)
    part_h()
    part_i()
    part_j()
    part_k()
    part_l(seqs)
    part_n(seqs, rng)
    part_o(seqs, rng)
    part_p()
    # (q) and (m) use their own streams, so that their results do not depend on the
    # order of the other parts
    part_q(seqs, np.random.default_rng(SEED + 17))
    part_m(seqs, np.random.default_rng(SEED + 13))


if __name__ == "__main__":
    main()
