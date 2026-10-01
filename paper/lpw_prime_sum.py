"""[S26]-[S28] The second-order LPW term summed over primes instead of integrated.

The heuristic (probp) assigns a probability to every PRIME index p,

    Pr(R_b(p) prime) ~ e^gamma (ln p + c_b) / (p ln b),   p not dividing b-1,

so the expected number of events in a window (n_0, n_1) is a SUM over the primes of the
window. lpw_second_order.py replaces this sum by the integral of (kappa/ln b)(1 + c_b/t)
dt, i.e. by the prime density 1/ln n. The two differ by the constant of Mertens' first
theorem, sum_{p<=x} ln p / p = ln x + E + o(1), E = -1.3326, evaluated at the lower end of
the window: for n_0 = 2..19 the integral overstates the exposure by about one unit of t
per base, for n_0 > 100 the difference is negligible. Here the exposure is computed as
the exact prime sum (sieve up to 1.4e8, which covers every n_max):

    S_sum(c) = sum_b sum_{n_0 < p < n_1, p prime} (ln p + c_b) / (p ln b),

with the deterministic zeros p | b-1 (then p | R_b(p)) and the Aurifeuillean case p = s,
s = squarefree part of b, s = 1 mod 4 (b = 5, 13, 17, 20). By Proposition 3 and Remark 8,
K | D ~ Poisson(kappa S_sum(c)) exactly for fixed c, so the Garwood interval applies.

Tags printed:
  [S26] exact conditional inference with the prime-sum exposure, all truncation levels;
        propagation of the calibration error of c_b (as [S24])
  [S27] simulation check with a generator that works on the primes themselves
        (Bernoulli trials with the heuristic probability), compared with the
        continuous-intensity generator of [S25]
  [S28] section 7.3: intercept of the regression of the Mersenne counting function under
        the heuristic itself (prime-level generator, conditioned on 50 events below X_ver)

Usage: python lpw_prime_sum.py [replications]   (default 10000; seed 20260930)
"""
import math
import sys

import numpy as np

from lpw_second_order import CB_EMP, SEQ, build, load, garwood, p_two, S_of

KAPPA = math.exp(0.5772156649015329)
SEED = 20260930
SE_C = 0.045
X_VER = 81_648_221
# Wagstaff-type <ln a_b>, lpw_second_order.py [S14]
CB_W = {2: 1.2422, 3: 1.4994, 5: 1.4882, 6: 1.5264, 7: 1.3271, 10: 1.3815, 11: 1.2629,
        12: 1.4994, 13: 1.2977, 14: 1.3126, 15: 1.7362, 17: 1.2592, 18: 1.6461, 19: 1.2305,
        20: 1.4882}
AUR = {5: 5, 13: 13, 17: 17, 20: 5}
# first-principles prediction of c_b, cb_theory.py [S23]
CB_TH = {2: 0.840, 3: 0.462, 5: 1.012, 6: 0.498, 7: 0.508, 10: 0.520, 11: 0.530, 12: 0.462,
         13: 0.747, 14: 0.514, 15: 0.523, 17: 0.709, 18: 0.840, 19: 0.541, 20: 1.012}
CUTS = [None, 10, 100, 1000, 10000]


def sieve(limit):
    s = np.ones(limit // 2, dtype=bool)
    s[0] = False
    for i in range(3, int(limit ** 0.5) + 1, 2):
        if s[i // 2]:
            s[i * i // 2::i] = False
    return np.concatenate(([2], 2 * np.nonzero(s)[0] + 1)).astype(np.float64)


P = sieve(140_000_000)
P_MAX = float(P[-1])
LNP = np.log(P)
CW0 = np.concatenate(([0.0], np.cumsum(LNP / P)))
CW1 = np.concatenate(([0.0], np.cumsum(1.0 / P)))


def idx_of(t):
    """The integer index n = e^t. The window ends MUST be recovered exactly: exp(ln 5)
    evaluates to 4.999..., and a strict inequality p > n_min would then admit the origin
    itself (this bug affected b = 7, 13, 19 in the first draft of version 3.1)."""
    return float(round(math.exp(t)))


def forced_zero(b, p):
    return (b - 1) % p == 0 or AUR.get(b) == p


def expo_sum(b, c, n0, n1):
    """sum over primes n0 < p < n1 of (ln p + c)/p (deterministic zeros removed); above the
    sieve limit the sum is replaced by the integral, where the difference is ~1e-4."""
    hi = min(n1, P_MAX)
    i = int(np.searchsorted(P, n0, side="right"))
    j = int(np.searchsorted(P, hi, side="left"))
    a = (CW0[j] - CW0[i]) + c * (CW1[j] - CW1[i])
    for p in P[i:min(j, i + 20)]:
        if p > 50:
            break
        if forced_zero(b, int(p)):
            a -= (math.log(p) + c) / p
    if n1 > P_MAX:
        t0, t1 = math.log(max(n0, P_MAX)), math.log(n1)
        a += (t1 - t0) + c * math.log(t1 / t0)
    return a


def S_sum(bases, c):
    return sum(expo_sum(bd.b, c[bd.b], idx_of(bd.t0), idx_of(bd.tl)) / bd.lnb for bd in bases)


def cdict(v):
    return {b: v for b in SEQ}


def fit_c_sum(bases):
    """Profile likelihood of a common c with the prime-sum exposure: the conditional
    log-likelihood is sum_i ln(ln p_i + c) - kappa S_sum(c) + const, maximised over kappa.
    Admissible c keeps every weight positive: c > -ln(smallest prime of any window)."""
    from scipy import optimize, stats
    K = sum(bd.K for bd in bases)
    lp = np.concatenate([bd.inner for bd in bases])          # ln p_i of the interior events
    cmin = -min(math.log(float(P[np.searchsorted(P, idx_of(bd.t0), side="right")]))
                for bd in bases) + 1e-6

    def prof(c):
        return K * math.log(K / S_sum(bases, cdict(c))) - K + float(np.sum(np.log(lp + c)))

    r = optimize.minimize_scalar(lambda c: -prof(c), bounds=(cmin, 50), method="bounded")
    lmax = -r.fun
    f = lambda c: 2 * (lmax - prof(c)) - stats.chi2.ppf(0.95, 1)
    lo = optimize.brentq(f, cmin, r.x) if f(cmin) > 0 else cmin
    hi = optimize.brentq(f, r.x, 50) if f(50) > 0 else float("inf")
    pval = lambda c: float(stats.chi2.sf(2 * (lmax - prof(c)), 1))
    return r.x, lo, hi, pval(0.0), pval(math.log(2))


def s26(seqs, rng):
    print("[S26] exact conditional inference, exposure summed over the primes (sieve to 1.4e8)")
    print("  %-11s %3s %4s | %-15s | %-15s | %-15s | %-26s %6s | %-6s %6s"
          % ("trunc", "B", "K", "c=0 int/sum", "c=ln2 int/sum", "c=meas int/sum",
             "meas, sum: 95% Garwood", "p", "W sum", "p(W)"))
    rows = [(tr, 2) for tr in CUTS] + [(10000, 1)]
    for tr, me in rows:
        bs = build(seqs, truncate=tr, min_events=me)
        K = sum(bd.K for bd in bs)
        out = []
        for c in (cdict(0.0), cdict(math.log(2)), CB_EMP):
            out.append((K / S_of(bs, [c[bd.b] for bd in bs]), K / S_sum(bs, c)))
        Sm = S_sum(bs, CB_EMP)
        lo, hi = garwood(K, Sm)
        Sw = S_sum(bs, CB_W)
        name = ("none" if tr is None else ">%d" % tr) + (", N>=1" if me == 1 else "")
        print("  %-11s %3d %4d | %.4f / %.4f | %.4f / %.4f | %.4f / %.4f | [%.4f; %.4f] %6.3f | %.4f %6.3f"
              % (name, len(bs), K, out[0][0], out[0][1], out[1][0], out[1][1], out[2][0], out[2][1],
                 lo, hi, p_two(K, KAPPA * Sm), K / Sw, p_two(K, KAPPA * Sw)))
        if me == 2:
            print("  %-11s shift of the measured-c prime-sum estimate vs the constant rate "
                  "(c=0, integral): %+.1f%%" % ("", 100 * (out[2][1] / out[0][0] - 1)))
    k_int, k_sum = {}, {}
    for tr in (None, 10, 10000):
        bs = build(seqs, truncate=tr)
        K = sum(bd.K for bd in bs)
        k_int[tr] = K / S_of(bs, [CB_EMP[bd.b] for bd in bs])
        k_sum[tr] = K / S_sum(bs, CB_EMP)
    print("  integral vs prime sum (measured c): %+.1f%% (all), %+.1f%% (>10); prime-sum estimate "
          "from >10 to >10000: %+.1f%%" % (100 * (k_int[None] / k_sum[None] - 1),
                                           100 * (k_int[10] / k_sum[10] - 1),
                                           100 * (k_sum[10000] / k_sum[10] - 1)))
    bs = build(seqs)
    d = [expo_sum(bd.b, 0.0, idx_of(bd.t0), idx_of(bd.tl)) - bd.D for bd in bs]
    print("  low-end deficit of the prime sum vs D_b (c=0), all data: from %.2f to %.2f, mean %.2f"
          % (min(d), max(d), float(np.mean(d))))
    for tr in (None, 10):
        bs = build(seqs, truncate=tr)
        K = sum(bd.K for bd in bs)
        print("  first-principles c_b^th (cb_theory.py [S23]), %-5s kappa = %.4f"
              % ("all" if tr is None else ">%d" % tr, K / S_sum(bs, CB_TH)))
    for tr in (None, 10, 1000):
        chat, lo, hi, p0, p2 = fit_c_sum(build(seqs, truncate=tr))
        print("  free common c, prime-sum likelihood, %-6s c_hat = %.2f, 95%% profile CI [%.2f; %s]; "
              "LR p(c=0) = %.3f, p(c=ln2) = %.3f"
              % ("all" if tr is None else ">%d" % tr, chat, lo,
                 "%.2f" % hi if np.isfinite(hi) else "inf", p0, p2))
    # Propagation of the calibration error of c_b. Version 3.2: the combined interval is
    # the MIXTURE of Garwood intervals over the c_b draws -- the lower end solves
    # mean_i P(Gamma(K) <= L S_i) = 0.025 and the upper end mean_i P(Gamma(K+1) <= U S_i)
    # = 0.975 -- so that with no c_b error it reduces exactly to the Garwood interval and
    # can only be wider. Version 3.1 drew Gamma(K + 1/2) (a Jeffreys-type interval), which is
    # NARROWER than Garwood; it is still printed for comparison. The random draws are kept
    # exactly as in 3.1 so that [S27]-[S28] below are unchanged.
    from scipy import optimize, stats
    for tr in (None, 10):
        bs = build(seqs, truncate=tr)
        K = sum(bd.K for bd in bs)
        ks, Ss = [], []
        for _ in range(20000):
            c = {b: v + SE_C * rng.standard_normal() for b, v in CB_EMP.items()}
            Si = S_sum(bs, c)
            Ss.append(Si)
            ks.append(rng.gamma(K + 0.5) / Si)
        ks, Ss = np.array(ks), np.array(Ss)
        lo = optimize.brentq(lambda x: np.mean(stats.gamma.cdf(x * Ss, K)) - 0.025, 0.5, 4.0)
        hi = optimize.brentq(lambda x: np.mean(stats.gamma.cdf(x * Ss, K + 1)) - 0.975, 0.5, 4.0)
        g0 = garwood(K, S_sum(bs, CB_EMP))
        print("  c_b error (se %.3f), %-5s combined 95%% (Garwood mixture) [%.3f; %.3f]; "
              "Garwood at c fixed [%.3f; %.3f]; Jeffreys-type (version 3.1) [%.3f; %.3f]; "
              "SD of S from c alone %.4f (relative %.4f)"
              % (SE_C, "all" if tr is None else ">%d" % tr, lo, hi, g0[0], g0[1],
                 np.quantile(ks, 0.025), np.quantile(ks, 0.975), Ss.std(), Ss.std() / Ss.mean()))


class Gen:
    """Prime-level generator for one base: Bernoulli trials at the primes, probability
    min(1, kappa (ln p + c)/(p ln b)); exact below 2000, Poisson-thinned above (the
    probabilities there are below 1e-2), continuous above the sieve limit."""

    def __init__(self, b, c, n0):
        self.b, self.c, self.lnb = b, c, math.log(b)
        m = P > n0
        p = P[m]
        w = np.minimum(1.0, KAPPA * (np.log(p) + c) / (p * self.lnb))
        w[[forced_zero(b, int(x)) for x in p[:50]] + [False] * (p.size - min(50, p.size))] = 0.0
        small = p < 2000
        self.ps, self.ws = p[small], w[small]
        self.pl = p[~small]
        self.cl = np.cumsum(w[~small])

    def draw(self, rng, n_end):
        ev = [self.ps[(self.ps <= n_end) & (rng.random(self.ps.size) < self.ws)]]
        j = int(np.searchsorted(self.pl, min(n_end, P_MAX), side="right"))
        if j:
            lam = self.cl[j - 1]
            ev.append(self.pl[np.searchsorted(self.cl[:j], rng.random(rng.poisson(lam)) * lam)])
        if n_end > P_MAX:
            t0, t1 = math.log(P_MAX), math.log(n_end)
            lmax = KAPPA / self.lnb * (1 + max(self.c, 0) / t0)
            t = rng.uniform(t0, t1, rng.poisson(lmax * (t1 - t0)))
            t = t[rng.random(t.size) < (1 + self.c / t) / (1 + max(self.c, 0) / t0)]
            ev.append(np.exp(t))
        return np.unique(np.concatenate(ev))


def s27(seqs, rng, reps):
    print("\n[S27] simulation check with a PRIME-LEVEL generator (%d replications, kappa = e^gamma, "
          "c_b = measured; origin = real n_min, frontier one mean gap beyond the real last find)" % reps)
    real = {bd.b: bd for bd in build(seqs, min_events=1)}
    gens = {b: Gen(b, CB_EMP[b], idx_of(bd.t0)) for b, bd in real.items()}
    names = ("const (c=0)", "integral S(c)", "prime sum S(c)")
    est = {tr: [[], [], []] for tr in CUTS}
    cov = {tr: [0, 0, 0] for tr in CUTS}
    for _ in range(reps):
        sim = {}
        for b, bd in real.items():
            lam_end = KAPPA / bd.lnb * (1 + CB_EMP[b] / bd.tl)
            n_end = math.exp(bd.tl + rng.exponential(1 / lam_end))
            sim[b] = np.concatenate([[idx_of(bd.t0)], gens[b].draw(rng, n_end)])
        for tr in CUTS:
            bs = build(sim, truncate=tr)
            K = sum(bd.K for bd in bs)
            if K == 0:
                continue
            for i, S in enumerate((sum(bd.D / bd.lnb for bd in bs),
                                   S_of(bs, [CB_EMP[bd.b] for bd in bs]), S_sum(bs, CB_EMP))):
                est[tr][i].append(K / S)
                lo, hi = garwood(K, S)
                cov[tr][i] += int(lo <= KAPPA <= hi)
    print("  %-8s" % "trunc" + "".join(" %22s" % n for n in names) + "   (mean kappa_hat/kappa, coverage)")
    for tr in CUTS:
        n = len(est[tr][0])
        print("  %-8s" % ("none" if tr is None else ">%d" % tr)
              + "".join("       %.4f  %.3f   " % (np.mean(est[tr][i]) / KAPPA, cov[tr][i] / n)
                        for i in range(3)))


def s28(seqs, rng, want=5000):
    print("\n[S28] section 7.3: regression intercept under the heuristic itself (b = 2, prime-level "
          "generator, conditioned on 50 exponents below X_ver = %d)" % X_VER)
    t50 = np.log(np.array(sorted(seqs[2]))[:50])
    obs = float(np.polyfit(t50, np.arange(50), 1)[1])
    m = P <= X_VER
    p, lp = P[m], LNP[m]
    for name, cc in (("c_2 = %.3f (measured)" % CB_EMP[2], np.full(p.size, CB_EMP[2])),
                     ("Wagstaff a = 2 or 6 by p mod 4", np.where(p % 4 == 3, math.log(2), math.log(6)))):
        w = np.minimum(1.0, KAPPA * (lp + cc) / (p * math.log(2)))
        small = p < 2000
        ws, ts = w[small], lp[small]
        cum, tl = np.cumsum(w[~small]), lp[~small]
        lam = cum[-1]
        ics, tries = [], 0
        while len(ics) < want and tries < 1_000_000:
            tries += 1
            es = ts[rng.random(ws.size) < ws]
            k = rng.poisson(lam)
            if es.size + k != 50:
                continue
            tt = np.sort(np.concatenate((es, tl[np.searchsorted(cum, rng.random(k) * lam)])))
            if np.unique(tt).size < 50:
                continue
            ics.append(np.polyfit(tt, np.arange(50), 1)[1])
        ics = np.array(ics)
        print("  %-32s E[count below X_ver] = %.2f; intercept median %.3f, 5-95%% [%.2f; %.2f]; "
              "observed %.4f at quantile %.3f (%d replications)"
              % (name, w.sum(), np.median(ics), np.quantile(ics, 0.05), np.quantile(ics, 0.95),
                 obs, np.mean(ics <= obs), ics.size))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    rng = np.random.default_rng(SEED)
    seqs = load(SEQ)
    s26(seqs, rng)
    s27(seqs, rng, reps)
    s28(seqs, rng)


if __name__ == "__main__":
    main()
