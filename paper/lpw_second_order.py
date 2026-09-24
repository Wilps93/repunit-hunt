"""[S14]-[S17] The second-order term of the LPW heuristic and the conditional likelihood.

The LPW heuristic gives Pr(R_b(p) prime) ~ e^gamma * ln(a p) / (p ln b), because every
prime divisor q of R_b(p) (p prime, p not dividing b-1) satisfies q = 1 (mod 2p), so
R_b(p) has no prime factor below 2p+1 (a = 2); Wagstaff's refinement for b = 2 uses the
quadratic-residue restriction on q (a = 2 or 6 by p mod 4). In t = ln n, with the density
of primes 1/ln n, this is a Poisson intensity

    lambda_b(t) = (kappa / ln b) * (1 + c_b / t),    c_b = <ln a_b>,  kappa = e^gamma.

So LPW itself predicts that a constant-rate fit overestimates e^gamma at finite n. By
Proposition 3 / Remark 8 of the paper, conditionally on the last event the interior
events form a Poisson process with this intensity on (t_0, t_0 + D_b), whatever the
(non-informative) frontier. Hence, for FIXED c, K | D ~ Poisson(kappa * S(c)) exactly, with

    S(c) = sum_b [ D_b + c_b * ln(t_last,b / t_0,b) ] / ln b,

and the Garwood interval applies verbatim.

Tags printed:
  [S14] Wagstaff-type <ln a_b> per base (quadratic-residue restriction)
  [S15] exact conditional inference for c = 0, c = ln 2, c = <ln a_b>, free c (profile LR)
  [S16] the same under lower truncation (does the truncation trend disappear?)
  [S17] model exp(c1/t + c2/t^2) (Remark 8 form) and homogeneity test at B = 20

Usage: python lpw_second_order.py        (needs data/ b-files; seed 20260924)
"""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
KAPPA = math.exp(0.5772156649015329)
SEED = 20260924
SEQ = {2: "A000043", 3: "A028491", 5: "A004061", 6: "A004062", 7: "A004063",
       10: "A004023", 11: "A005808", 12: "A004064", 13: "A016054", 14: "A006032",
       15: "A006033", 17: "A006034", 18: "A133857", 19: "A006035", 20: "A127995"}
SEQ_EXTRA = {21: "A127996", 22: "A127997", 23: "A204940", 24: "A127998", 26: "A127999"}
A000043_EXTRA = [82589933, 136279841]


def load(table):
    out = {}
    for b, sid in table.items():
        vals = [int(l.split()[-1]) for l in (DATA / (sid + ".txt")).read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        if b == 2:
            vals = sorted(set(vals) | set(A000043_EXTRA))
        out[b] = np.array(sorted(vals), dtype=np.float64)
    return out


# ---------------------------------------------------------------- [S14]
def primes_between(lo, hi):
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0][np.nonzero(sieve)[0] >= lo]


def jacobi(a, n):
    a %= n
    r = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                r = -r
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            r = -r
        a %= n
    return r if n == 1 else 0


def mean_ln_a(b, lo=100_000, hi=200_000):
    """Average of ln(2k) over primes p, where 2kp+1 is the smallest candidate divisor
    compatible with the necessary condition (b/q) = +1 (b is a square mod q because
    b^p = 1 and p is odd). This is Wagstaff's device for b = 2 (a = 2 or 6)."""
    vals = []
    for p in primes_between(lo, hi):
        if (b - 1) % p == 0:
            continue
        k = 1
        while jacobi(b, 2 * k * p + 1) != 1:
            k += 1
        vals.append(math.log(2 * k))
    return float(np.mean(vals))


# ---------------------------------------------------------------- data blocks
class Base:
    def __init__(self, b, idx):
        self.b, self.lnb = b, math.log(b)
        t = np.log(idx)
        self.t0, self.tl = float(t[0]), float(t[-1])
        self.inner = t[1:-1]              # events used by the conditional likelihood
        self.K = len(self.inner)
        self.D = self.tl - self.t0
        self.L = math.log(self.tl / self.t0)   # int_{t0}^{tl} dt / t


def build(seqs, truncate=None, min_events=2):
    out = []
    for b in sorted(seqs):
        idx = seqs[b] if truncate is None else seqs[b][seqs[b] > truncate]
        if len(idx) - 1 >= min_events:
            out.append(Base(b, idx))
    return out


def S_of(bases, c):
    c = np.broadcast_to(np.asarray(c, dtype=float), (len(bases),))
    return float(sum((bd.D + ci * bd.L) / bd.lnb for bd, ci in zip(bases, c)))


def garwood(k, S, a=0.05):
    return stats.chi2.ppf(a / 2, 2 * k) / (2 * S), stats.chi2.ppf(1 - a / 2, 2 * (k + 1)) / (2 * S)


def p_two(k, mu):
    return min(1.0, 2 * min(stats.poisson.cdf(k, mu), stats.poisson.sf(k - 1, mu)))


def exact(bases, c):
    K = sum(bd.K for bd in bases)
    S = S_of(bases, c)
    lo, hi = garwood(K, S)
    return K / S, lo, hi, p_two(K, KAPPA * S)


def profile_ll(bases, c):
    """Conditional log-likelihood maximised over kappa, for a common c."""
    K = sum(bd.K for bd in bases)
    S = S_of(bases, c)
    return K * math.log(K / S) - K + sum(float(np.sum(np.log1p(c / bd.inner))) for bd in bases)


def fit_c(bases):
    cmin = -min(min(bd.t0 for bd in bases), 50) + 1e-6
    r = optimize.minimize_scalar(lambda c: -profile_ll(bases, c), bounds=(cmin, 50), method="bounded")
    chat = r.x
    lmax = -r.fun
    # 95% profile-likelihood interval for c
    f = lambda c: 2 * (lmax - profile_ll(bases, c)) - stats.chi2.ppf(0.95, 1)
    lo = optimize.brentq(f, cmin, chat) if f(cmin) > 0 else cmin
    hi = optimize.brentq(f, chat, 50) if f(50) > 0 else float("inf")
    return chat, lo, hi, lmax


# ---------------------------------------------------------------- [S17]
def ll_exp_model(bases, th):
    """Conditional log-likelihood of lambda = (kappa/ln b) exp(c1/t + c2/t^2)."""
    lk, c1, c2 = th
    total = 0.0
    for bd in bases:
        g = lambda t: math.exp(c1 / t + c2 / t ** 2)
        integ, _ = integrate.quad(g, bd.t0, bd.tl, limit=200)
        total += bd.K * (lk - math.log(bd.lnb)) + float(np.sum(c1 / bd.inner + c2 / bd.inner ** 2)) \
            - math.exp(lk) / bd.lnb * integ
    return total


def homogeneity(bases, rng, reps=1_000_000):
    K = np.array([bd.K for bd in bases])
    s = np.array([bd.D / bd.lnb for bd in bases])
    e = K.sum() * s / s.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = 2 * float(np.sum(np.where(K > 0, K * np.log(K / e), 0)))
        hits = 0
        for _ in range(reps // 100_000):
            x = rng.multinomial(K.sum(), s / s.sum(), size=100_000)
            t = 2 * np.sum(np.where(x > 0, x * np.log(x / e), 0), axis=1)
            hits += int(np.sum(t >= lam - 1e-9))
    return lam, hits / reps, float(stats.chi2.sf(lam, len(bases) - 1))


def main():
    rng = np.random.default_rng(SEED)
    seqs = load(SEQ)
    bases = build(seqs)

    print("[S14] Wagstaff-type <ln a_b> over primes p in [1e5, 2e5]")
    lna = {}
    for bd in bases:
        lna[bd.b] = mean_ln_a(bd.b)
        print("  b=%2d  <ln a> = %.4f  (a_eff = %.3f)" % (bd.b, lna[bd.b], math.exp(lna[bd.b])))
    c_w = [lna[bd.b] for bd in bases]

    print("\n[S15] exact conditional inference, all data (B=%d, K=%d)" % (len(bases), sum(bd.K for bd in bases)))
    rows = [("c = 0 (constant rate)", 0.0), ("c = ln 2 (q = 1 mod 2p)", math.log(2)),
            ("c = <ln a_b> (Wagstaff-type)", c_w)]
    for name, c in rows:
        k, lo, hi, p = exact(bases, c)
        print("  %-30s S=%9.4f  kappa=%.4f  95%% [%.4f; %.4f]  p=%.3f  kappa/e^g=%.4f"
              % (name, S_of(bases, c), k, lo, hi, p, k / KAPPA))
    chat, clo, chi, lmax = fit_c(bases)
    k, lo, hi, p = exact(bases, chat)
    lr0 = 2 * (lmax - profile_ll(bases, 0.0))
    lr2 = 2 * (lmax - profile_ll(bases, math.log(2)))
    print("  free c: c_hat=%.3f  95%% profile CI [%.3f; %s]  kappa(c_hat)=%.4f  95%% [%.4f; %.4f]"
          % (chat, clo, "%.3f" % chi if np.isfinite(chi) else "inf", k, lo, hi))
    print("  LR test c=0: %.3f (p=%.3f);  LR test c=ln2: %.3f (p=%.3f)"
          % (lr0, stats.chi2.sf(lr0, 1), lr2, stats.chi2.sf(lr2, 1)))
    # joint test of LPW with c = ln 2: kappa = e^gamma under c fixed is the exact Poisson test above

    print("\n[S16] lower truncation (origin = first surviving index, N_b >= 2)")
    print("  %-8s %3s %4s  %-9s %-9s %-9s  %-6s %-6s" % ("trunc", "B", "K", "k(c=0)", "k(ln2)", "k(W)", "p(0)", "p(ln2)"))
    for tr in [None, 10, 100, 1000, 10000]:
        bs = build(seqs, truncate=tr)
        cw = [lna[bd.b] for bd in bs]
        k0, _, _, p0 = exact(bs, 0.0)
        k2, _, _, p2 = exact(bs, math.log(2))
        kw, _, _, _ = exact(bs, cw)
        print("  %-8s %3d %4d  %.4f    %.4f    %.4f     %.3f  %.3f"
              % ("none" if tr is None else ">%d" % tr, len(bs), sum(b.K for b in bs), k0, k2, kw, p0, p2))
    print("  free c per truncation level (profile likelihood):")
    for tr in [None, 10, 100, 1000, 10000]:
        bs = build(seqs, truncate=tr)
        chat, clo, chi, lmax = fit_c(bs)
        kc, lo, hi, pc = exact(bs, chat)
        lr0 = 2 * (lmax - profile_ll(bs, 0.0))
        lr2 = 2 * (lmax - profile_ll(bs, math.log(2)))
        lrw = 2 * (lmax - max(profile_ll(bs, c) for c in [1.30, 1.40, 1.50]))
        _, _, _, pw = exact(bs, [lna[bd.b] for bd in bs])
        print("  %-8s c_hat=%6.3f  CI [%6.3f; %s]  kappa(c_hat)=%.4f [%.3f; %.3f]  "
              "LR(c=0) p=%.3f  LR(c=ln2) p=%.3f  p(kappa=e^g | c=W)=%.3f"
              % ("none" if tr is None else ">%d" % tr, chat, clo,
                 "%6.3f" % chi if np.isfinite(chi) else "  inf", kc, lo, hi,
                 stats.chi2.sf(lr0, 1), stats.chi2.sf(lr2, 1), pw))

    print("\n[S17] model lambda = (kappa/ln b) exp(c1/t + c2/t^2), conditional likelihood")
    f = lambda th: -ll_exp_model(bases, th)
    r0 = optimize.minimize(lambda x: f([x[0], 0.0, 0.0]), [math.log(1.8)], method="Nelder-Mead")
    r1 = optimize.minimize(f, [math.log(1.8), 0.5, 0.0], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    lr = 2 * (r0.fun - r1.fun)
    print("  null: kappa=%.4f lnL=%.3f | full: kappa=%.4f c1=%.3f c2=%.3f lnL=%.3f | LR=%.3f p=%.3f"
          % (math.exp(r0.x[0]), -r0.fun, math.exp(r1.x[0]), r1.x[1], r1.x[2], -r1.fun, lr, stats.chi2.sf(lr, 2)))

    allb = build({**seqs, **load(SEQ_EXTRA)}, min_events=2)
    M = sum(bd.K + 1 for bd in allb)
    S = sum(bd.D / bd.lnb for bd in allb)
    lam, pex, pas = homogeneity(allb, rng)
    print("  B=%d: M=%d  (M-1)/S=%.4f  (M-B)/S=%.4f  Lambda=%.3f  exact p=%.3f  (chi2 p=%.3f)"
          % (len(allb), M, (M - 1) / S, (M - len(allb)) / S, lam, pex, pas))


if __name__ == "__main__":
    sys.exit(main())
