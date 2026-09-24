"""[S19] Empirical calibration of the second-order constant c_b of section 9.

The LPW heuristic writes Pr(R_b(p) prime) = e^gamma ln(a_b p) / (p ln b); section 9 sets
c_b = <ln a_b>. Here c_b is MEASURED rather than taken from Wagstaff's device.

For a random integer N, Mertens gives Pr(N prime | no prime factor < Y) ~ e^gamma ln Y / ln N.
Hence Pr(R_b(p) prime) ~ F_b(p, Y) e^gamma ln Y / (p ln b), where F_b(p, Y) is the
probability that R_b(p) has no prime factor below Y. Matching the two forms gives

    c_b(Y) = < F_b(p, Y) ln Y - ln p >   (average over primes p),

which should stabilise as Y grows. The indicator F_b is computed EXACTLY for every p:
R_b(p) (p prime, p not dividing b-1) has a prime factor below Y  <=>  b^p = 1 (mod q) for
some q = 2kp+1 < Y. (=>: a prime divisor r of R_b(p) has ord_r(b) = p, so r = 2kp+1.
<=: every prime factor r of such q exceeds b, divides b^p - 1 but not b - 1, hence divides
R_b(p), and r <= q < Y.) So no primality test of q is needed.

Y = m p with m fixed, so ln(Y/p) = ln m for every p. The standard error of c_b(Y) is
ln Y * sd(F)/sqrt(#p) and is printed.

Usage: python cb_empirical.py   (deterministic, about a minute)
"""
import math

import numpy as np

BASES = [2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
P_LO, P_HI = 20_000, 400_000
MULTS = [100, 1000, 3000]
CHUNK = 1500


def primes_in(lo, hi):
    s = np.ones(hi, dtype=bool)
    s[:2] = False
    for i in range(2, int(hi ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0][np.nonzero(s)[0] >= lo].astype(np.uint64)


def powmod_vec(base, exps, mods):
    """base ** exps mod mods, elementwise; mods < 2^32 so products fit in uint64."""
    result = np.ones_like(mods)
    b = np.full_like(mods, base) % mods
    e = exps.copy()
    while np.any(e):
        odd = (e & np.uint64(1)).astype(bool)
        result[odd] = (result[odd] * b[odd]) % mods[odd]
        b = (b * b) % mods
        e >>= np.uint64(1)
    return result


def first_divisor_k(b, ps, kmax):
    """For each p: smallest k <= kmax with b^p = 1 mod (2kp+1), or 0."""
    out = np.zeros(len(ps), dtype=np.int64)
    ks = np.arange(1, kmax + 1, dtype=np.uint64)
    for i in range(0, len(ps), CHUNK):
        p = ps[i:i + CHUNK]
        q = (2 * ks[None, :] * p[:, None] + 1).ravel()
        e = np.repeat(p, kmax)
        hit = (powmod_vec(b, e, q) == 1).reshape(len(p), kmax)
        has = hit.any(axis=1)
        out[i:i + CHUNK] = np.where(has, hit.argmax(axis=1) + 1, 0)
    return out


def main():
    ps_all = primes_in(P_LO, P_HI)
    kmax = MULTS[-1] // 2
    print("[S19] empirical c_b(Y) = <F_b(p,Y) ln Y - ln p>, primes p in [%d, %d): %d primes"
          % (P_LO, P_HI, len(ps_all)))
    print("  %3s " % "b" + "".join("   Y=%-5dp (+-se)" % m for m in MULTS))
    for b in BASES:
        ps = ps_all[(b - 1) % ps_all.astype(np.int64) != 0] if b > 2 else ps_all
        k = first_divisor_k(b, ps, kmax)
        lp = np.log(ps.astype(float))
        cells = []
        for m in MULTS:
            # divisor 2kp+1 < m p  <=>  k < (m p - 1) / (2p), i.e. k <= (m - 1) // 2 for m p integer
            free = (k == 0) | (2 * k * ps.astype(np.int64) + 1 >= m * ps.astype(np.int64))
            lnY = np.log(m) + lp
            c = free * lnY - lp
            cells.append("   %6.3f (%.3f)   " % (c.mean(), c.std(ddof=1) / math.sqrt(len(c))))
        print("  %3d " % b + "".join(cells))


if __name__ == "__main__":
    main()
