"""[S23] A first-principles prediction of the second-order constant c_b (section 9).

cb_empirical.py [S19] calibrates c_b from the ACTUAL divisibility of R_b(p). Here the
same quantity is predicted WITHOUT computing b^p mod q, from two ingredients only:

  (i)  which candidates q = 2kp+1 < Y are prime (exact, by a sieve);
  (ii) the quadratic character (b/q) (exact, deterministic): if b^p = 1 (mod q) with
       p odd, then b is a square mod q.

A prime q = 2kp+1 divides R_b(p) iff b lies in the subgroup of index 2k of F_q^*. Given
that b is a quadratic residue (index-2 subgroup), the heuristic "uniform within the
residues" gives probability 2/(2k) = 1/k (exactly 1 for k = 1: if 2p+1 is prime and b
is a residue, 2p+1 | R_b(p), as for Mersenne numbers with p = 3 mod 4). Hence

    F_th(p, Y) = prod_{k < Y/(2p), q=2kp+1 prime, q not | b, (b/q)=+1} (1 - 1/k),
    c_b^th(Y)  = < F_th(p, Y) ln Y - ln p >_p .

Refinement for bases with a square factor (b = s^2 b0, b0 squarefree, s > 1): the
4th-power character of b is then partly determined by the quadratic character of s,
so "uniform within the residues" is only approximate; such bases (12, 18, 20) are
flagged in the output.

Usage: python cb_theory.py   (deterministic)
"""
import math

import numpy as np

BASES = [2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
P_LO, P_HI = 20_000, 400_000
MULT = 1000                      # Y = MULT * p, as in the calibration column of [S19]
CB_EMP = {2: 0.959, 3: 0.443, 5: 1.063, 6: 0.587, 7: 0.487, 10: 0.554, 11: 0.473, 12: 0.428,
          13: 0.667, 14: 0.533, 15: 0.537, 17: 0.755, 18: 0.861, 19: 0.538, 20: 1.106}


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def jacobi_vec(a, n):
    """Jacobi symbol (a/n) for a fixed small a and an array of odd n."""
    a = np.full_like(n, a) % n
    n = n.copy()
    r = np.ones_like(n)
    active = a != 0
    while np.any(active):
        # remove factors of two from a
        while True:
            even = active & (a % 2 == 0) & (a != 0)
            if not np.any(even):
                break
            a[even] //= 2
            flip = even & ((n % 8 == 3) | (n % 8 == 5))
            r[flip] = -r[flip]
        # reciprocity and swap
        swap = active & (a != 0)
        flip = swap & (a % 4 == 3) & (n % 4 == 3)
        r[flip] = -r[flip]
        a_s, n_s = a[swap].copy(), n[swap].copy()
        a[swap], n[swap] = n_s % a_s, a_s
        active = a != 0
    return np.where(n == 1, r, 0)


def squarefree_part(b):
    b0, d = 1, 2
    x = b
    while d * d <= x:
        e = 0
        while x % d == 0:
            x //= d
            e += 1
        if e % 2:
            b0 *= d
        d += 1
    return b0 * x


def main():
    isprime = sieve(MULT * P_HI + 1)
    ps = np.nonzero(isprime[P_LO:P_HI])[0] + P_LO
    kmax = MULT // 2
    ks = np.arange(1, kmax + 1, dtype=np.int64)
    print("[S23] predicted c_b from primality of 2kp+1 and the quadratic character (b/q)")
    print("      primes p in [%d, %d): %d; Y = %d p; paired with the actual divisibility"
          % (P_LO, P_HI, len(ps), MULT))
    print("  %3s %9s %9s %9s %7s %6s  %s" % ("b", "c_b^th", "c_b^act", "diff", "se", "z", "note"))
    th_all, act_all, zs = [], [], []
    for b in BASES:
        th, act = [], []
        for p in ps:
            p = int(p)
            if (b - 1) % p == 0:
                continue
            q = 2 * ks * p + 1
            ok = isprime[q] & (b % q != 0)
            qq = q[ok]
            kk = ks[ok]
            chi = jacobi_vec(b, qq)
            use = chi == 1
            k_use = kk[use]
            F = 0.0 if np.any(k_use == 1) else float(np.prod(1.0 - 1.0 / k_use))
            # actual: a prime divisor of R_b(p) below Y must be one of these q
            # (prime, = 1 mod 2p, (b/q) = +1)
            hit = any(pow(b, p, int(x)) == 1 for x in qq[use])
            lnY, lnp = math.log(MULT * p), math.log(p)
            th.append(F * lnY - lnp)
            act.append((0.0 if hit else 1.0) * lnY - lnp)
        th, act = np.array(th), np.array(act)
        d = act - th
        se = d.std(ddof=1) / math.sqrt(len(d))
        z = d.mean() / se
        zs.append(z)
        th_all.append(th.mean())
        act_all.append(act.mean())
        note = "square factor" if squarefree_part(b) != b else ""
        print("  %3d %9.3f %9.3f %+9.3f %7.3f %+6.2f  %s"
              % (b, th.mean(), act.mean(), d.mean(), se, z, note))
    zs = np.array(zs)
    chi2 = float(np.sum(zs ** 2))
    from scipy import stats
    print("  correlation(theory, actual) = %.3f; sum z^2 = %.1f on %d df, p = %.3f; "
          "mean diff = %+.3f"
          % (np.corrcoef(th_all, act_all)[0, 1], chi2, len(zs), stats.chi2.sf(chi2, len(zs)),
             float(np.mean(np.array(act_all) - np.array(th_all)))))


if __name__ == "__main__":
    main()
