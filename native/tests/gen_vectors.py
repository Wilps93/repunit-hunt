"""Генерация тестовых векторов для Montgomery-арифметики трёх ширин.

Выход: native/tests/mont_vectors.h  (массив векторов + эталонные b^k mod q)
и печать «боевых» троек (b,k,q) для проверки самих ядер.
"""
import random, sys
from sympy import isprime, nextprime, primerange

random.seed(20260826)
OUT = sys.argv[1] if len(sys.argv) > 1 else "mont_vectors.h"

def rand_odd(lo, hi):
    while True:
        x = random.randrange(lo, hi) | 1
        if lo <= x < hi:
            return x

def rand_prime(lo, hi):
    while True:
        p = nextprime(random.randrange(lo, hi))
        if p < hi:
            return p

vectors = []   # (width, b, k, q, expected = b^k mod q)

# ── Width 0: Mont64, q < 2^63 ────────────────────────────────────────
for _ in range(40):
    q = rand_prime(1 << 32, 1 << 62)
    b = random.randrange(2, 1 << 63)
    k = rand_prime(3, 1 << 20)
    vectors.append((0, b, k, q, pow(b % q, k, q)))
# краевые: q близко к 2^63, маленькие q, b > q, k с длинной битовой записью
vectors.append((0, 10, 1031, rand_prime((1 << 62) - 10**6, (1 << 62)), None))
vectors.append((0, 2, 65537, 4294967291, None))
vectors.append((0, (1 << 64) - 59, 999983, rand_prime(1 << 61, 1 << 62), None))

# ── Width 1: Mont96, 2^64 <= q < 2^95 (плюс перекрытие ниже 2^64) ────
for _ in range(40):
    q = rand_prime(1 << 64, 1 << 94)
    b = random.randrange(2, 1 << 64)
    k = rand_prime(3, 1 << 20)
    vectors.append((1, b, k, q, None))
for _ in range(10):                       # Mont96 обязана работать и для q < 2^64
    q = rand_prime(1 << 40, 1 << 63)
    b = random.randrange(2, 1 << 60)
    k = rand_prime(3, 1 << 16)
    vectors.append((1, b, k, q, None))

# ── Width 2: Mont128, 2^95 <= q < 2^127 ─────────────────────────────
for _ in range(40):
    q = rand_prime(1 << 95, 1 << 126)
    b = random.randrange(2, 1 << 64)
    k = rand_prime(3, 1 << 20)
    vectors.append((2, b, k, q, None))
# краевой: q у самой границы 2^127
vectors.append((2, 10, 1031, rand_prime((1 << 127) - 10**7, (1 << 127) - 1), None))

# досчитываем эталоны
vectors = [(w, b, k, q, pow(b % q, k, q)) for (w, b, k, q, _) in vectors]

def lo(x): return x & 0xFFFFFFFFFFFFFFFF
def hi(x): return (x >> 64) & 0xFFFFFFFFFFFFFFFF

with open(OUT, "w", encoding="utf-8") as f:
    f.write("/* Сгенерировано scratchpad/gen_vectors.py — не редактировать вручную.\n")
    f.write(" * Эталон b^k mod q посчитан независимо (Python int, произвольная точность). */\n")
    f.write("#pragma once\n#include <stdint.h>\n\n")
    f.write("typedef struct { uint32_t width; uint64_t b, k, q_lo, q_hi, e_lo, e_hi; } mont_vec_t;\n\n")
    f.write(f"#define MONT_VEC_N {len(vectors)}\n")
    f.write("static const mont_vec_t g_mont_vecs[MONT_VEC_N] = {\n")
    for (w, b, k, q, e) in vectors:
        f.write(f"  {{ {w}u, {b}ull, {k}ull, {lo(q)}ull, {hi(q)}ull, {lo(e)}ull, {hi(e)}ull }},\n")
    f.write("};\n")

print(f"записано векторов: {len(vectors)} -> {OUT}")
for w in (0, 1, 2):
    n = sum(1 for v in vectors if v[0] == w)
    print(f"  width={w}: {n}")

# ── «Боевые» тройки для проверки самих ядер ──────────────────────────
print("\n=== боевые векторы (настоящие делители репьюнитов) ===")

# W128: делитель, найденный нашим же P-1 на прогоне k=1153, b=10
q128, k128, b128 = 145074808610313100243540874761, 1153, 10
N = (10**k128 - 1) // 9
ok = N % q128 == 0
m128 = (q128 - 1) // (2 * k128)
print(f"W128: b={b128} k={k128} q={q128}")
print(f"      q делит R_k: {ok}; q простое: {isprime(q128)}")
print(f"      q = 2*m*k+1 ровно: {2*m128*k128+1 == q128}; m={m128}")
print(f"      log2(q) = {q128.bit_length()} бит -> ширина {'W128' if q128 >= 2**95 else 'W96'}")

# W96: конструируем q чуть выше 2^64 с элементом порядка k, помещающимся в u64
print("\nW96: ищу q > 2^64 c ord_q(b) = k и b < 2^64 ...")
found = None
for k in [7, 11, 13]:
    lo_q = 1 << 64
    q = lo_q
    for _ in range(4000):
        q = nextprime(q)
        if (q - 1) % (2 * k):
            continue
        g = 2
        while pow(g, (q - 1) // 2, q) == 1:      # ищем невычет -> вероятный генератор
            g += 1
        h = pow(g, (q - 1) // k, q)              # элемент порядка k (или 1)
        if h == 1:
            continue
        # весь подгруппный цикл: ищем представителя < 2^64
        x = h
        for _ in range(k):
            if 2 <= x < (1 << 64):
                found = (x, k, q)
                break
            x = x * h % q
        if found:
            break
    if found:
        break

if found:
    b, k, q = found
    R = (b**k - 1) // (b - 1)
    m = (q - 1) // (2 * k)
    print(f"W96: b={b} k={k} q={q}")
    print(f"      q делит R_k(b): {R % q == 0}; q простое: {isprime(q)}")
    print(f"      q = 2*m*k+1: {2*m*k+1 == q}; m={m}")
    print(f"      log2(q) = {q.bit_length()} бит")
else:
    print("W96: подходящая тройка не найдена")
