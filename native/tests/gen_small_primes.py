"""Генерация таблицы для быстрой проверки делимости на малые простые.

ИДЕЯ. На GPU нет аппаратного целочисленного деления: каждое `q % p`
разворачивается в десятки инструкций. Но для НЕЧЁТНОГО p делимость
проверяется одним умножением:

    p | q   ⟺   (q · p⁻¹ mod 2⁶⁴) ≤ ⌊(2⁶⁴−1)/p⌋

где p⁻¹ — обратный к p по модулю 2⁶⁴. Умножение на обратный переставляет
вычеты так, что кратные p (и только они) попадают в начальный отрезок.

Замер (GTX 1650, 54 простых): деление — 36.1 мс/launch, умножение — см. README.

Запуск: python native/tests/gen_small_primes.py native/cuda/small_primes.h
"""
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "small_primes.h"
MASK = (1 << 64) - 1

# Простые 3..257 (двойка не нужна: q = 2mk+1 всегда нечётное)
PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
          53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
          113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
          193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]


def inv64(a: int) -> int:
    """a⁻¹ mod 2⁶⁴ методом Ньютона (a нечётное)."""
    x = a & MASK
    for _ in range(6):
        x = (x * (2 - a * x)) & MASK
    assert (a * x) & MASK == 1
    return x


with open(OUT, "w", encoding="utf-8") as f:
    f.write("/* Сгенерировано native/tests/gen_small_primes.py — не редактировать.\n")
    f.write(" *\n")
    f.write(" * Проверка делимости без деления:\n")
    f.write(" *     p | q  <=>  (q * inv[i]) <= lim[i]      (умножение по модулю 2^64)\n")
    f.write(" * где inv[i] = p^-1 mod 2^64, lim[i] = (2^64-1)/p.\n")
    f.write(" * Обоснование: умножение на обратный биективно на Z_2^64 и переводит\n")
    f.write(" * кратные p ровно в отрезок [0, (2^64-1)/p].\n")
    f.write(" */\n#pragma once\n#include <stdint.h>\n\n")
    f.write(f"#define RH_SMALL_PRIMES_MAX {len(PRIMES)}\n\n")

    f.write("__constant__ uint32_t d_sp_p[RH_SMALL_PRIMES_MAX] = {\n")
    for i in range(0, len(PRIMES), 10):
        f.write("    " + ", ".join(f"{p}u" for p in PRIMES[i:i + 10]) + ",\n")
    f.write("};\n\n")

    f.write("__constant__ uint64_t d_sp_inv[RH_SMALL_PRIMES_MAX] = {\n")
    for p in PRIMES:
        f.write(f"    {inv64(p)}ull,\n")
    f.write("};\n\n")

    f.write("__constant__ uint64_t d_sp_lim[RH_SMALL_PRIMES_MAX] = {\n")
    for p in PRIMES:
        f.write(f"    {MASK // p}ull,\n")
    f.write("};\n")

print(f"записано {len(PRIMES)} простых -> {OUT}")

# самопроверка на хосте
bad = 0
for p in PRIMES:
    inv, lim = inv64(p), MASK // p
    for q in (p, 3 * p, p * 1000003, p + 2, 12345678901234567, 2**63 + 1):
        expect = (q % p == 0)
        got = ((q * inv) & MASK) <= lim
        if expect != got:
            bad += 1
            print(f"РАСХОЖДЕНИЕ p={p} q={q}: ожидалось {expect}, получено {got}")
print("самопроверка:", "ок" if bad == 0 else f"ОШИБОК {bad}")
