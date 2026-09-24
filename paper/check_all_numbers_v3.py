#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Полная сверка: КАЖДОЕ десятичное число в тексте paper_ru_v3.tex и paper_en_v3.tex
(кроме комментариев и списка литературы) должно найтись в выводе скриптов
(results_v2.txt, results_v3.txt) с точностью до округления, в том числе в виде
процента или отклонения от единицы. Исключения перечислены явно с обоснованием.
Код возврата 1, если найдено число без источника.

Запуск:  python check_all_numbers_v3.py
"""
import bisect
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = (HERE / "results_v2.txt").read_text(encoding="utf-8") + \
    (HERE / "results_v3.txt").read_text(encoding="utf-8")

# Числа, которые не печатаются скриптами, с обоснованием (проверены вручную).
KNOWN = {
    "511.3": "УДК", "519.234": "УДК",
    "0.26150": "постоянная Мейсселя -- Мертенса",
    "1.3326": "постоянная E первой теоремы Мертенса (|E|)",
    "0.950": "гарантированная нижняя граница покрытия интервала Гарвуда",
    "0.89": "1 - 0.111 (верхний хвост, frontiers.py)",
    "0.90": "1 - 0.103 (верхний хвост, frontiers.py)",
    "6.8": "2.0990/1.9648 - 1 (таблица усечения)",
    "15.7": "1 - 1.5781/1.8712 (§9)",
    "0.108": "c1 модели версии 1 (results.txt)", "0.826": "c2 модели версии 1 (results.txt)",
    "0.8564": "корень 1 + 0.108/t - 0.826/t^2", "2.355": "exp(0.8564)",
    "0.5634": "значение того же множителя при t = ln 2", "1.0986": "ln 3",
}


def pool():
    out = set()
    for v in (float(x) for x in re.findall(r"-?\d+\.\d+(?:e[+-]?\d+)?", RES)):
        for w in (v, 100 * v, v / 100, 100 * (v - 1)):
            out.add(abs(w))
    return sorted(out)


def main():
    P = pool()
    bad = 0
    for name in ("paper_ru_v3.tex", "paper_en_v3.tex"):
        path = HERE / name
        if not path.exists():
            continue
        tex = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
        cut = tex.find("\\begin{thebibliography}")
        tex = tex[:cut] if cut >= 0 else tex
        nums = re.findall(r"(?<![\d{,])(\d+)\{,\}(\d+)", tex) + \
            re.findall(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", tex)
        seen, miss = set(), []
        for a, b in nums:
            key = a + "." + b
            if key in seen:
                continue
            seen.add(key)
            x, tol = float(key), 0.5 * 10 ** (-len(b)) + 1e-12
            i = bisect.bisect_left(P, x - tol)
            if not (i < len(P) and P[i] <= x + tol) and key not in KNOWN:
                miss.append(key)
        bad += len(miss)
        print("%s: %d различных десятичных чисел, без источника: %s"
              % (name, len(seen), ", ".join(miss) if miss else "нет"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
