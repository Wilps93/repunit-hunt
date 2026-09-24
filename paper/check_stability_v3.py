#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Устойчивость выводов версии 3 к зерну генератора.

analysis_v2.py прогоняется при нескольких зёрнах (--seed). Детерминированные
величины (точная инверсия пуассоновского распределения) обязаны совпасть
побитово; величины Монте-Карло -- остаться в содержательных коридорах: не
«совпало до знака», а «вывод тот же».

Запуск:  python check_stability_v3.py [зёрна...]   (по умолчанию 1 2 3)
Код возврата 1, если хоть одна величина вышла из коридора.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [int(x) for x in sys.argv[1:]] or [1, 2, 3]

# (подпись, регулярное выражение, группа, нижняя граница, верхняя граница)
CORRIDORS = [
    ("точное p однородности", r"точное p = ([\d.]+)", 1, 0.90, 0.96),
    ("cv при мощности 80%", r"мощность 80% при cv ~ ([\d.]+)", 1, 0.33, 0.41),
    ("ошибка I рода, правило 1", r"E\[#\] = наблюдённое\s+([\d.]+)", 1, 0.14, 0.19),
    ("ошибка I рода, фронт вдвое дальше", r"вдвое дальше по t\s+([\d.]+)", 1, 0.09, 0.14),
    ("E[(M-B)/S], правило 1", r"E\[#\] = наблюдённое\s+[\d.]+\s+[\d.]+\s+([\d.]+)", 1, 0.995, 1.005),
    ("E[(M-1)/S], правило 1", r"E\[#\] = наблюдённое\s+[\d.]+\s+([\d.]+)", 1, 1.05, 1.09),
    ("смещение kappa* при схеме B, %", r"схема B: смещение kappa\* = \+([\d.]+)%", 1, 6.5, 8.5),
    ("p свободного члена §7", r"нулевой закон свободного члена .*p = ([\d.]+)", 1, 0.45, 0.70),
    ("контроль схемы A", r"A \(контроль\)\s+([\d.]+)", 1, 0.04, 0.06),
]
# Строки, которые не зависят от генератора и должны совпасть побитово.
EXACT = [r"схема B: \(M-B\)/S = .*", r"схема A: \(M-1\)/S = .*", r"Lambda = [\d.]+, df = 14",
         r"80%: только Мерсенн .*", r"исключение по одному: .*"]


def run(seed):
    out = HERE / ("results_v2_seed%d.txt" % seed)
    subprocess.run([sys.executable, str(HERE / "analysis_v2.py"), "--seed", str(seed)],
                   check=True, stdout=subprocess.DEVNULL)
    text = out.read_text(encoding="utf-8")
    out.unlink()
    return text


def main():
    base = (HERE / "results_v2.txt").read_text(encoding="utf-8")
    bad = 0
    for seed in SEEDS:
        text = run(seed)
        print("-- зерно %d" % seed)
        for label, rx, g, lo, hi in CORRIDORS:
            m = re.search(rx, text)
            if not m:
                print("  НЕТ В ВЫВОДЕ  %s" % label)
                bad += 1
                continue
            v = float(m.group(g))
            ok = lo <= v <= hi
            bad += not ok
            print("  %-4s %-36s %8.4f  [%g; %g]" % ("ok" if ok else "ВЫШЛА", label, v, lo, hi))
        for rx in EXACT:
            a, b = re.search(rx, base), re.search(rx, text)
            same = bool(a and b and a.group(0) == b.group(0))
            bad += not same
            print("  %-4s побитово: %s" % ("ok" if same else "РАСХОЖДЕНИЕ", rx[:40]))
    print("величин вне коридора или с расхождением: %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
