#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сверка ключевых чисел paper_ru.tex и paper_en.tex (версия 3) с выводом
analysis_v2.py (results_v2.txt) и results_v3.txt.

Каждая проверка: (регулярное выражение по выводу, номер группы, формат, подпись).
Число из вывода форматируется и ищется в тексте статьи (в русской версии -- с
запятой {,}, в английской -- с точкой). Код возврата 1 при любом расхождении.

Запуск:  python check_numbers_v3.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = (HERE / "results_v2.txt").read_text(encoding="utf-8") + \
    (HERE / "results_v3.txt").read_text(encoding="utf-8")
PAPERS = [("ru", HERE / "paper_ru.tex", "{,}"), ("en", HERE / "paper_en.tex", ".")]

N = r"(-?\d+\.\d+)"
CHECKS = [
    (r"S = " + N + ", B = 15", 1, "%.4f", "S"),
    (r"b <= 26: B = 20, M = 253, \(M-1\)/S = " + N, 1, "%.4f", "(M-1)/S при B=20"),
    (r"b <= 26: .*\(M-B\)/S = " + N, 1, "%.4f", "(M-B)/S при B=20"),
    (r"схема B: \(M-B\)/S = " + N, 1, "%.4f", "(M-B)/S"),
    (r"95% Гарвуд \[" + N + "; " + N + r"\]", 1, "%.4f", "Гарвуд, нижний"),
    (r"95% Гарвуд \[" + N + "; " + N + r"\]", 2, "%.4f", "Гарвуд, верхний"),
    (r"95% Гарвуд .*p = " + N, 1, "%.3f", "p схемы B"),
    (r"mid-p " + N + r"\), e\^gamma", 1, "%.3f", "mid-p схемы B"),
    (r"покрытие интервала: " + N, 1, "%.4f", "покрытие при e^gamma"),
    (r"схема A: \(M-1\)/S = " + N, 1, "%.4f", "(M-1)/S"),
    (r"схема A: \(M-1\)/S = .*95% \[" + N, 1, "%.4f", "ДИ схемы A, нижний"),
    (r"схема A: \(M-1\)/S = .*p = " + N, 1, "%.3f", "p схемы A"),
    (r"SD\(kappa_B\) = " + N, 1, "%.4f", "SD схемы B"),
    (r"Lambda = " + N + ", df = 14", 1, "%.3f", "Lambda"),
    (r"Lambda = .*точное p = " + N, 1, "%.3f", "точное p однородности"),
    (r"наибольшие расхождения форм: b=18 " + r"(\d+\.\d)%", 1, "%s", "расхождение b=18"),
    (r"отклонения от e\^-gamma: медиана (\d+\.\d)%", 1, "%s", "медиана отклонений"),
    (r"80%: только Мерсенн \+(\d+\.\d)%", 1, "%s", "порог Мерсенна"),
    (r"объединение \+(\d+\.\d)%", 1, "%s", "порог объединения"),
    (r"односторонний \+(\d+\.\d)%", 1, "%s", "односторонний порог"),
    (r"исключение по одному: \(M-B\)/S от " + N, 1, "%.3f", "исключение по одному, мин."),
    (r"исключение по одному: .* до " + N + r" \(без", 1, "%.3f", "исключение по одному, макс."),
    (r"наклон " + N + ", свободный член", 1, "%.4f", "наклон §7"),
    (r"свободный член " + N + r" \(SE", 1, "%.4f", "свободный член §7"),
    (r"Ньюи -- Уэст " + N, 1, "%.4f", "Ньюи -- Уэст"),
    (r"нулевой закон свободного члена .*медиана " + N, 1, "%.3f", "медиана нулевого закона"),
    (r"бутстрэп \(\d+\): медиана " + N, 1, "%.3f", "медиана бутстрэпа"),
    (r"lambda\^ = " + N + r" \(ЛПВ", 1, "%.4f", "lambda^ при известном фронте"),
    (r"kappa_2 = " + N, 1, "%.4f", "kappa_2 при известном фронте"),
    (r"условная оценка того же среза: " + N, 1, "%.4f", "условная оценка среза"),
    (r"C\^fix: среднее C_b = " + N, 1, "%.3f", "среднее C^fix"),
    (r"C\^fix: .*kappa\^ = " + N, 1, "%.3f", "kappa C^fix"),
    (r"^\s+A\s+1\s+" + N, 1, "%.3f", "ln L модели A"),
    (r"c = ln 2 \(q = 1 mod 2p\)\s+S=\s*[\d.]+\s+kappa=" + N, 1, "%.3f", "kappa при c=ln2"),
    (r"c = <ln a_b> \(Wagstaff-type\)\s+S=\s*[\d.]+\s+kappa=" + N, 1, "%.3f", "kappa по Вагстаффу"),
    (r"full: kappa=" + N, 1, "%.3f", "kappa модели exp(c1/t+c2/t^2)"),
    (r"B=20: M=253 .*Lambda=" + N, 1, "%.2f", "Lambda при B=20"),
    (r"B=20: M=253 .*exact p=" + N, 1, "%.3f", "p однородности при B=20"),
    (r"hybrid estimator.*\n\s+K=205 S=[\d.]+ kappa=" + N, 1, "%.3f", "гибридная оценка"),
    (r"drought-2\s+[\d.]+\s+" + N, 1, "%.4f", "(M-B)/S при засухе g=2"),
]


def main():
    bad = 0
    for lang, path, dec in PAPERS:
        if not path.exists():
            print("-- %s: файла нет, пропущено" % path.name)
            continue
        tex = path.read_text(encoding="utf-8")
        print("-- %s" % path.name)
        for rx, grp, fmt, label in CHECKS:
            m = re.search(rx, RES, re.M)
            if not m:
                print("  НЕТ В ВЫВОДЕ    %s" % label)
                bad += 1
                continue
            v = m.group(grp)
            s = fmt % (float(v) if fmt != "%s" else v)
            s = s.lstrip("-")
            want = s.replace(".", dec)
            ok = want in tex
            if not ok:
                bad += 1
            print("  %-4s %-36s %s" % ("ok" if ok else "НЕТ", label, want))
    print("расхождений: %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
