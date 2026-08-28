#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сверка ключевых чисел статьи с выводом analysis.py.

Каждое утверждение задано парой (что искать в results.txt, что должно стоять
в paper_ru.tex). Скрипт возвращает 1, если хоть одно расходится, — годится для CI.

Запуск:  python3 check_numbers.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Обе версии статьи сверяются с одним и тем же выводом analysis.py. Русская
# пишет дробную часть через запятую (1{,}9648), английская — через точку
# (1.9648); в остальном числа обязаны совпадать.
TEX_RU = (HERE / "paper_ru.tex").read_text(encoding="utf-8")
TEX_EN = (HERE / "paper_en.tex").read_text(encoding="utf-8")
VERSIONS = (("ru", TEX_RU, "{,}"), ("en", TEX_EN, "."))
RES = (HERE / "results.txt").read_text(encoding="utf-8")


def res_num(pattern, group=1, flags=0):
    """Достать число из results.txt по регулярному выражению."""
    m = re.search(pattern, RES, flags)
    if not m:
        return None
    return float(m.group(group))


CHECKS = []


def both(name, ru, en):
    """Проверка по строке-контексту: своя формулировка в каждой версии."""
    miss = [tag for tag, tex, _ in VERSIONS if (ru if tag == "ru" else en) not in tex]
    return miss


def literal(x, digits, sep):
    """Число в том виде, в каком оно записано в TeX: 1{,}8381 или 1.8381."""
    return ("%.*f" % (digits, x)).replace(".", sep)


def check(name, value, digits):
    """Проверить, что значение присутствует в обеих версиях статьи."""
    if value is None:
        CHECKS.append((name, "НЕ НАЙДЕНО в results.txt", False))
        return
    miss = []
    shown = None
    for tag, tex, sep in VERSIONS:
        lit = literal(value, digits, sep)
        if shown is None:
            shown = lit
        # Число должно стоять как самостоятельная величина: не быть частью
        # более длинного числа и не примыкать к другому десятичному разделителю.
        tail = r"(?![\d]|\{,\})" if sep == "{,}" else r"(?![\d])"
        if not re.search(r"(?<![\d])" + re.escape(lit) + tail, tex):
            miss.append(tag)
    CHECKS.append((name + (" [нет в %s]" % ",".join(miss) if miss else ""),
                   shown, not miss))


# ── основные величины ────────────────────────────────────────────────
check("M (число событий)", res_num(r"M = (\d+), S = "), 0)
check("S (экспозиция)", res_num(r"M = \d+, S = ([\d.]+)"), 4)
check("kappa~ = (M-1)/S", res_num(r"схема A: kappa~ = \(M-1\)/S = ([\d.]+)"), 4)
check("ДИ схемы A, низ", res_num(r"95% ДИ = \[([\d.]+); [\d.]+\], p"), 4)
check("ДИ схемы A, верх", res_num(r"95% ДИ = \[[\d.]+; ([\d.]+)\], p"), 4)
check("kappa^ = (M-B)/S", res_num(r"схема B: kappa\^ = \(M-B\)/S = ([\d.]+)"), 4)
check("ДИ схемы B, низ", res_num(r"kappa\^_B = [\d.]+, 95% ДИ = \[([\d.]+); "), 3)
check("ДИ схемы B, верх", res_num(r"kappa\^_B = [\d.]+, 95% ДИ = \[[\d.]+; ([\d.]+)\]"), 3)
check("p схемы B", res_num(r"kappa\^_B = .*p = ([\d.]+)"), 3)

# ── таблица 4: ошибка I рода ─────────────────────────────────────────
err1 = res_num(r"B, сдвиг 0\.0\s+([\d.]+)")
check("ошибка I рода при схеме B", err1, 4)
if err1 is not None:
    raw = "%.1f" % (100 * err1)
    miss = both("", "$" + raw.replace(".", "{,}") + "\\%$", "$" + raw + "\\%$")
    CHECKS.append(("ошибка I рода в тексте (%s\\%%)" % raw
                   + (" [нет в %s]" % ",".join(miss) if miss else ""),
                   raw, not miss))

# ── таблица 5 / §3.4 ─────────────────────────────────────────────────
check("G_end для b=18", res_num(r"b=18: G_end=([\d.]+)"), 4)
check("G_OLS для b=18", res_num(r"b=18: G_end=[\d.]+ \(ранг \d+ из \d+\), G_OLS=([\d.]+)"), 4)
rank_end = res_num(r"b=18: G_end=[\d.]+ \(ранг (\d+) из", 1)
rank_ols = res_num(r"b=18: .*G_OLS=[\d.]+ \(ранг (\d+) из", 1)
CHECKS.append(("ранг b=18 по концевой форме = 3", "3",
               rank_end == 3
               and not both("", "третье по величине", "the third largest")))
CHECKS.append(("ранг b=18 по МНК = 10", "10",
               rank_ols == 10
               and not both("", "десятое место", "tenth of fifteen")))

# ── §5.3 ─────────────────────────────────────────────────────────────
check("Lambda однородности", res_num(r"Lambda = ([\d.]+) при 14"), 2)

# ── §6 ───────────────────────────────────────────────────────────────
check("мощность 80%: только Мерсенн", res_num(r"только Мерсенн \(M=\d+\): мощность 80% против отклонения \+([\d.]+)%"), 1)
check("мощность 80%: объединение", res_num(r"объединение \(M=\d+\): мощность 80% против отклонения \+([\d.]+)%"), 1)

# ── §7 ───────────────────────────────────────────────────────────────
check("наклон МНК", res_num(r"наклон = ([\d.]+) \(SE"), 4)
check("свободный член", -abs(res_num(r"свободный член = -([\d.]+)")), 4)
check("SE МНК", res_num(r"свободный член = -[\d.]+ \(SE ([\d.]+)\)"), 4)
check("SE Ньюи-Уэста", res_num(r"SE свободного члена ([\d.]+) \(было"), 4)
check("rho_1", res_num(r"rho_1 = ([\d.]+)"), 3)
check("max|r|", res_num(r"max\|r\| наблюдённое ([\d.]+)"), 3)
check("медиана max|r|", res_num(r"медиана симуляций ([\d.]+), p = [\d.]+\n"), 3)

# ── §7.2 ─────────────────────────────────────────────────────────────
check("lambda^ для b=2", res_num(r"lambda\^ = ([\d.]+) против"), 3)
check("kappa^_2", res_num(r"kappa\^_2 = ([\d.]+)"), 3)
check("kappa~_2 на срезе 50", res_num(r"условная оценка на том же срезе \(50 показателей\): kappa~_2 = ([\d.]+)"), 3)

# ── §8 ───────────────────────────────────────────────────────────────
for m in ("A", "B", "C", "D"):
    v = res_num(r"^\s+%s\s+\d+\s+(-[\d.]+)" % m, 1, re.M)
    check("ln L модели %s" % m, v, 3)
check("kappa^ при C^fix", res_num(r"C\^fix: среднее C_b = [\d.]+, kappa\^ = ([\d.]+)"), 3)
check("лог-поправка kappa^",
      res_num(r"Логарифмические поправки.*?kappa\^ = ([\d.]+)", 1, re.S), 3)
check("лог-поправка c1",
      res_num(r"Логарифмические поправки.*?c1\^ = \+([\d.]+)", 1, re.S), 3)
check("лог-поправка LR",
      res_num(r"Логарифмические поправки.*?LR = ([\d.]+)", 1, re.S), 2)

# ── проверка независимого пересчёта ──────────────────────────────────
ver = HERE / "verify"
if (ver / "b2.json").exists():
    import json
    tot = 0
    for f in sorted(ver.glob("b*.json")):
        tot += len(json.loads(f.read_text())["prp_exponents"])
    CHECKS.append(("подтверждённых членов при пересчёте = %d" % tot,
                   str(tot),
                   not both("", "$%d$ члена подтверждены" % tot,
                            "$%d$ terms confirmed" % tot)))


def main():
    bad = 0
    print("%-46s %-14s %s" % ("величина", "значение", "в статье"))
    print("-" * 72)
    for name, val, ok in CHECKS:
        print("%-46s %-14s %s" % (name, val, "да" if ok else "НЕТ"))
        if not ok:
            bad += 1
    print("-" * 72)
    print("расхождений: %d из %d" % (bad, len(CHECKS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
