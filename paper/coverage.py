#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Что именно проверено практическим пересчётом, а что — нет.

Сверка с OEIS подтверждает перечисленные члены; сплошной пересчёт подтверждает
ещё и ОТСУТСТВИЕ пропусков — но только ниже границы прогона. Скрипт показывает
эту границу по каждому основанию честно, включая слабые места.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, VER = HERE / "data", HERE / "verify"

SEQ = {
    2: "A000043", 3: "A028491", 5: "A004061", 6: "A004062", 7: "A004063",
    10: "A004023", 11: "A005808", 12: "A004064", 13: "A016054", 14: "A006032",
    15: "A006033", 17: "A006034", 18: "A133857", 19: "A006035", 20: "A127995",
}
MAIN = set(SEQ)
EXTRA = {21: "A127996", 22: "A127997", 23: "A204940",
         24: "A127998", 26: "A127999"}
A43_EXTRA = [82589933, 136279841]


def terms(sid, b):
    v = [int(l.split()[-1]) for l in (DATA / (sid + ".txt")).read_text().splitlines()
         if l.strip() and not l.startswith("#")]
    if b == 2:
        v = sorted(set(v) | set(A43_EXTRA))
    return sorted(v)


def main():
    print("Практический пересчёт: что закрыто, а что нет")
    print("=" * 86)
    print("%4s %9s %8s %9s %9s %9s  %s" %
          ("b", "OEIS", "всего", "проверено", "граница", "покрытие", "первый непроверенный"))
    print("-" * 86)
    tot_all = tot_ok = 0
    weak = []
    for b, sid in sorted({**SEQ, **EXTRA}.items()):
        t = terms(sid, b)
        f = VER / ("b%d.json" % b)
        kmax = json.loads(f.read_text())["k_max"] if f.exists() else 0
        under = [x for x in t if x < kmax]
        nxt = next((x for x in t if x >= kmax), None)
        frac = len(under) / len(t)
        mark = "  <-- основной набор" if b in MAIN else ""
        print("%4d %9s %8d %9d %9d %8.0f%%  %-12s%s" %
              (b, sid, len(t), len(under), kmax, 100 * frac,
               "{:,}".format(nxt).replace(",", " ") if nxt else "все", mark))
        if b in MAIN:
            tot_all += len(t)
            tot_ok += len(under)
            if len(under) <= 2:
                weak.append((b, len(under), len(t), nxt))
    print("-" * 86)
    print("Основной набор 2 <= b <= 20: подтверждено сплошным пересчётом %d "
          "из %d известных индексов (%.0f%%)."
          % (tot_ok, tot_all, 100 * tot_ok / tot_all))
    print()
    print("ВАЖНО: все пятнадцать допустимых оснований 2..20 пересчитаны, но")
    print("КАЖДОЕ — только до своей границы. Выше границы пропуск члена")
    print("пересчётом не исключён; там мы опираемся на OEIS.")
    if weak:
        print()
        print("Слабые места (подтверждено не более двух членов):")
        for b, u, n, nxt in weak:
            need = nxt
            bits = need * math.log2(b)
            print("  b = %-3d %d из %d; следующий индекс %s (~%s бит) --- "
                  "чтобы его закрыть, нужна граница %s"
                  % (b, u, n, "{:,}".format(need).replace(",", " "),
                     "{:,}".format(int(bits)).replace(",", " "),
                     "{:,}".format(need + 1).replace(",", " ")))
    print()
    print("Не проверялись практически: основания 4, 8, 9, 16 (точные степени).")
    print("Они исключены из модели алгебраически, до всякого счёта, поэтому")
    print("пересчёт по ним ничего не добавил бы к выводам работы.")


if __name__ == "__main__":
    main()
