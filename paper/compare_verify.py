#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сверка независимого пересчёта (repunit-hunt) с записями OEIS.

Проверяется не только совпадение перечисленных членов, но и ОТСУТСТВИЕ
пропусков: поисковик закрывает все индексы n < KMAX подряд, поэтому любой
не отмеченный в OEIS индекс всплыл бы как лишний PRP.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VER = HERE / "verify"

SEQ = {
    2: "A000043", 3: "A028491", 5: "A004061", 6: "A004062", 7: "A004063",
    10: "A004023", 11: "A005808", 12: "A004064", 13: "A016054", 14: "A006032",
    15: "A006033", 17: "A006034", 18: "A133857", 19: "A006035", 20: "A127995",
    21: "A127996", 22: "A127997", 23: "A204940", 24: "A127998", 26: "A127999",
}


def oeis_terms(sid):
    p = DATA / (sid + ".txt")
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line.split()[-1]))
    return sorted(out)


def main():
    rows = []
    tot_conf = tot_extra = tot_miss = 0
    for b in sorted(SEQ):
        f = VER / ("b%d.json" % b)
        if not f.exists():
            rows.append((b, None, None, None, None, "не запускалось"))
            continue
        rep = json.loads(f.read_text())
        kmax = rep["k_max"]
        found = set(rep["prp_exponents"])
        ref = set(x for x in oeis_terms(SEQ[b]) if x < kmax)
        extra = sorted(found - ref)
        miss = sorted(ref - found)
        status = "совпало" if not extra and not miss else "РАСХОЖДЕНИЕ"
        rows.append((b, SEQ[b], kmax, len(ref), len(found), status))
        tot_conf += len(ref & found)
        tot_extra += len(extra)
        tot_miss += len(miss)
        if extra:
            print("  b=%d ЛИШНИЕ (нет в OEIS): %s" % (b, extra))
        if miss:
            print("  b=%d ПРОПУЩЕНЫ поисковиком: %s" % (b, miss))

    print()
    print("%4s %9s %9s %10s %10s %s" %
          ("b", "OEIS", "n < KMAX", "членов OEIS", "найдено", "статус"))
    for r in rows:
        if r[1] is None:
            print("%4d %9s %9s %10s %10s %s" % (r[0], "-", "-", "-", "-", r[5]))
        else:
            print("%4d %9s %9d %10d %10d %s" % r)
    print()
    print("Итого: подтверждено %d членов, лишних %d, пропущенных %d"
          % (tot_conf, tot_extra, tot_miss))
    return 1 if (tot_extra or tot_miss) else 0


if __name__ == "__main__":
    sys.exit(main())
