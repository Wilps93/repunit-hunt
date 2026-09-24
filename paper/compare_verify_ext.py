#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сверка расширенного пересчёта (paper/verify_ext/) с b-файлами OEIS.

Дополняет compare_verify.py, который читает только paper/verify/b<b>.json
(один файл на основание, диапазон [2, KMAX)). Здесь диапазонов у основания
может быть много: итоговые файлы verify_range.sh (verify_ext/<b>_<a>_<c>.json)
и маркеры отдельных кусков (verify_ext/chunks/b<b>/<a>_<c>.json). Схема у
всех одна — как у results.json поисковика: base, k_min, k_max, prp_exponents.

Для каждого диапазона [k_min, k_max) проверяется:
  * ЛИШНИЕ  — PRP, найденные поисковиком, но отсутствующие в OEIS;
  * ПРОПУЩЕННЫЕ — члены OEIS из диапазона, которых поисковик не нашёл.
Кроме того, диапазоны объединяются с каноническим префиксом paper/verify/
(n < 10^4), и печатается СПЛОШНАЯ граница: наибольшее X, для которого все
n < X закрыты без дыр. Именно эта граница может стоять в статье.

Код выхода: 0 — расхождений нет; 1 — есть лишние или пропущенные;
2 — нет ни одного файла расширенного пересчёта И задан --require-ext
(без флага в этом случае сверяется только префикс, код 0 — так скрипт
можно звать из validate.sh на чистой копии репозитория).

Запуск:
  python3 paper/compare_verify_ext.py                 # все основания
  python3 paper/compare_verify_ext.py --base 7 --base 18
  python3 paper/compare_verify_ext.py --prefix verify_32k   # другой префикс
  python3 paper/compare_verify_ext.py --no-prefix
"""
import argparse
import json
import sys
from pathlib import Path

try:  # консоль Windows (cp866/cp1251) иначе падает на кириллице
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXT = HERE / "verify_ext"

SEQ = {
    2: "A000043", 3: "A028491", 5: "A004061", 6: "A004062", 7: "A004063",
    10: "A004023", 11: "A005808", 12: "A004064", 13: "A016054", 14: "A006032",
    15: "A006033", 17: "A006034", 18: "A133857", 19: "A006035", 20: "A127995",
    21: "A127996", 22: "A127997", 23: "A204940", 24: "A127998", 26: "A127999",
}
MAIN = {2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20}


def oeis_terms(b):
    """Члены OEIS — тем же разбором, что в compare_verify.py."""
    out = []
    for line in (DATA / (SEQ[b] + ".txt")).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(int(line.split()[-1]))
    return sorted(out)


def _rec(path, src):
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"base": int(r["base"]), "lo": int(r["k_min"]), "hi": int(r["k_max"]),
            "prp": set(int(x) for x in r["prp_exponents"]), "src": src,
            "complete": r.get("complete", True), "raw": r}


def load_records(prefix_dirs=("verify",), use_ext=True, ext=None):
    """Все диапазоны по основаниям: {b: [rec, ...]}."""
    ext_dir = Path(ext) if ext else EXT
    recs = {}
    for d in prefix_dirs:
        for f in sorted((HERE / d).glob("b*.json")):
            r = _rec(f, "%s/%s" % (d, f.name))
            recs.setdefault(r["base"], []).append(r)
    if use_ext and ext_dir.exists():
        for f in sorted(ext_dir.glob("*_*_*.json")):
            r = _rec(f, "verify_ext/" + f.name)
            recs.setdefault(r["base"], []).append(r)
        for f in sorted(ext_dir.glob("chunks/b*/*.json")):
            r = _rec(f, "verify_ext/" + f.relative_to(ext_dir).as_posix())
            recs.setdefault(r["base"], []).append(r)
    for b in recs:
        recs[b] = [r for r in recs[b] if r["complete"]]
    return recs


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def frontier(recs_b):
    """Сплошная граница: все n < X закрыты (X = 2, если префикса нет)."""
    m = merge([(r["lo"], r["hi"]) for r in recs_b])
    if m and m[0][0] <= 2:
        return m[0][1], m
    return 2, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=int, action="append", help="только эти основания")
    ap.add_argument("--prefix", action="append",
                    help="каталог(и) префикса относительно paper/ (по умолчанию verify)")
    ap.add_argument("--no-prefix", action="store_true", help="не учитывать префикс")
    ap.add_argument("--ext", help="каталог результатов вместо paper/verify_ext")
    ap.add_argument("--require-ext", action="store_true",
                    help="код 2, если в verify_ext/ нет ни одного диапазона")
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать каждый диапазон")
    a = ap.parse_args()

    prefix = [] if a.no_prefix else (a.prefix or ["verify"])
    recs = load_records(prefix, ext=a.ext)
    ext_count = sum(1 for rs in recs.values() for r in rs if r["src"].startswith("verify_ext"))
    if ext_count == 0:
        print("В verify_ext/ нет ни одного законченного диапазона "
              "(запустите paper/verify_range.sh); сверяется только префикс.")
        if a.require_ext:
            return 2

    bases = sorted(a.base) if a.base else sorted(SEQ)
    tot_conf_new = tot_conf_isl = 0
    all_extra, all_miss = set(), set()   # (b, n): один член — одно расхождение,
    # даже если он виден и в итоговом файле диапазона, и в маркере куска
    warn = []
    rows = []
    for b in bases:
        rs = recs.get(b, [])
        if b not in SEQ:
            print("b=%d: нет записи OEIS в SEQ — пропускаю" % b)
            continue
        ref_all = oeis_terms(b)
        found_cov, ref_cov = set(), set()
        bad = False
        for r in rs:
            ref = set(x for x in ref_all if r["lo"] <= x < r["hi"])
            outside = sorted(x for x in r["prp"] if not (r["lo"] <= x < r["hi"]))
            extra = sorted(r["prp"] - ref)
            miss = sorted(ref - r["prp"])
            if outside:
                print("  b=%d %s: PRP вне заявленного диапазона: %s" % (b, r["src"], outside))
                bad = True
            if extra:
                print("  b=%d %s: ЛИШНИЕ (нет в OEIS): %s" % (b, r["src"], extra))
                all_extra |= {(b, x) for x in extra}
                bad = True
            if miss:
                print("  b=%d %s: ПРОПУЩЕНЫ поисковиком: %s" % (b, r["src"], miss))
                all_miss |= {(b, x) for x in miss}
                bad = True
            if a.verbose:
                print("  b=%-3d [%d, %d) %-45s OEIS %d, найдено %d" %
                      (b, r["lo"], r["hi"], r["src"], len(ref), len(r["prp"])))
            found_cov |= r["prp"]
            ref_cov |= ref
            raw = r["raw"]
            if raw.get("double_check_mismatches"):
                warn.append("b=%d %s: double-check исправил %d вердикт(а) — см. run.log"
                            % (b, r["src"], raw["double_check_mismatches"]))
            if raw.get("gpu_false_positives"):
                warn.append("b=%d %s: %d ложных срабатываний GPU TF"
                            % (b, r["src"], raw["gpu_false_positives"]))
        X, merged = frontier(rs)
        pre = max((r["hi"] for r in rs if r["src"].startswith(tuple(d + "/" for d in prefix)) and r["lo"] <= 2),
                  default=2) if prefix else 2
        # подтверждён = есть в OEIS И найден поисковиком (пропущенный член
        # закрытого диапазона подтверждённым не считается)
        new_conf = len([x for x in ref_all if pre <= x < X and x in found_cov])
        tot_conf_new += new_conf
        tot_conf_isl += len([x for x in (ref_cov & found_cov) if x >= X])
        islands = [m for m in merged if m[0] > 2 and m[0] >= X]
        nxt = next((x for x in ref_all if x >= X), None)
        rows.append((b, SEQ[b], X, len([x for x in ref_all if x < X]), new_conf,
                     nxt, islands, "РАСХОЖДЕНИЕ" if bad else "совпало"))

    print()
    print("%4s %8s %9s %11s %9s %10s  %-11s %s" %
          ("b", "OEIS", "сплошь n<", "членов < X", "из них", "следующий", "статус",
           "отдельные закрытые участки выше X"))
    print("%4s %8s %9s %11s %9s %10s" % ("", "", "X", "", "новых", "член OEIS"))
    print("-" * 100)
    for b, sid, X, under, new, nxt, isl, st in rows:
        mark = "*" if b in MAIN else " "
        print("%3d%s %8s %9d %11d %9d %10s  %-11s %s" %
              (b, mark, sid, X, under, new, nxt if nxt else "-", st,
               ", ".join("[%d,%d)" % tuple(i) for i in isl) or ""))
    print("-" * 100)
    print("* — основной набор 2 <= b <= 20.  «новых» — члены OEIS, закрытые "
          "сверх префикса (%s)." % (", ".join(prefix) or "без префикса"))
    for w in warn:
        print("ВНИМАНИЕ: " + w)
    tot_extra, tot_miss = len(all_extra), len(all_miss)
    print()
    print("Итого: сверх префикса подтверждено %d членов сплошным пересчётом "
          "(+%d на отдельных участках выше X); лишних %d, пропущенных %d"
          % (tot_conf_new, tot_conf_isl, tot_extra, tot_miss))
    return 1 if (tot_extra or tot_miss) else 0


if __name__ == "__main__":
    sys.exit(main())
