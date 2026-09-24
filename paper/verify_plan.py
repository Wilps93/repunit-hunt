#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""План расширенного пересчёта: что считать дальше и сколько это займёт.

Для каждого из двадцати оснований печатает:
  * текущую СПЛОШНУЮ границу X (paper/verify/ + paper/verify_ext/, как в
    compare_verify_ext.py);
  * следующий диапазон: [X, 10^5), а если он закрыт, — [X, 3*10^5);
  * число простых показателей в нём и ожидаемое число PRP-тестов;
  * ОЦЕНКУ времени: процессорные часы и часы по стене — при double-check
    0.05 (как verify_sequences.sh по умолчанию) и 1.0 (--double-check).

Оценка строится по короткому замеру (--bench, ~1–2 мин):
  1. одиночные PRP-тесты на ОДНОМ потоке, сито и TF выключены, чтобы тест
     точно дошёл до PRP: GWNUM на трёх размерах (b=10), GMP на трёх размерах
     (GMP — второй бэкенд double-check'а), плюс по одному GWNUM-тесту на
     каждое основание (поправка на основание при том же числе бит);
  2. подгонка t = c * bits^alpha (в лог-координатах) отдельно для GWNUM и
     GMP; ориентир — PRP_EXP = 2.239 из src/tuner.rs (подгонка по 2756
     реальным замерам 20k..200k бит);
  3. калибровочный прогон поисковика с настройками verify_range.sh на
     b=10, n in [20000, 21000): из него — доля простых n, доживающих до PRP
     (остальные снимает сито и GPU TF), и эффективный параллелизм
     P = sum(t_1поток) / стена.
Результат замера кэшируется в paper/verify_ext/bench.json; без --bench
используются кэш, а если его нет — встроенные значения, измеренные на
машине автора (GTX 1650, 16 логических CPU, WSL2) 24.09.2026.

Модель: T_cpu(диапазон) = s * sum_p [ t_gw(bits_p)*f_b + r * t_gmp(bits_p) ]
          + sum_(известные члены OEIS в диапазоне) 3 * t_gmp(bits)   (GMP-перепроверка находок)
        T_стена = T_cpu / P
где s — доля доживших до PRP, r — доля double-check, f_b — поправка основания.
GPU trial factoring идёт параллельно с PRP (тюнер держит его дешевле
экономии на PRP) и отдельно не суммируется.

ВНИМАНИЕ: экстраполяция с ~130 тыс. бит на 1–1.4 млн бит (b=26, n=3*10^5) —
в 10 раз по размеру; ступени длины FFT в GWNUM и нехватка кэша при 16
одновременных тестах могут дать ошибку в 1.5–2 раза. Уточняйте по маркерам
законченных кусков (--from-markers).

Запуск:
  python3 paper/verify_plan.py                 # план по кэшу/встроенным данным
  python3 paper/verify_plan.py --bench         # сначала замер (~1–2 мин)
  python3 paper/verify_plan.py --dc 0.01 --threads-eff 6
  python3 paper/verify_plan.py --from-markers  # учесть фактические времена кусков
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:  # консоль Windows иначе падает на кириллице
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
sys.dont_write_bytecode = True  # не плодить paper/__pycache__
import compare_verify_ext as cve  # noqa: E402

BENCH_FILE = cve.EXT / "bench.json"
STAGES = (100_000, 300_000)
PRP_EXP_TUNER = 2.239

# Замер на машине автора (24.09.2026, WSL2 Ubuntu, 16 лог. CPU, GTX 1650),
# используется, пока нет paper/verify_ext/bench.json. См. --bench.
BUILTIN = {
    "note": "встроенные значения (одиночный замер 24.09.2026)",
    "gw_points": [[66472, 1.374]],
    "gmp_points": [[66472, 25.59]],
    "gw_alpha": PRP_EXP_TUNER, "gmp_alpha": 2.2,
    "base_factor": {},
    "survival": 0.48, "p_eff": 8.0,
}


# ─────────────────────────── запуск поисковика ───────────────────────────
FIND_BIN = r'''
[ -f /etc/profile.d/rust.sh ] && . /etc/profile.d/rust.sh
BIN=${RH_BIN:-}
if [ -z "$BIN" ]; then
  for c in "${CARGO_TARGET_DIR:-/nonexistent}/release/repunit-hunt" \
           "$HOME/rh-target/release/repunit-hunt" "__REPO__/target/release/repunit-hunt"; do
    [ -x "$c" ] && { BIN=$c; break; }
  done
fi
[ -x "$BIN" ] || { echo "NOBIN"; exit 3; }
'''


def repo_for_shell():
    repo = HERE.parent.as_posix()
    if os.name == "nt":  # C:/Users/... -> /mnt/c/Users/...
        m = re.match(r"^([A-Za-z]):/(.*)$", repo)
        if m:
            repo = "/mnt/%s/%s" % (m.group(1).lower(), m.group(2))
    return repo


def run_shell(script, timeout):
    """Выполнить bash-скрипт на Linux-стороне (на Windows — через WSL)."""
    script = FIND_BIN.replace("__REPO__", repo_for_shell()) + script
    if os.name == "nt":
        cmd = ["wsl.exe", "-d", os.environ.get("RH_WSL_DISTRO", "Ubuntu"),
               "--exec", "bash", "-s"]
    else:
        cmd = ["bash", "-s"]
    # байты, а не текст: на Windows текстовый stdin превратил бы \n в \r\n
    p = subprocess.run(cmd, input=script.encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    out = p.stdout.decode("utf-8", errors="replace")
    if "NOBIN" in out:
        sys.exit("бинарь repunit-hunt не найден — соберите его (README, 3.1) "
                 "или задайте RH_BIN")
    return out


def primes_upto(n):
    s = bytearray([1]) * n
    s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(range(i * i, n, i)))
    return [i for i in range(n) if s[i]]


PRIMES = primes_upto(STAGES[-1] + 1)


def prime_near(n):
    return next(p for p in PRIMES if p >= n)


def bits_of(b, n):
    return int((n - 1) * math.log2(b)) + 1


def single_tests(jobs):
    """jobs: [(base, n, backend)] -> [(base, n, backend, bits, secs)]."""
    lines = "\n".join("%d %d %s" % j for j in jobs)
    script = r'''
d=$(mktemp -d); cd "$d" || exit 4
while read base n be; do
  printf 'bitsieve_q_limit = 2\ntf_enabled = false\nprp_backend = "%s"\nthreads = 1\ndouble_check_ratio = 0.0\n' "$be" > c.toml
  rm -f worklog.jsonl results.json
  "$BIN" -c c.toml --base "$base" --kmin "$n" --kmax $((n+1)) >/dev/null 2>&1
  echo "BENCH $base $n $be $(tail -1 worklog.jsonl 2>/dev/null)"
done <<'EOF'
''' + lines + '''
EOF
cd /; rm -rf "$d"
'''
    out = run_shell(script, timeout=900)
    res = []
    for line in out.splitlines():
        if not line.startswith("BENCH "):
            continue
        _, b, n, be, js = line.split(" ", 4)
        try:
            r = json.loads(js)
            res.append((int(b), int(n), be, int(r["bits"]), float(r["secs"])))
        except (ValueError, KeyError):
            print("  замер не удался: b=%s n=%s %s" % (b, n, be))
    return res


def calibration_run(base=10, lo=20000, hi=21000):
    """Прогон с настройками verify_range.sh (double-check 0): доля доживших и стена."""
    script = r'''
d=$(mktemp -d); cd "$d" || exit 4
"$BIN" --base %d --kmin %d --kmax %d --double-check 0 >run.log 2>&1
grep -E "Малое сито готово|Готово за" run.log
echo "WORKLOG"; cat worklog.jsonl
cd /; rm -rf "$d"
''' % (base, lo, hi)
    out = run_shell(script, timeout=600)
    wall = sieve = None
    tests = []
    in_wl = False
    for line in out.splitlines():
        if line == "WORKLOG":
            in_wl = True
            continue
        if in_wl:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") in ("PRP", "composite"):
                tests.append((int(r["bits"]), float(r["secs"])))
            continue
        m = re.search(r"Малое сито готово за ([\d.]+)s", line)
        if m:
            sieve = float(m.group(1))
        m = re.search(r"Готово за ([\d.]+)s", line)
        if m:
            wall = float(m.group(1))
    nprimes = len([p for p in PRIMES if lo <= p < hi])
    return {"base": base, "range": [lo, hi], "primes": nprimes, "tests": tests,
            "wall": wall, "sieve": sieve}


def fit(points, alpha_default):
    """t = c * bits^alpha по точкам (bits, secs); alpha — подгонкой, если точек >= 2."""
    pts = [(math.log(b), math.log(t)) for b, t in points if t > 0]
    if len(pts) >= 2:
        mx = sum(x for x, _ in pts) / len(pts)
        my = sum(y for _, y in pts) / len(pts)
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        alpha = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else alpha_default
        # защита от шума на коротких тестах: разумный коридор
        alpha = min(max(alpha, 1.9), 2.6)
    else:
        alpha = alpha_default
    c = math.exp(sum(y - alpha * x for x, y in pts) / len(pts))
    return c, alpha


def do_bench():
    print("Замер: одиночные PRP-тесты (1 поток, без сита и TF) ...", flush=True)
    t0 = time.time()
    gw_series = [(10, prime_near(n), "gwnum") for n in (10000, 20000, 40000)]
    gmp_series = [(10, prime_near(n), "gmp") for n in (3000, 6000, 12000)]
    per_base = [(b, prime_near(15000), "gwnum") for b in sorted(cve.SEQ)]
    res = single_tests(gw_series + gmp_series + per_base)
    gw_pts = [(bits, s) for b, n, be, bits, s in res if be == "gwnum" and b == 10]
    gmp_pts = [(bits, s) for b, n, be, bits, s in res if be == "gmp"]
    c_gw, a_gw = fit(gw_pts, PRP_EXP_TUNER)
    c_gmp, a_gmp = fit(gmp_pts, 2.2)
    base_factor = {}
    for b, n, be, bits, s in res:
        if be == "gwnum" and n == prime_near(15000):
            base_factor[str(b)] = round(s / (c_gw * bits ** a_gw), 3)
    print("  готово за %.0f с; калибровочный прогон b=10, n in [20000, 21000) ..."
          % (time.time() - t0), flush=True)
    cal = calibration_run()
    surv = len(cal["tests"]) / cal["primes"] if cal["primes"] else BUILTIN["survival"]
    single = sum(c_gw * bits ** a_gw * base_factor.get("10", 1.0) for bits, _ in cal["tests"])
    busy = (cal["wall"] or 0) - (cal["sieve"] or 0)
    p_eff = single / busy if busy > 0 else BUILTIN["p_eff"]
    bench = {
        "note": "замер %s, %s" % (time.strftime("%Y-%m-%d %H:%M"),
                                 "WSL" if os.name == "nt" else os.uname().nodename),
        "gw_points": gw_pts, "gmp_points": gmp_pts,
        "gw_alpha": round(a_gw, 4), "gmp_alpha": round(a_gmp, 4),
        "base_factor": base_factor,
        "survival": round(surv, 3), "p_eff": round(p_eff, 2),
        "calibration": {k: v for k, v in cal.items() if k != "tests"},
        "calibration_tests": len(cal["tests"]),
        "bench_secs": round(time.time() - t0, 1),
    }
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps(bench, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  записано в %s" % BENCH_FILE.relative_to(HERE.parent))
    return bench


class Model:
    def __init__(self, bench, survival=None, p_eff=None):
        self.b = bench
        self.c_gw, self.a_gw = fit(bench["gw_points"], bench.get("gw_alpha", PRP_EXP_TUNER))
        if len(bench["gw_points"]) < 2:
            self.a_gw = bench.get("gw_alpha", PRP_EXP_TUNER)
            b0, t0 = bench["gw_points"][0]
            self.c_gw = t0 / b0 ** self.a_gw
        self.c_gmp, self.a_gmp = fit(bench["gmp_points"], bench.get("gmp_alpha", 2.2))
        if len(bench["gmp_points"]) < 2:
            self.a_gmp = bench.get("gmp_alpha", 2.2)
            b0, t0 = bench["gmp_points"][0]
            self.c_gmp = t0 / b0 ** self.a_gmp
        self.s = survival if survival is not None else bench["survival"]
        self.p = p_eff if p_eff is not None else bench["p_eff"]
        self.bf = {int(k): v for k, v in bench.get("base_factor", {}).items()}

    def t_gw(self, b, bits):
        return self.bf.get(b, 1.0) * self.c_gw * bits ** self.a_gw

    def t_gmp(self, bits):
        return self.c_gmp * bits ** self.a_gmp

    def estimate(self, b, lo, hi, dc, terms):
        ps = [p for p in PRIMES if lo <= p < hi]
        gw = gmp = 0.0
        for p in ps:
            bits = bits_of(b, p)
            gw += self.t_gw(b, bits)
            gmp += self.t_gmp(bits)
        hits = sum(3 * self.t_gmp(bits_of(b, x)) for x in terms if lo <= x < hi)
        cpu = self.s * (gw + dc * gmp) + hits
        return {"primes": len(ps), "tests": self.s * len(ps),
                "cpu_gw": self.s * gw, "cpu_dc_full": self.s * gmp,
                "cpu": cpu, "wall": cpu / self.p}


def markers_summary():
    """Фактические времена законченных кусков — для сверки модели."""
    rows = []
    for f in sorted(cve.EXT.glob("chunks/b*/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("prp_tests"):
            rows.append(r)
    return rows


def hours(s):
    h = s / 3600
    if h < 0.1:
        return "%4.0f мин" % (s / 60)
    if h < 48:
        return "%5.1f ч" % h
    return "%5.1f сут" % (h / 24)


def main():
    ap = argparse.ArgumentParser(description="План расширенного пересчёта")
    ap.add_argument("--bench", action="store_true", help="сделать замер (~1–2 мин) и сохранить")
    ap.add_argument("--dc", type=float, default=0.05, help="доля double-check (по умолчанию 0.05)")
    ap.add_argument("--survival", type=float, help="доля простых n, доходящих до PRP")
    ap.add_argument("--threads-eff", type=float, help="эффективный параллелизм P")
    ap.add_argument("--from-markers", action="store_true",
                    help="сравнить модель с фактическими временами законченных кусков")
    ap.add_argument("--prefix", action="append", help="каталог префикса (как в compare_verify_ext)")
    a = ap.parse_args()

    if a.bench:
        bench = do_bench()
    elif BENCH_FILE.exists():
        bench = json.loads(BENCH_FILE.read_text(encoding="utf-8"))
    else:
        bench = BUILTIN
    m = Model(bench, a.survival, a.threads_eff)

    print("Модель: %s" % bench.get("note", ""))
    print("  GWNUM: t = %.3g * bits^%.3f  (100k бит -> %.1f с, 1M бит -> %.0f с на 1 поток)"
          % (m.c_gw, m.a_gw, m.c_gw * 1e5 ** m.a_gw, m.c_gw * 1e6 ** m.a_gw))
    print("  GMP:   t = %.3g * bits^%.3f  (100k бит -> %.0f с, 1M бит -> %.0f с)  "
          "=> GMP/GWNUM ~ %.0fx при 300k бит"
          % (m.c_gmp, m.a_gmp, m.t_gmp(1e5), m.t_gmp(1e6),
             m.t_gmp(3e5) / (m.c_gw * 3e5 ** m.a_gw)))
    print("  доля доживших до PRP s = %.2f; эффективный параллелизм P = %.1f; "
          "double-check r = %.2f" % (m.s, m.p, a.dc))
    if m.bf:
        print("  поправки оснований f_b: " +
              " ".join("%d:%.2f" % (b, f) for b, f in sorted(m.bf.items())))
    print()

    recs = cve.load_records(a.prefix or ["verify"])
    plan = []
    print("%4s %9s  %-18s %7s %7s %11s %11s %11s   %s" %
          ("b", "сплошь<X", "следующий диапазон", "простых", "тестов",
           "CPU (r)", "стена (r)", "стена r=1", "членов OEIS в диапазоне"))
    print("-" * 118)
    for b in sorted(cve.SEQ):
        X, _ = cve.frontier(recs.get(b, []))
        terms = cve.oeis_terms(b)
        for stage_hi in STAGES:
            lo = max(X, STAGES[0]) if stage_hi == STAGES[1] else X
            if lo >= stage_hi:
                continue
            e = m.estimate(b, lo, stage_hi, a.dc, terms)
            e1 = m.estimate(b, lo, stage_hi, 1.0, terms)
            new = [x for x in terms if lo <= x < stage_hi]
            plan.append((b, lo, stage_hi, e, e1, new))
            print("%3d%s %9d  [%6d, %6d)   %7d %7.0f %11s %11s %11s   %s" %
                  (b, "*" if b in cve.MAIN else " ", X, lo, stage_hi, e["primes"],
                   e["tests"], hours(e["cpu"]), hours(e["wall"]), hours(e1["wall"]),
                   ", ".join(map(str, new)) or "-"))
    print("-" * 118)
    print("* — основной набор.  r = %.2f; «стена r=1» — с --double-check (все составные "
          "пересчитываются на GMP)." % a.dc)
    print()

    for stage_hi in STAGES:
        rows = sorted([p for p in plan if p[2] == stage_hi], key=lambda p: p[3]["wall"])
        if not rows:
            continue
        tot = sum(p[3]["wall"] for p in rows)
        tot1 = sum(p[4]["wall"] for p in rows)
        tot_main = sum(p[3]["wall"] for p in rows if p[0] in cve.MAIN)
        new = sum(len(p[5]) for p in rows)
        new_main = sum(len(p[5]) for p in rows if p[0] in cve.MAIN)
        print("Этап до n < %d: все 20 оснований — %s по стене (r=%.2f), %s при r=1; "
              "основной набор — %s. Закроет ещё %d членов OEIS (%d в основном наборе)."
              % (stage_hi, hours(tot), a.dc, hours(tot1), hours(tot_main), new, new_main))
        acc = 0.0
        order = []
        for p in rows:
            acc += p[3]["wall"]
            order.append("%d (%s)" % (p[0], hours(acc).strip()))
        print("  порядок (дешёвые первыми, нарастающим итогом): " + ", ".join(order))
        print()

    if a.from_markers:
        rows = markers_summary()
        if not rows:
            print("Законченных кусков с PRP-тестами пока нет.")
        else:
            print("Сверка модели с фактом (законченные куски):")
            print("%4s %-16s %7s %7s %10s %10s %10s %10s" %
                  ("b", "диапазон", "простых", "тестов", "s факт", "Σ t факт", "Σ t модель",
                   "стена"))
            for r in rows:
                b = r["base"]
                model = sum(m.t_gw(b, bits) for bits, _ in r.get("prp_timings", []))
                print("%4d [%6d,%6d) %7d %7d %10.2f %9.0fs %9.0fs %9.0fs" %
                      (b, r["k_min"], r["k_max"], r["primes_checked"], r["prp_tests"],
                       r["prp_tests"] / max(r["primes_checked"], 1), r["prp_secs_sum"],
                       model, r["wall_secs"]))
            print("(Σ t факт — время тестов под нагрузкой всех потоков; модель — на одном "
                  "потоке; их отношение показывает потерю от конкуренции за память.)")


if __name__ == "__main__":
    main()
