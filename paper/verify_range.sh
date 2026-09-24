#!/usr/bin/env bash
# Сплошной пересчёт R_b(n) = (b^n-1)/(b-1) по ВСЕМ простым n из [N_FROM, N_TO)
# для одного основания — продолжение verify_sequences.sh за границу n < 10^4.
#
# Запуск:
#   bash paper/verify_range.sh BASE N_FROM N_TO [--double-check]
#
#   --double-check   пересчитать ВСЕ «составные» вердикты GWNUM вторым,
#                    независимым бэкендом (GMP): доля 1.0 вместо DC (0.05).
#                    Внимание: GMP в ~20 раз медленнее GWNUM, см. VERIFY_EXTENDED.md.
#
# Настройки поиска — те же, что у verify_sequences.sh: конфиг по умолчанию
# (сито 2^27, GPU trial factoring, PRP auto: GMP < 2500 бит <= GWNUM),
# --double-check ${DC:-0.05}. Единственное отличие — диапазон [N_FROM, N_TO)
# вместо [2, KMAX) и то, что он режется на куски.
#
# Возобновляемость. Диапазон режется на куски по CHUNK показателей (границы
# кратны CHUNK). Каждый кусок считается в своём рабочем каталоге
# $RH_WORK/b<b>/<a>_<c>/; если прогон прерван, повторный запуск продолжает
# кусок по worklog.jsonl, а законченные куски пропускает. Кусок считается
# законченным, только если в журнале есть вердикт для КАЖДОГО простого n
# куска (делитель / составное / PRP) и нет ни одной записи "failed"; тогда
# пишется маркер paper/verify_ext/chunks/b<b>/<a>_<c>.json.
#
# Результат (когда готовы все куски диапазона):
#   paper/verify_ext/<b>_<from>_<to>.txt   — b-файл-подобный список найденных
#                                            PRP ("i n"), шапка "#" с границами;
#   paper/verify_ext/<b>_<from>_<to>.json  — схема results.json (base, k_min,
#                                            k_max, prp_exponents, ...), та же,
#                                            что читает compare_verify.py.
# Сверка: python3 paper/compare_verify_ext.py
#
# Переменные окружения:
#   DC=0.05          доля double-check (перекрывается флагом --double-check)
#   CHUNK=5000       размер куска по n
#   RH_BIN=...       путь к бинарю repunit-hunt (иначе ищется в
#                    $CARGO_TARGET_DIR, ~/rh-target, <repo>/target)
#   RH_WORK=~/rh-verify-ext   рабочие каталоги (лучше ext4, не /mnt/c)
#   RH_EXTRA_ARGS="--no-gpu --threads 8"   доп. флаги поисковику
#   RH_WSL_DISTRO=Ubuntu      дистрибутив WSL при запуске из Git Bash
#                    (из Git Bash скрипт ВСЕГДА перезапускается в WSL; пути в
#                    RH_BIN/RH_WORK тогда задавайте как пути внутри WSL)
#   RESTART=1        выбросить незаконченные рабочие каталоги и начать заново
#
# Коды выхода: 0 — весь диапазон закрыт; 1 — есть незаконченные куски;
#              2 — ошибка аргументов/окружения или занята блокировка.
set -u

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ── Аргументы ────────────────────────────────────────────────────────────
POS=()
DC_FLAG=0
for a in "$@"; do
  case "$a" in
    --double-check) DC_FLAG=1 ;;
    -h|--help) usage ;;
    *) POS+=("$a") ;;
  esac
done
[ ${#POS[@]} -eq 3 ] || usage
BASE=${POS[0]}; NFROM=${POS[1]}; NTO=${POS[2]}
for v in "$BASE" "$NFROM" "$NTO"; do
  case "$v" in ''|*[!0-9]*) echo "ОШИБКА: ожидались целые числа: $BASE $NFROM $NTO" >&2; exit 2 ;; esac
done
if [ "$BASE" -lt 2 ] || [ "$NFROM" -lt 2 ] || [ "$NTO" -le "$NFROM" ]; then
  echo "ОШИБКА: нужно BASE >= 2 и 2 <= N_FROM < N_TO" >&2; exit 2
fi

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")

# ── Git Bash / MSYS на Windows: поисковик собирается только под Linux ─────
# (нативный слой POSIX-only, см. README, раздел 3.1), поэтому перезапускаем
# этот же скрипт внутри WSL. Пути /c/... -> /mnt/c/...
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    # RH_BIN здесь — путь ВНУТРИ WSL (Linux-бинарь), он передаётся туда.
    # RH_NO_WSL=1 — не перезапускаться (если когда-нибудь появится
    # нативная Windows-сборка; сейчас её нет, см. README 3.1).
    if [ "${RH_NO_WSL:-}" != 1 ] && command -v wsl.exe >/dev/null 2>&1; then
      wrepo=$(printf '%s' "$REPO" | sed -E 's#^/([a-zA-Z])/#/mnt/\L\1/#')
      echo "Git Bash: перезапуск в WSL (${RH_WSL_DISTRO:-Ubuntu}), репозиторий $wrepo"
      # MSYS иначе «исправит» аргументы, похожие на пути, в C:/Program Files/Git/...
      export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
      fwd="DC='${DC:-}' CHUNK='${CHUNK:-}' RH_WORK='${RH_WORK:-}' RH_EXTRA_ARGS='${RH_EXTRA_ARGS:-}' RESTART='${RESTART:-}' RH_BIN='${RH_BIN:-}'"
      exec wsl.exe -d "${RH_WSL_DISTRO:-Ubuntu}" --exec bash -lc \
        "cd \"\$1\" && shift && env $fwd bash paper/verify_range.sh \"\$@\"" \
        _ "$wrepo" "$@"
    fi
    ;;
esac

# ── Окружение ────────────────────────────────────────────────────────────
[ -f /etc/profile.d/rust.sh ] && . /etc/profile.d/rust.sh
PY=$(command -v python3 || command -v python || true)
[ -n "$PY" ] || { echo "ОШИБКА: нужен python3" >&2; exit 2; }

BIN=${RH_BIN:-}
if [ -z "$BIN" ]; then
  for c in "${CARGO_TARGET_DIR:-/nonexistent}/release/repunit-hunt" \
           "$HOME/rh-target/release/repunit-hunt" \
           "$REPO/target/release/repunit-hunt"; do
    [ -x "$c" ] && { BIN=$c; break; }
  done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "ОШИБКА: бинарь repunit-hunt не найден. Соберите его (README, раздел 3.1):" >&2
  echo "  export CARGO_TARGET_DIR=~/rh-target; cargo build --release" >&2
  echo "или укажите путь: RH_BIN=/путь/к/repunit-hunt" >&2
  exit 2
fi

if [ "$DC_FLAG" = 1 ]; then DCR=1.0; else DCR=${DC:-0.05}; fi
[ -n "$DCR" ] || DCR=0.05
CHUNK=${CHUNK:-5000}
[ -n "$CHUNK" ] || CHUNK=5000
WORK=${RH_WORK:-$HOME/rh-verify-ext}
[ -n "$WORK" ] || WORK=$HOME/rh-verify-ext
EXTRA=${RH_EXTRA_ARGS:-}
OUT="$HERE/verify_ext"
MARK="$OUT/chunks/b$BASE"
mkdir -p "$MARK" "$WORK/b$BASE"

# ── Блокировка ───────────────────────────────────────────────────────────
# Тот же файл, что у verify_sequences.sh: оба прогона делят GPU и не должны
# идти одновременно. Где нет flock (MSYS), — каталог-замок с PID.
LOCKF="${TMPDIR:-/tmp}/rh-verify.lock"
if command -v flock >/dev/null 2>&1; then
  # Файл мог остаться от прогона под другим пользователем (root): тогда
  # открываем на чтение — flock работает и с таким дескриптором.
  if [ -e "$LOCKF" ] && [ ! -w "$LOCKF" ]; then exec 9<"$LOCKF"; else exec 9>>"$LOCKF"; fi
  if ! flock -n 9; then
    echo "ОШИБКА: другой пересчёт (verify_sequences.sh / verify_range.sh) уже идёт." >&2
    exit 2
  fi
else
  LOCKD="$LOCKF.d"
  if ! mkdir "$LOCKD" 2>/dev/null; then
    op=$(cat "$LOCKD/pid" 2>/dev/null || echo)
    if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then
      echo "ОШИБКА: другой пересчёт уже идёт (PID $op)." >&2; exit 2
    fi
    rm -rf "$LOCKD"; mkdir "$LOCKD"
  fi
  echo $$ >"$LOCKD/pid"
  trap 'rm -rf "$LOCKD"' EXIT
fi

VERSION=$("$BIN" --version 2>/dev/null | head -1)
echo "=== b=$BASE  n ∈ [$NFROM, $NTO)  куски по $CHUNK  double-check=$DCR"
echo "    бинарь: $BIN ($VERSION)"
echo "    рабочие каталоги: $WORK/b$BASE   маркеры: $MARK"
[ -n "$EXTRA" ] && echo "    доп. флаги: $EXTRA"

# ── Проверка/финализация куска (python) ───────────────────────────────────
# Читает worklog.jsonl, проверяет, что закрыт КАЖДЫЙ простой n из [a, c),
# и пишет маркер. Код 0 — кусок полон, 1 — нет.
finalize_chunk() {  # $1=a $2=c $3=workdir $4=marker $5=wall_secs_this_run
  "$PY" - "$BASE" "$1" "$2" "$3" "$4" "$5" "$DCR" "$VERSION" "$EXTRA" <<'PYEOF'
import json, os, re, sys, time
base, a, c = (int(x) for x in sys.argv[1:4])
wd, marker, wall, dcr, version, extra = sys.argv[4:10]
wall = float(wall)

def primes(lo, hi):
    s = bytearray([1]) * hi
    s[0:2] = b"\x00\x00"
    for i in range(2, int(hi ** 0.5) + 1):
        if s[i]:
            s[i*i::i] = bytearray(len(range(i*i, hi, i)))
    return [i for i in range(max(lo, 2), hi) if s[i]]

need = primes(a, c)
recs = {}
prp, failed = set(), set()
n_small = n_gpu = n_pm1 = n_comp = 0
prp_tests = 0
secs = 0.0
bits_secs = []           # (bits, secs) — для калибровки verify_plan.py
wl = os.path.join(wd, "worklog.jsonl")
if os.path.exists(wl):
    for line in open(wl, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k, st = r.get("k"), r.get("status")
        if not isinstance(k, int) or not (a <= k < c):
            continue
        recs.setdefault(k, set()).add(st)
        if st == "PRP":
            prp.add(k)
        elif st == "failed":
            failed.add(k)
        elif st == "composite":
            n_comp += 1
        elif st == "factored":
            stage = r.get("stage")
            n_small += stage == "small"
            n_gpu += stage == "gpu"
            n_pm1 += stage == "pm1"
        if st in ("PRP", "composite") and "secs" in r:
            prp_tests += 1
            secs += float(r["secs"])
            bits_secs.append((int(r.get("bits", 0)), round(float(r["secs"]), 4)))

missing = [p for p in need if p not in recs]
# «failed» закрывается только последующим настоящим вердиктом того же k
unresolved = sorted(k for k in failed if not (recs[k] & {"PRP", "composite", "factored"}))
alien = sorted(set(recs) - set(need))

# Сводки из run.log (он дописывается при каждом возобновлении)
dc_checked = dc_mism = gpu_fp = 0
log = os.path.join(wd, "run.log")
if os.path.exists(log):
    txt = open(log, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"Double-check: пересчитано (\d+), расхождений (\d+)", txt):
        dc_checked += int(m.group(1)); dc_mism += int(m.group(2))
    gpu_fp = len(re.findall(r"ЛОЖНОЕ СРАБАТЫВАНИЕ GPU", txt))

prev_wall = 0.0
st_path = os.path.join(wd, "wall_secs")
if os.path.exists(st_path):
    try:
        prev_wall = float(open(st_path).read().strip() or 0)
    except ValueError:
        pass
wall_total = prev_wall + wall
open(st_path, "w").write("%.1f" % wall_total)

ok = not missing and not unresolved and not alien
print("    простых n: %d | сито %d, GPU TF %d, P-1 %d | PRP-тестов %d "
      "(Σ %.0f с) | PRP: %s" % (len(need), n_small, n_gpu, n_pm1, prp_tests,
                               secs, sorted(prp) or "нет"))
if dc_checked or dc_mism:
    print("    double-check: пересчитано %d, расхождений %d" % (dc_checked, dc_mism))
if dc_mism:
    print("    ВНИМАНИЕ: основной бэкенд терял находку, исправлено пересчётом — "
          "см. run.log (DOUBLE-CHECK)")
if gpu_fp:
    print("    ВНИМАНИЕ: %d ложных срабатываний GPU TF — см. run.log" % gpu_fp)
if missing:
    print("    НЕ ЗАКРЫТО %d простых n (первые: %s)" % (len(missing), missing[:10]))
if unresolved:
    print("    НЕРЕШЁННЫЕ (status=failed): %s — журнал считает их закрытыми, "
          "поэтому перезапустите кусок с RESTART=1" % unresolved)
if alien:
    print("    ОШИБКА: в журнале n вне куска или не простые: %s" % alien[:10])
if not ok:
    sys.exit(1)

rep = {
    "base": base, "k_min": a, "k_max": c,
    "prp_exponents": sorted(prp),
    "note": "PRP, не доказательство простоты — требуется ECPP/Primo",
    "complete": True,
    "primes_checked": len(need),
    "factored_small": n_small, "factored_gpu": n_gpu, "factored_pm1": n_pm1,
    "prp_tests": prp_tests, "prp_secs_sum": round(secs, 2),
    "wall_secs": round(wall_total, 1),
    "double_check_ratio": float(dcr),
    "double_checked": dc_checked, "double_check_mismatches": dc_mism,
    "gpu_false_positives": gpu_fp,
    "searcher": version, "extra_args": extra,
    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    "prp_timings": bits_secs,
}
tmp = marker + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=1)
os.replace(tmp, marker)
PYEOF
}

# Маркер годен, если описывает ровно этот кусок и помечен complete.
marker_ok() {  # $1=marker $2=a $3=c
  [ -f "$1" ] || return 1
  "$PY" - "$1" "$2" "$3" "$BASE" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
ok = (r.get("complete") and r.get("base") == int(sys.argv[4])
      and r.get("k_min") == int(sys.argv[2]) and r.get("k_max") == int(sys.argv[3]))
sys.exit(0 if ok else 1)
PYEOF
}

# ── Основной цикл по кускам ──────────────────────────────────────────────
done_n=0; todo_n=0; fail_n=0
a=$NFROM
while [ "$a" -lt "$NTO" ]; do
  c=$(( (a / CHUNK + 1) * CHUNK ))
  [ "$c" -gt "$NTO" ] && c=$NTO
  tag="${a}_${c}"
  mk="$MARK/$tag.json"
  wd="$WORK/b$BASE/$tag"
  if marker_ok "$mk" "$a" "$c"; then
    echo "--- [$a, $c): уже закрыт, пропускаю"
    done_n=$((done_n+1)); a=$c; continue
  fi
  todo_n=$((todo_n+1))
  [ "${RESTART:-}" = 1 ] && rm -rf "$wd"
  mkdir -p "$wd"
  # Возобновление допустимо только с теми же настройками.
  want="dc=$DCR extra=$EXTRA"
  if [ -f "$wd/settings" ] && [ "$(cat "$wd/settings")" != "$want" ]; then
    echo "ОШИБКА: $wd начат с другими настройками ($(cat "$wd/settings")), сейчас: $want" >&2
    echo "        Запустите с RESTART=1, чтобы начать кусок заново." >&2
    fail_n=$((fail_n+1)); a=$c; continue
  fi
  echo "$want" >"$wd/settings"
  resumed=""; [ -s "$wd/worklog.jsonl" ] && resumed=" (возобновление)"
  echo "--- [$a, $c)$resumed  $(date '+%F %T')"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  ( cd "$wd" && "$BIN" --base "$BASE" --kmin "$a" --kmax "$c" \
        --double-check "$DCR" $EXTRA >>run.log 2>&1 )
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    echo "    ОШИБКА поисковика (код $rc), хвост $wd/run.log:"
    tail -5 "$wd/run.log" | sed 's/^/      /'
    fail_n=$((fail_n+1)); a=$c; continue
  fi
  if finalize_chunk "$a" "$c" "$wd" "$mk" "$((t1-t0))"; then
    echo "    закрыт за $((t1-t0)) с"
    done_n=$((done_n+1))
  else
    fail_n=$((fail_n+1))
  fi
  a=$c
done

# ── Сборка итогового файла диапазона ─────────────────────────────────────
if [ "$fail_n" -ne 0 ]; then
  echo "=== b=$BASE: не закрыто кусков: $fail_n (закрыто $done_n). Повторный запуск продолжит."
  exit 1
fi

"$PY" - "$BASE" "$NFROM" "$NTO" "$CHUNK" "$MARK" "$OUT" "$DCR" <<'PYEOF'
import json, os, sys, time
base, a, c, chunk = (int(x) for x in sys.argv[1:5])
mark, out, dcr = sys.argv[5], sys.argv[6], sys.argv[7]
parts, x = [], a
while x < c:
    y = min((x // chunk + 1) * chunk, c)
    parts.append(json.load(open(os.path.join(mark, "%d_%d.json" % (x, y)), encoding="utf-8")))
    x = y
prp = sorted(set(k for p in parts for k in p["prp_exponents"]))
tot = lambda key: sum(p.get(key, 0) for p in parts)
stem = os.path.join(out, "%d_%d_%d" % (base, a, c))
rep = {
    "base": base, "k_min": a, "k_max": c, "prp_exponents": prp,
    "note": "PRP, не доказательство простоты — требуется ECPP/Primo",
    "complete": True, "chunks": len(parts),
    "primes_checked": tot("primes_checked"), "prp_tests": tot("prp_tests"),
    "prp_secs_sum": round(tot("prp_secs_sum"), 1), "wall_secs": round(tot("wall_secs"), 1),
    "double_check_ratio": min(p.get("double_check_ratio", 0) for p in parts),
    "double_checked": tot("double_checked"),
    "double_check_mismatches": tot("double_check_mismatches"),
    "gpu_false_positives": tot("gpu_false_positives"),
    "searcher": parts[-1].get("searcher", ""),
    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(stem + ".json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
with open(stem + ".txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("# repunit-hunt: сплошной пересчёт R_b(n) = (b^n-1)/(b-1), b = %d\n" % base)
    f.write("# проверены ВСЕ простые n из [%d, %d); ниже — все найденные PRP\n" % (a, c))
    f.write("# k_min %d\n# k_max %d\n# double_check %s\n" % (a, c, rep["double_check_ratio"]))
    f.write("# простых n %d, PRP-тестов %d, Σ PRP %.0f с, стена %.0f с, кусков %d\n"
            % (rep["primes_checked"], rep["prp_tests"], rep["prp_secs_sum"],
               rep["wall_secs"], rep["chunks"]))
    f.write("# double-check: пересчитано %d, расхождений %d; ложных GPU %d\n"
            % (rep["double_checked"], rep["double_check_mismatches"], rep["gpu_false_positives"]))
    f.write("# %s, %s\n" % (rep["searcher"], rep["finished"]))
    for i, k in enumerate(prp, 1):
        f.write("%d %d\n" % (i, k))
print("=== b=%d  [%d, %d) закрыт полностью: PRP %s" % (base, a, c, prp or "нет"))
print("    %s.txt / .json" % stem)
PYEOF
exit 0
