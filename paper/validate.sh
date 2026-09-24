#!/bin/bash
# ============================================================================
#  Полная валидация статьи (версия 3: paper_ru.tex, paper_en.tex).
#  Архив версии 1 проверяется отдельно: bash validate_v1.sh
#
#  Прогон 1. ЧИСТАЯ КОМНАТА
#            Удаляются производные файлы (figs_v2/, results_v2.txt, PDF) и
#            воссоздаются с нуля: analysis_v2.py, make_results_v3.sh,
#            build_paper.sh. Калибровка c_b [S19] берётся из кэша
#            results_S19.txt (с --full пересчитывается, около 10 минут).
#  Прогон 2. СВЕРКА
#            Каждое десятичное число обеих версий против вывода скриптов;
#            ключевые числа; ГОСТ Р 7.0.7-2021; данные против OEIS; самоповтор;
#            типографика; извлекаемость PDF (если есть pdftotext).
#  Прогон 3. УСТОЙЧИВОСТЬ К ЗЕРНУ
#            analysis_v2.py при трёх зёрнах: детерминированное совпадает
#            побитово, Монте-Карло остаётся в содержательных коридорах.
#  Прогон 4. ЖИВЫЕ ИСТОЧНИКИ
#            b-файлы OEIS и фронт GIMPS сегодня. Известный дрейф (новые члены,
#            уже учтённые в анализе; фронт GIMPS, ушедший вперёд) не считается
#            провалом, но печатается.
#  Прогон 5. НЕЗАВИСИМЫЙ ПЕРЕСЧЁТ           (только с --full)
#            verify_sequences.sh (n < 10^4) и сверка verify/ + verify_ext/.
#
#  Запуск:  bash validate.sh          прогоны 1-4
#           bash validate.sh --full   прогоны 1-5
#  Код возврата 0, только если пройдено всё.
# ============================================================================
set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
TMP="${TMPDIR:-/tmp}"

if command -v flock >/dev/null 2>&1; then
  exec 8>"$TMP/rh-validate.lock"
  flock -n 8 || { echo "ОШИБКА: валидация уже выполняется." >&2; exit 2; }
fi
trap 'pkill -P $$ 2>/dev/null; exit 130' INT TERM

FULL=0
[ "${1:-}" = "--full" ] && FULL=1
FAILED=()
PASSED=0
SKIPPED=0

hdr()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
step() { printf '  %-56s ' "$1"; }
ok()   { printf 'ОК\n'; PASSED=$((PASSED+1)); }
bad()  { printf 'ПРОВАЛ\n'; FAILED+=("$1"); }
skip() { printf 'пропущено (%s)\n' "$1"; SKIPPED=$((SKIPPED+1)); }
run()  { step "$1"; local name=$1; shift
         if "$@" >"$TMP"/v.log 2>&1; then ok; else
           bad "$name"; sed 's/^/      /' "$TMP"/v.log | tail -12; fi; }

# ── Прогон 1 ────────────────────────────────────────────────────────────────
hdr "Прогон 1. Чистая комната"
rm -rf figs_v2 results_v2.txt paper_ru.pdf paper_en.pdf results_v2_seed*.txt
run "analysis_v2.py -> results_v2.txt, figs_v2/" "$PY" analysis_v2.py
step "четыре рисунка в двух языковых наборах"
[ "$(ls figs_v2/*.pdf 2>/dev/null | wc -l)" = 4 ] &&
[ "$(ls figs_v2/en/*.pdf 2>/dev/null | wc -l)" = 4 ] && ok || bad "figs"
if [ $FULL = 1 ]; then
  run "make_results_v3.sh --full -> results_v3.txt" bash make_results_v3.sh --full
else
  run "make_results_v3.sh -> results_v3.txt" bash make_results_v3.sh
fi
run "build_paper.sh -> paper_ru.pdf, paper_en.pdf" bash build_paper.sh

# ── Прогон 2 ────────────────────────────────────────────────────────────────
hdr "Прогон 2. Сверка"
run "каждое десятичное число статьи против вывода"   "$PY" check_all_numbers_v3.py
run "ключевые числа обеих версий"                  "$PY" check_numbers_v3.py
run "ГОСТ Р 7.0.7-2021"                            "$PY" check_gost.py paper_ru.tex
run "пересчёт n < 10^4 против OEIS"                "$PY" compare_verify.py
run "расширенный пересчёт против OEIS"             "$PY" compare_verify_ext.py
run "самоповтор с ранними редакциями"              "$PY" check_overlap.py

for doc in paper_ru paper_en; do
  step "типографика $doc: нет переполнений строк"
  if ! command -v pdflatex >/dev/null 2>&1; then skip "нет pdflatex"; continue; fi
  rm -f $doc.aux $doc.log $doc.out
  for i in 1 2 3 4; do pdflatex -interaction=nonstopmode $doc.tex >"$TMP"/box.log 2>&1; done
  rm -f $doc.aux $doc.log $doc.out
  nb=$(grep -c 'Overfull \\hbox' "$TMP"/box.log)
  worst=$(grep -o 'Underfull \\hbox (badness [0-9]*' "$TMP"/box.log | grep -o '[0-9]*$' | sort -rn | head -1)
  worst=${worst:-0}
  echo -n "(overfull=$nb, худшая badness=$worst) "
  if [ "$nb" = 0 ] && [ "$worst" -lt 10000 ]; then ok; else bad "boxes-$doc"; fi
done

step "в английской версии нет остатков ГОСТ-аппарата"
grep -qE 'УДК|Научная статья|encaptab|Список источников|otherlanguage' paper_en.tex \
  && bad "gost-leak" || ok

step "текст извлекается из PDF (кириллица, лигатуры)"
if command -v pdftotext >/dev/null 2>&1; then
  pdftotext -enc UTF-8 paper_ru.pdf "$TMP"/f.txt 2>/dev/null
  pdftotext -enc UTF-8 paper_en.pdf "$TMP"/e.txt 2>/dev/null
  grep -q "схема наблюдения" "$TMP"/f.txt && grep -q "observation scheme" "$TMP"/e.txt \
    && grep -q "different" "$TMP"/e.txt && ok || bad "pdf-text"
else
  skip "нет pdftotext"
fi

step "нет незаполненных плейсхолдеров \\recompute"
n=$(grep -c '\\recompute{' paper_ru.tex paper_en.tex | awk -F: '{s+=$2} END {print s}')
[ "${n:-0}" = 0 ] && ok || { bad "placeholders"; echo "      найдено: $n"; }

# ── Прогон 3 ────────────────────────────────────────────────────────────────
hdr "Прогон 3. Устойчивость выводов к зерну"
run "три зерна, коридоры и побитовые совпадения" "$PY" check_stability_v3.py 1 2 3

# ── Прогон 4 ────────────────────────────────────────────────────────────────
hdr "Прогон 4. Живые источники"
step "b-файлы OEIS против data/"
"$PY" - <<'PY' >"$TMP"/oeis.log 2>&1
import subprocess, sys, urllib.request
from pathlib import Path
ids = "A000043 A028491 A004061 A004062 A004063 A004023 A005808 A004064 A016054 A006032 " \
      "A006033 A006034 A133857 A006035 A127995 A127996 A127997 A204940 A127998 A127999".split()
extra = {"A000043": {82589933, 136279841}}      # включены в анализ явно (analysis.py)
bad, known, fail = [], [], []
for s in ids:
    local = {int(l.split()[-1]) for l in Path("data/%s.txt" % s).read_text().splitlines()
             if l.strip() and not l.startswith("#")}
    # OEIS отвечает 403 на User-Agent по умолчанию у urllib
    req = urllib.request.Request("https://oeis.org/%s/b%s.txt" % (s, s[1:]),
                                 headers={"User-Agent": "Mozilla/5.0 (repunit-hunt validate.sh)"})
    try:
        txt = urllib.request.urlopen(req, timeout=60).read().decode()
    except Exception as e:
        fail.append(s); continue
    remote = {int(l.split()[-1]) for l in txt.splitlines() if l.strip() and not l.startswith("#")}
    if remote == local:
        continue
    if remote - local <= extra.get(s, set()) and local <= remote:
        known.append(s)
    else:
        bad.append((s, sorted(remote - local)[:5], sorted(local - remote)[:5]))
print("известный дрейф (новые члены уже учтены):", known)
print("не скачалось:", fail)
for b in bad: print("РАСХОЖДЕНИЕ:", b)
# несостоявшаяся загрузка ничего не говорит о данных: это не «ОК»
sys.exit(1 if bad else (3 if len(fail) == len(ids) else 0))
PY
rc=$?
if [ $rc = 0 ]; then ok; elif [ $rc = 3 ]; then skip "OEIS не скачался"; else bad "oeis-drift"; fi
sed 's/^/      /' "$TMP"/oeis.log

step "фронт GIMPS не ниже значения статьи"
xv=$(curl -s -m 60 https://www.mersenne.org/report_milestones/ \
     | sed -e 's/&thinsp;//g' -e 's/&nbsp;/ /g' -e 's/<[^>]*>/ /g' \
     | "$PY" -c '
import re, sys
t = sys.stdin.read().replace(" ", "").replace(" ", "").replace(",", "")
t = re.sub(r"(?<=\d)[ \t]+(?=\d)", "", t)
m = re.search(r"All exponents below\s+(\d+)\s+have been tested and verified", t)
print(m.group(1) if m else "")')
inpaper=$(grep -o 'X_{\\mathrm{ver}}=[0-9\\,]*' paper_ru.tex | head -1 | sed 's/.*=//; s/\\,//g')
echo -n "(отчёт: ${xv:-?}, статья: ${inpaper:-?}) "
if [ -n "$xv" ] && [ -n "$inpaper" ] && [ "$xv" -ge "$inpaper" ]; then ok
elif [ -z "$xv" ]; then skip "отчёт не скачался"
else bad "gimps-frontier"; fi

# ── Прогон 5 ────────────────────────────────────────────────────────────────
if [ $FULL = 1 ]; then
  hdr "Прогон 5. Независимый пересчёт"
  run "repunit-hunt: все n < 10^4, сплошной double-check" bash -c "DC=1.0 bash verify_sequences.sh 10000"
  run "результат пересчёта против OEIS" "$PY" compare_verify_ext.py
else
  hdr "Прогон 5. Независимый пересчёт — пропущен (запустите с --full)"
fi

hdr "ИТОГ"
echo "  пройдено: $PASSED, пропущено (нет инструмента/сети): $SKIPPED"
if [ ${#FAILED[@]} -eq 0 ]; then echo "  провалов: 0"; exit 0
else echo "  провалы: ${FAILED[*]}"; exit 1; fi
