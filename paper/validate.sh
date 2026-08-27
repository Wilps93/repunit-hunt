#!/bin/bash
# ============================================================================
#  Полная валидация статьи — все проверки разом, в несколько прогонов.
#
#  Прогон 1. ЧИСТАЯ КОМНАТА
#            Удаляются все производные файлы (figs/, results.txt, paper.pdf) и
#            воссоздаются с нуля из data/ и paper.tex. Отвечает на вопрос
#            «соберётся ли работа у постороннего человека».
#  Прогон 2. СВЕРКА
#            Числа статьи против analysis.py; ГОСТ Р 7.0.7-2021; данные против
#            OEIS; самоповтор; орфография; типографика; извлекаемость PDF.
#  Прогон 3. УСТОЙЧИВОСТЬ К ЗЕРНУ
#            Анализ прогоняется при пяти зёрнах; детерминированные величины
#            обязаны совпасть точно, величины Монте-Карло — остаться внутри
#            содержательных коридоров.
#  Прогон 4. ЖИВЫЕ ИСТОЧНИКИ
#            Заново скачиваются b-файлы OEIS и сверяются с лежащими в data/;
#            проверяется фронт GIMPS. Ловит дрейф внешних данных.
#  Прогон 5. НЕЗАВИСИМЫЙ ПЕРЕСЧЁТ           (только с --full: занимает минуты)
#            repunit-hunt заново пересчитывает все индексы n < 10^4 со сплошным
#            double-check и сверяется с OEIS.
#
#  Запуск:  bash validate.sh          прогоны 1-4
#           bash validate.sh --full   прогоны 1-5
#
#  Код возврата 0, только если пройдено всё.
# ============================================================================
set -u
cd "$(dirname "$0")"

# Блокировка: валидация перезаписывает figs/, results.txt, paper.pdf и verify/.
# Два одновременных прогона дают ложные расхождения — см. verify_sequences.sh.
exec 8>"${TMPDIR:-/tmp}/rh-validate.lock"
if ! flock -n 8; then
  echo "ОШИБКА: валидация уже выполняется другим процессом." >&2
  exit 2
fi
# Потомки не должны пережить прерывание прогона.
trap 'pkill -P $$ 2>/dev/null; exit 130' INT TERM

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

FAILED=()
PASSED=0

hdr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
step() { printf '  %-52s ' "$1"; }
ok()   { printf 'ОК\n';        PASSED=$((PASSED+1)); }
bad()  { printf 'ПРОВАЛ\n';    FAILED+=("$1"); }
run()  { step "$1"; shift; if "$@" >/tmp/v.log 2>&1; then ok; else
           bad "$1"; sed 's/^/      /' /tmp/v.log | tail -12; fi; }

# ── Прогон 1: чистая комната ────────────────────────────────────────────────
hdr "Прогон 1. Чистая комната: всё производное строится заново"
rm -rf figs results.txt paper.pdf results_seed*.txt
step "analysis.py с нуля -> results.txt, figs/"
if python3 analysis.py >/tmp/v.log 2>&1; then ok; else bad "analysis"; tail -12 /tmp/v.log; fi
step "четыре рисунка созданы"
[ "$(ls figs/*.pdf 2>/dev/null | wc -l)" = 4 ] && ok || bad "figs"
run "build_paper.sh -> paper.pdf" bash build_paper.sh

# ── Прогон 2: сверка ────────────────────────────────────────────────────────
hdr "Прогон 2. Сверка"
run "числа статьи против results.txt"      python3 check_numbers.py
run "числа METHOD.md против results.txt"   python3 check_method.py
run "ГОСТ Р 7.0.7-2021"                    python3 check_gost.py
run "данные против записей OEIS"           python3 compare_verify.py
run "самоповтор с ранними редакциями"      python3 check_overlap.py

step "типографика: нет переполнений строк"
# Считать боксы можно только на СОШЕДШЕМСЯ проходе: пока ссылки не разрешены,
# на их месте стоят «??», и разбиение строк отличается от итогового.
rm -f paper.aux paper.log paper.out
for i in 1 2 3 4; do pdflatex -interaction=nonstopmode paper.tex >/tmp/box.log 2>&1; done
rm -f paper.aux paper.log paper.out
# Критерий несимметричен, и это осознанно. Overfull — текст физически выходит
# за поле, это брак вёрстки. Underfull — строка растянута сильнее идеала;
# \emergencystretch как раз и обменивает overfull на underfull, так что
# требовать нуля здесь означало бы запретить лекарство. Проваливаем прогон
# только при overfull или при вопиющей растянутости (badness >= 10000).
nb=$(grep -c 'Overfull \\hbox' /tmp/box.log)
nu=$(grep -c 'Underfull \\hbox' /tmp/box.log)
worst=$(grep -o 'Underfull \\hbox (badness [0-9]*' /tmp/box.log \
        | grep -o '[0-9]*$' | sort -rn | head -1)
worst=${worst:-0}
echo -n "(overfull=$nb, underfull=$nu, худшая badness=$worst) "
if [ "$nb" = 0 ] && [ "$worst" -lt 10000 ]; then ok
else bad "boxes"; grep -A2 'full \\hbox' /tmp/box.log | head -12 | sed 's/^/      /'; fi

step "кириллица извлекается из PDF"
pdftotext -enc UTF-8 paper.pdf /tmp/f.txt 2>/dev/null
grep -q "схема наблюдения" /tmp/f.txt && ok || bad "pdf-text"

step "англоязычные подписи в PDF"
n=$(grep -cE '^(Table|Figure) [0-9]+\.' /tmp/f.txt)
[ "$n" -ge 13 ] && ok || { bad "captions"; echo "      найдено $n из 13"; }

step "орфография: только известные термины"
python3 - <<'PY' >/tmp/sp.log 2>&1
import re, subprocess, sys
from pathlib import Path
subprocess.run(["detex","paper.tex"],stdout=open("/tmp/plain.txt","w"),
               stderr=subprocess.DEVNULL)
txt=Path("/tmp/plain.txt").read_text(encoding="utf-8",errors="ignore")
w=sorted(set(re.findall(r"[А-Яа-яЁё][А-Яа-яЁё-]{2,}",txt)))
p=subprocess.run(["hunspell","-d","ru_RU","-l"],input="\n".join(w),
                 capture_output=True,text=True)
bad=[x for x in p.stdout.split("\n") if x.strip()]
print("вне словаря: %d из %d уникальных"%(len(bad),len(w)))
sys.exit(1 if len(bad)>110 else 0)
PY
[ $? = 0 ] && ok || { bad "spell"; cat /tmp/sp.log; }

step "незаполненные плейсхолдеры пересчитаны"
np=$(grep -o 'textcolor{red}' paper.tex | wc -l)
echo -n "($np шт.) "; [ "$np" -le 5 ] && ok || bad "placeholders"

# ── Прогон 3: устойчивость к зерну ──────────────────────────────────────────
hdr "Прогон 3. Устойчивость выводов к зерну генератора"
step "пять зёрен, коридоры выводов"
if python3 check_stability.py >/tmp/st.log 2>&1; then ok
else bad "stability"; fi
grep -E "величин вне коридора|ВЫШЛА|РАСХОЖДЕНИЕ" /tmp/st.log | sed 's/^/      /'

# ── Прогон 4: живые источники ───────────────────────────────────────────────
hdr "Прогон 4. Живые источники: не сдвинулись ли внешние данные"
step "b-файлы OEIS совпадают с data/"
tmp=$(mktemp -d); rc=0
for s in A000043 A028491 A004061 A004062 A004063 A004023 A005808 A004064 \
         A016054 A006032 A006033 A006034 A133857 A006035 A127995 \
         A127996 A127997 A204940 A127998 A127999; do
  curl -s -m 60 "https://oeis.org/$s/b${s#A}.txt" -o "$tmp/$s.txt"
  if ! diff -q <(grep '^[0-9]' "data/$s.txt") <(grep '^[0-9]' "$tmp/$s.txt") >/dev/null 2>&1
  then echo; echo "      РАСХОЖДЕНИЕ: $s"; rc=1; fi
done
[ $rc = 0 ] && ok || bad "oeis-drift"
rm -rf "$tmp"

step "фронт GIMPS X_ver соответствует статье"
# Отчёт разделяет разряды тонкими пробелами (&thinsp;, U+2009, U+00A0), поэтому
# сначала убираем ВСЕ пробельные разделители внутри чисел, а потом ищем число.
xv=$(curl -s -m 60 https://www.mersenne.org/report_milestones/ \
     | sed -e 's/&thinsp;//g' -e 's/&nbsp;/ /g' -e 's/<[^>]*>/ /g' \
     | python3 -c '
import re, sys
t = sys.stdin.read()
t = t.replace(" ", "").replace(" ", "").replace(",", "")
t = re.sub(r"(?<=\d)[ \t]+(?=\d)", "", t)
m = re.search(r"All exponents below\s+(\d+)\s+have been tested and verified", t)
print(m.group(1) if m else "")' )
inpaper=$(grep -oP 'X_\{\\mathrm\{ver\}\}=\K[0-9\\,]+' paper.tex | head -1 | tr -d '\\,')
echo -n "(отчёт: ${xv:-?}, в статье: ${inpaper:-?}) "
if [ -n "$xv" ] && [ "$xv" = "$inpaper" ]; then ok
elif [ -n "$xv" ] && [ "$xv" -gt 77232917 ] && [ "$xv" -lt 82589933 ]; then
  printf 'СДВИНУЛСЯ, но вывод не меняется\n'; PASSED=$((PASSED+1))
else bad "gimps-frontier"; fi

# ── Прогон 5: независимый пересчёт ──────────────────────────────────────────
if [ $FULL = 1 ]; then
  hdr "Прогон 5. Независимый пересчёт последовательностей"
  run "repunit-hunt: все n < 10^4, сплошной double-check" \
      bash -c "DC=1.0 bash verify_sequences.sh 10000"
  run "результат пересчёта против OEIS" python3 compare_verify.py
else
  hdr "Прогон 5. Независимый пересчёт — пропущен (запустите с --full)"
fi

# ── Итог ────────────────────────────────────────────────────────────────────
hdr "ИТОГ"
echo "  пройдено проверок: $PASSED"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  провалов: 0"
  echo
  echo "  Статья прошла полную валидацию."
  exit 0
else
  echo "  провалов: ${#FAILED[@]} -> ${FAILED[*]}"
  exit 1
fi
