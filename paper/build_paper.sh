#!/bin/bash
# Сборка статьи в обеих версиях: paper_ru.tex (ГОСТ Р 7.0.7-2021) и
# paper_en.tex (международный формат). Рисунки общие и должны быть созданы
# заранее: python3 analysis.py
#
#   bash build_paper.sh            обе версии
#   bash build_paper.sh paper_en   только указанную
set -e
cd "$(dirname "$0")"
[ -f figs/fig_kappa.pdf ] || { echo "нет рисунков — запустите analysis.py"; exit 1; }

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(paper_ru paper_en)

rc=0
for t in "${TARGETS[@]}"; do
  t=${t%.tex}
  [ -f "$t.tex" ] || { echo "нет файла $t.tex"; exit 1; }
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" >"/tmp/$t-1.log" 2>&1 || {
    echo "ОШИБКИ LaTeX в $t.tex:"
    grep -E '^!|^l\.[0-9]+' "/tmp/$t-1.log" | head -20; exit 1; }

  # Прогоняем до схождения перекрёстных ссылок, но не более пяти раз.
  # Двух проходов НЕ хватает: номера в двуязычных подписях и ссылки на
  # приложение стабилизируются только на третьем.
  for i in 2 3 4 5; do
    pdflatex -interaction=nonstopmode "$t.tex" >"/tmp/$t-$i.log" 2>&1
    grep -q "Rerun to get cross-references right" "/tmp/$t-$i.log" || break
  done
  last="/tmp/$t-$i.log"
  rm -f "$t.aux" "$t.log" "$t.out"
  u=$(grep -cE "Reference .* undefined|Citation .* undefined" "$last" || true)
  r=$(grep -c "Rerun to get cross-references right" "$last" || true)
  echo "$t.pdf собран ($(wc -c < "$t.pdf") байт) за $i прохода(ов);"
  echo "  неразрешённых ссылок: $u, требуется пересчёт: $r"
  [ "$u" = 0 ] && [ "$r" = 0 ] || rc=1
done
exit $rc
