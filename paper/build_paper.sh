#!/bin/bash
# Сборка статьи. Рисунки должны быть заранее созданы: python3 analysis.py
set -e
cd "$(dirname "$0")"
[ -f figs/fig_kappa.pdf ] || { echo "нет рисунков — запустите analysis.py"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/tex1.log 2>&1 || {
  echo "ОШИБКИ LaTeX:"; grep -E '^!|^l\.[0-9]+' /tmp/tex1.log | head -20; exit 1; }

# Прогоняем до схождения перекрёстных ссылок, но не более пяти раз.
# Двух проходов НЕ хватает: номера в двуязычных подписях и ссылки на
# приложение стабилизируются только на третьем.
for i in 2 3 4 5; do
  pdflatex -interaction=nonstopmode paper.tex >/tmp/tex$i.log 2>&1
  grep -q "Rerun to get cross-references right" /tmp/tex$i.log || break
done
last=/tmp/tex$i.log
rm -f paper.aux paper.log paper.out
u=$(grep -cE "Reference .* undefined|Citation .* undefined" "$last" || true)
r=$(grep -c "Rerun to get cross-references right" "$last" || true)
echo "paper.pdf собран ($(wc -c < paper.pdf) байт) за $i прохода(ов);"
echo "  неразрешённых ссылок: $u, требуется пересчёт: $r"
[ "$u" = 0 ] && [ "$r" = 0 ] || exit 1
